## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.025922619999999997


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0137026, 0.0065541, -0.0137026, 0.0065541, -0.0202566, 0.0202566)
1: (-0.0093954, 0.0032056, -0.0093954, 0.0032056, -0.0126010, 0.0126010)
2: (0.0211217, 0.0614488, 0.0211217, 0.0614488, -0.0403271, 0.0403271)
3: (-0.0044447, 0.0131971, -0.0044447, 0.0131971, -0.0176418, 0.0176418)
4: (-0.0156401, 0.0125832, -0.0156401, 0.0125832, -0.0282234, 0.0282234)
5: (0.0007681, 0.0250758, 0.0007681, 0.0250758, -0.0243077, 0.0243077)
6: (-0.0382287, 0.0157631, -0.0382287, 0.0157631, -0.0539918, 0.0539918)
7: (0.9421708, 0.9810067, 0.9421708, 0.9810067, -0.0388359, 0.0388359)
8: (-0.0347546, 0.0243594, -0.0347546, 0.0243594, -0.0591140, 0.0591140)
9: (-0.0208363, 0.0202776, -0.0208363, 0.0202776, -0.0411139, 0.0411139)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.42 + 2.13 = 3.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
time: 1.20 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
time: 1.22 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.55 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.55
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.55
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0135362, 0.0062385, -0.0134245, 0.0060267, -0.0195629, 0.0196630
1: -0.0091716, 0.0030708, -0.0090213, 0.0029803, -0.0121518, 0.0120920
2: 0.0219048, 0.0609699, 0.0224306, 0.0606484, -0.0387436, 0.0385393
3: -0.0044226, 0.0128710, -0.0044077, 0.0126521, -0.0170747, 0.0172787
4: -0.0154531, 0.0122122, -0.0153276, 0.0119631, -0.0274163, 0.0275398
5: 0.0009511, 0.0248545, 0.0010739, 0.0247058, -0.0237548, 0.0237806
6: -0.0377070, 0.0153652, -0.0373568, 0.0150980, -0.0528051, 0.0527220
7: 0.9431276, 0.9809504, 0.9437699, 0.9809123, -0.0377847, 0.0371804
8: -0.0344453, 0.0235847, -0.0342377, 0.0230647, -0.0575100, 0.0578224
9: -0.0203396, 0.0198109, -0.0200061, 0.0194977, -0.0398373, 0.0398171

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
time: 1.44 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
time: 1.35 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0134801, 0.0061322, -0.0136953, 0.0065404, -0.0200205, 0.0198275
1: -0.0090961, 0.0030253, -0.0093857, 0.0031997, -0.0122958, 0.0124111
2: 0.0221688, 0.0608085, 0.0211557, 0.0614280, -0.0392592, 0.0396528
3: -0.0044151, 0.0127611, -0.0044437, 0.0131829, -0.0175981, 0.0172048
4: -0.0153901, 0.0120872, -0.0156320, 0.0125672, -0.0279573, 0.0277192
5: 0.0010127, 0.0247798, 0.0007761, 0.0250662, -0.0240535, 0.0240038
6: -0.0375312, 0.0152310, -0.0382060, 0.0157458, -0.0532770, 0.0534371
7: 0.9434501, 0.9809313, 0.9422123, 0.9810043, -0.0375542, 0.0387190
8: -0.0343411, 0.0233237, -0.0347412, 0.0243258, -0.0586668, 0.0580649
9: -0.0201722, 0.0196537, -0.0208147, 0.0202573, -0.0404295, 0.0404684

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
time: 1.18 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
time: 1.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.74 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.74
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.74
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.74
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.74
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0135362, 0.0062385, -0.0135362, 0.0062385, -0.0197747, 0.0197747
1: -0.0091716, 0.0030708, -0.0091716, 0.0030708, -0.0122423, 0.0122423
2: 0.0219048, 0.0609699, 0.0219048, 0.0609699, -0.0390651, 0.0390651
3: -0.0044226, 0.0128710, -0.0044226, 0.0128710, -0.0172936, 0.0172936
4: -0.0154531, 0.0122122, -0.0154531, 0.0122122, -0.0276654, 0.0276654
5: 0.0009511, 0.0248545, 0.0009511, 0.0248545, -0.0239034, 0.0239034
6: -0.0377070, 0.0153652, -0.0377070, 0.0153652, -0.0530722, 0.0530722
7: 0.9431276, 0.9809504, 0.9431276, 0.9809504, -0.0378227, 0.0378227
8: -0.0344453, 0.0235847, -0.0344453, 0.0235847, -0.0580301, 0.0580301
9: -0.0203396, 0.0198109, -0.0203396, 0.0198109, -0.0401505, 0.0401505

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0302991
time: 1.37 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
time: 1.26 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0135362, 0.0062385, -0.0134801, 0.0061322, -0.0196684, 0.0197186
1: -0.0091716, 0.0030708, -0.0090961, 0.0030253, -0.0121969, 0.0121669
2: 0.0219048, 0.0609699, 0.0221688, 0.0608085, -0.0389036, 0.0388011
3: -0.0044226, 0.0128710, -0.0044151, 0.0127611, -0.0171837, 0.0172861
4: -0.0154531, 0.0122122, -0.0153901, 0.0120872, -0.0275403, 0.0276023
5: 0.0009511, 0.0248545, 0.0010127, 0.0247798, -0.0238288, 0.0238417
6: -0.0377070, 0.0153652, -0.0375312, 0.0152310, -0.0529381, 0.0528964
7: 0.9431276, 0.9809504, 0.9434501, 0.9809313, -0.0378037, 0.0375003
8: -0.0344453, 0.0235847, -0.0343411, 0.0233237, -0.0577690, 0.0579258
9: -0.0203396, 0.0198109, -0.0201722, 0.0196537, -0.0399933, 0.0399831

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0304973
time: 1.23 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
time: 1.21 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0134801, 0.0061322, -0.0135362, 0.0062385, -0.0197186, 0.0196684
1: -0.0090961, 0.0030253, -0.0091716, 0.0030708, -0.0121669, 0.0121969
2: 0.0221688, 0.0608085, 0.0219048, 0.0609699, -0.0388011, 0.0389036
3: -0.0044151, 0.0127611, -0.0044226, 0.0128710, -0.0172861, 0.0171837
4: -0.0153901, 0.0120872, -0.0154531, 0.0122122, -0.0276023, 0.0275403
5: 0.0010127, 0.0247798, 0.0009511, 0.0248545, -0.0238417, 0.0238288
6: -0.0375312, 0.0152310, -0.0377070, 0.0153652, -0.0528964, 0.0529381
7: 0.9434501, 0.9809313, 0.9431276, 0.9809504, -0.0375003, 0.0378037
8: -0.0343411, 0.0233237, -0.0344453, 0.0235847, -0.0579258, 0.0577690
9: -0.0201722, 0.0196537, -0.0203396, 0.0198109, -0.0399831, 0.0399933

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0302991
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
time: 1.28 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0134801, 0.0061322, -0.0134801, 0.0061322, -0.0196123, 0.0196123
1: -0.0090961, 0.0030253, -0.0090961, 0.0030253, -0.0121214, 0.0121214
2: 0.0221688, 0.0608085, 0.0221688, 0.0608085, -0.0386397, 0.0386397
3: -0.0044151, 0.0127611, -0.0044151, 0.0127611, -0.0171762, 0.0171762
4: -0.0153901, 0.0120872, -0.0153901, 0.0120872, -0.0274773, 0.0274773
5: 0.0010127, 0.0247798, 0.0010127, 0.0247798, -0.0237671, 0.0237671
6: -0.0375312, 0.0152310, -0.0375312, 0.0152310, -0.0527623, 0.0527623
7: 0.9434501, 0.9809313, 0.9434501, 0.9809313, -0.0374812, 0.0374812
8: -0.0343411, 0.0233237, -0.0343411, 0.0233237, -0.0576648, 0.0576648
9: -0.0201722, 0.0196537, -0.0201722, 0.0196537, -0.0398259, 0.0398259

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0302991
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
time: 1.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.22 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0302991
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0304973
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0302991
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0302991
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.22
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0129387, 0.0051052, -0.0135021, 0.0061738, -0.0191125, 0.0186073
1: -0.0083676, 0.0025866, -0.0091257, 0.0030431, -0.0114107, 0.0117123
2: 0.0247174, 0.0592499, 0.0220655, 0.0608717, -0.0361542, 0.0371844
3: -0.0043432, 0.0117000, -0.0044180, 0.0128041, -0.0171473, 0.0161180
4: -0.0147815, 0.0108797, -0.0154148, 0.0121361, -0.0269176, 0.0262944
5: 0.0016080, 0.0240594, 0.0009886, 0.0248090, -0.0232010, 0.0230708
6: -0.0358335, 0.0139360, -0.0376000, 0.0152835, -0.0511170, 0.0515360
7: 0.9465639, 0.9807475, 0.9433239, 0.9809387, -0.0343748, 0.0374236
8: -0.0333346, 0.0208027, -0.0343819, 0.0234258, -0.0567604, 0.0551846
9: -0.0185557, 0.0181351, -0.0202376, 0.0197152, -0.0382709, 0.0383728

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0300605
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0134576, 0.0060894, -0.0135190, 0.0062059, -0.0196635, 0.0196084
1: -0.0090658, 0.0030071, -0.0091484, 0.0030568, -0.0121226, 0.0121555
2: 0.0222750, 0.0607436, 0.0219858, 0.0609203, -0.0386453, 0.0387577
3: -0.0044121, 0.0127169, -0.0044203, 0.0128373, -0.0172494, 0.0171372
4: -0.0153647, 0.0120369, -0.0154338, 0.0121738, -0.0275386, 0.0274707
5: 0.0010375, 0.0247498, 0.0009700, 0.0248316, -0.0237940, 0.0237799
6: -0.0374605, 0.0151771, -0.0376530, 0.0153240, -0.0527845, 0.0528301
7: 0.9435797, 0.9809236, 0.9432266, 0.9809444, -0.0373647, 0.0376970
8: -0.0342992, 0.0232186, -0.0344133, 0.0235046, -0.0578038, 0.0576320
9: -0.0201048, 0.0195904, -0.0202882, 0.0197627, -0.0398675, 0.0398786

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301192
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0135021, 0.0061738, -0.0129175, 0.0050649, -0.0185670, 0.0190912
1: -0.0091257, 0.0030431, -0.0083390, 0.0025694, -0.0116950, 0.0113821
2: 0.0220655, 0.0608717, 0.0248174, 0.0591888, -0.0371233, 0.0360543
3: -0.0044180, 0.0128041, -0.0043404, 0.0116584, -0.0160764, 0.0171445
4: -0.0154148, 0.0121361, -0.0147577, 0.0108323, -0.0262471, 0.0268938
5: 0.0009886, 0.0248090, 0.0016314, 0.0240312, -0.0230426, 0.0231777
6: -0.0376000, 0.0152835, -0.0357669, 0.0138852, -0.0514852, 0.0510505
7: 0.9433239, 0.9809387, 0.9466859, 0.9807403, -0.0374164, 0.0342528
8: -0.0343819, 0.0234258, -0.0332951, 0.0207038, -0.0550857, 0.0567209
9: -0.0202376, 0.0197152, -0.0184923, 0.0180756, -0.0383132, 0.0382075

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0302843
time: 1.23 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0294948
time: 1.37 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0135190, 0.0062059, -0.0133912, 0.0059635, -0.0194824, 0.0195971
1: -0.0091484, 0.0030568, -0.0089764, 0.0029533, -0.0121017, 0.0120333
2: 0.0219858, 0.0609203, 0.0225874, 0.0605524, -0.0385666, 0.0383329
3: -0.0044203, 0.0128373, -0.0044033, 0.0125868, -0.0170071, 0.0172406
4: -0.0154338, 0.0121738, -0.0152901, 0.0118888, -0.0273226, 0.0274640
5: 0.0009700, 0.0248316, 0.0011105, 0.0246615, -0.0236915, 0.0237210
6: -0.0376530, 0.0153240, -0.0372523, 0.0150183, -0.0526713, 0.0525764
7: 0.9432266, 0.9809444, 0.9439616, 0.9809010, -0.0376744, 0.0369828
8: -0.0344133, 0.0235046, -0.0341758, 0.0229095, -0.0573229, 0.0576804
9: -0.0202882, 0.0197627, -0.0199066, 0.0194042, -0.0396925, 0.0396693

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301192
time: 1.26 seconds

## Relational analysis of NS_A1_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.33 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0129175, 0.0050649, -0.0135021, 0.0061738, -0.0190912, 0.0185670
1: -0.0083390, 0.0025694, -0.0091257, 0.0030431, -0.0113821, 0.0116950
2: 0.0248174, 0.0591888, 0.0220655, 0.0608717, -0.0360543, 0.0371233
3: -0.0043404, 0.0116584, -0.0044180, 0.0128041, -0.0171445, 0.0160764
4: -0.0147577, 0.0108323, -0.0154148, 0.0121361, -0.0268938, 0.0262471
5: 0.0016314, 0.0240312, 0.0009886, 0.0248090, -0.0231777, 0.0230426
6: -0.0357669, 0.0138852, -0.0376000, 0.0152835, -0.0510505, 0.0514852
7: 0.9466859, 0.9807403, 0.9433239, 0.9809387, -0.0342528, 0.0374164
8: -0.0332951, 0.0207038, -0.0343819, 0.0234258, -0.0567209, 0.0550857
9: -0.0184923, 0.0180756, -0.0202376, 0.0197152, -0.0382075, 0.0383132

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302843, upper bound: 0.0293418
time: 1.82 seconds

## Relational analysis of NS_A2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0133912, 0.0059635, -0.0135190, 0.0062059, -0.0195971, 0.0194824
1: -0.0089764, 0.0029533, -0.0091484, 0.0030568, -0.0120333, 0.0121017
2: 0.0225874, 0.0605524, 0.0219858, 0.0609203, -0.0383329, 0.0385666
3: -0.0044033, 0.0125868, -0.0044203, 0.0128373, -0.0172406, 0.0170071
4: -0.0152901, 0.0118888, -0.0154338, 0.0121738, -0.0274640, 0.0273226
5: 0.0011105, 0.0246615, 0.0009700, 0.0248316, -0.0237210, 0.0236915
6: -0.0372523, 0.0150183, -0.0376530, 0.0153240, -0.0525764, 0.0526713
7: 0.9439616, 0.9809010, 0.9432266, 0.9809444, -0.0369828, 0.0376744
8: -0.0341758, 0.0229095, -0.0344133, 0.0235046, -0.0576804, 0.0573229
9: -0.0199066, 0.0194042, -0.0202882, 0.0197627, -0.0396693, 0.0396925

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301192, upper bound: 0.0293418
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.53 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0129175, 0.0050649, -0.0134420, 0.0060598, -0.0189773, 0.0185069
1: -0.0083390, 0.0025694, -0.0090448, 0.0029944, -0.0113334, 0.0116142
2: 0.0248174, 0.0591888, 0.0223482, 0.0606987, -0.0358814, 0.0368406
3: -0.0043404, 0.0116584, -0.0044101, 0.0126864, -0.0170268, 0.0160684
4: -0.0147577, 0.0108323, -0.0153473, 0.0120022, -0.0267598, 0.0261796
5: 0.0016314, 0.0240312, 0.0010546, 0.0247291, -0.0230978, 0.0229766
6: -0.0357669, 0.0138852, -0.0374117, 0.0151399, -0.0509068, 0.0512969
7: 0.9466859, 0.9807403, 0.9436692, 0.9809183, -0.0342324, 0.0370711
8: -0.0332951, 0.0207038, -0.0342702, 0.0231462, -0.0564413, 0.0549741
9: -0.0184923, 0.0180756, -0.0200584, 0.0195468, -0.0380390, 0.0381340

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302843, upper bound: 0.0293418
time: 2.61 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 1.46 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0133912, 0.0059635, -0.0134608, 0.0060956, -0.0194868, 0.0194243
1: -0.0089764, 0.0029533, -0.0090702, 0.0030097, -0.0119861, 0.0120234
2: 0.0225874, 0.0605524, 0.0222596, 0.0607530, -0.0381655, 0.0382928
3: -0.0044033, 0.0125868, -0.0044126, 0.0127233, -0.0171266, 0.0169994
4: -0.0152901, 0.0118888, -0.0153684, 0.0120441, -0.0273343, 0.0272572
5: 0.0011105, 0.0246615, 0.0010339, 0.0247542, -0.0236437, 0.0236276
6: -0.0372523, 0.0150183, -0.0374707, 0.0151849, -0.0524372, 0.0524890
7: 0.9439616, 0.9809010, 0.9435611, 0.9809247, -0.0369631, 0.0373399
8: -0.0341758, 0.0229095, -0.0343052, 0.0232338, -0.0574096, 0.0572147
9: -0.0199066, 0.0194042, -0.0201146, 0.0195996, -0.0395062, 0.0395188

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.13 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0300605
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301192
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0302843
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0294948
NS_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301192
NS_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0302843, upper bound: 0.0293418
NS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0301192, upper bound: 0.0293418
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0302843, upper bound: 0.0293418
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.13
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0129301, 0.0050889, -0.0134507, 0.0060765, -0.0190066, 0.0185397
1: -0.0083560, 0.0025796, -0.0090566, 0.0030015, -0.0113575, 0.0116362
2: 0.0247579, 0.0592252, 0.0223071, 0.0607240, -0.0359660, 0.0369181
3: -0.0043421, 0.0116831, -0.0044112, 0.0127035, -0.0170456, 0.0160943
4: -0.0147719, 0.0108605, -0.0153571, 0.0120217, -0.0267935, 0.0262176
5: 0.0016175, 0.0240480, 0.0010450, 0.0247408, -0.0231233, 0.0230030
6: -0.0358065, 0.0139154, -0.0374392, 0.0151608, -0.0509674, 0.0513546
7: 0.9466134, 0.9807446, 0.9436190, 0.9809213, -0.0343078, 0.0371256
8: -0.0333186, 0.0207626, -0.0342865, 0.0231869, -0.0565055, 0.0550491
9: -0.0185300, 0.0181110, -0.0200845, 0.0195713, -0.0381013, 0.0381955

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 1.30 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0129184, 0.0050667, -0.0145496, 0.0081607, -0.0210790, 0.0196163
1: -0.0083402, 0.0025701, -0.0105352, 0.0038919, -0.0122322, 0.0131053
2: 0.0248131, 0.0591914, 0.0171345, 0.0638870, -0.0390740, 0.0420570
3: -0.0043405, 0.0116601, -0.0045572, 0.0148572, -0.0191977, 0.0162174
4: -0.0147587, 0.0108344, -0.0165922, 0.0144723, -0.0292310, 0.0274266
5: 0.0016304, 0.0240324, -0.0001632, 0.0262029, -0.0245725, 0.0241956
6: -0.0357698, 0.0138874, -0.0408846, 0.0177892, -0.0535590, 0.0547720
7: 0.9466808, 0.9807407, 0.9372995, 0.9812942, -0.0346134, 0.0434412
8: -0.0332968, 0.0207081, -0.0363292, 0.0283033, -0.0616001, 0.0570373
9: -0.0184950, 0.0180782, -0.0233652, 0.0226532, -0.0411482, 0.0414433

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0134493, 0.0060737, -0.0134682, 0.0061095, -0.0195588, 0.0195419
1: -0.0090546, 0.0030003, -0.0090800, 0.0030156, -0.0120703, 0.0120804
2: 0.0223139, 0.0607197, 0.0222251, 0.0607741, -0.0384602, 0.0384947
3: -0.0044110, 0.0127007, -0.0044135, 0.0127377, -0.0171487, 0.0171142
4: -0.0153554, 0.0120184, -0.0153767, 0.0120605, -0.0274159, 0.0273951
5: 0.0010466, 0.0247388, 0.0010259, 0.0247639, -0.0237173, 0.0237130
6: -0.0374345, 0.0151573, -0.0374937, 0.0152025, -0.0526370, 0.0526511
7: 0.9436274, 0.9809207, 0.9435188, 0.9809272, -0.0372999, 0.0374019
8: -0.0342838, 0.0231801, -0.0343189, 0.0232680, -0.0575518, 0.0574990
9: -0.0200801, 0.0195672, -0.0201365, 0.0196202, -0.0397003, 0.0397037

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.43 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0134376, 0.0060516, -0.0145669, 0.0081935, -0.0216311, 0.0206185
1: -0.0090389, 0.0029909, -0.0105585, 0.0039060, -0.0129449, 0.0135494
2: 0.0223688, 0.0606862, 0.0170529, 0.0639369, -0.0415681, 0.0436332
3: -0.0044095, 0.0126778, -0.0045595, 0.0148911, -0.0193006, 0.0172374
4: -0.0153423, 0.0119924, -0.0166117, 0.0145109, -0.0298532, 0.0286041
5: 0.0010594, 0.0247233, -0.0001823, 0.0262259, -0.0251665, 0.0249056
6: -0.0373980, 0.0151294, -0.0409389, 0.0178306, -0.0552286, 0.0560683
7: 0.9436943, 0.9809169, 0.9371998, 0.9813001, -0.0376058, 0.0437170
8: -0.0342621, 0.0231258, -0.0363614, 0.0283839, -0.0626460, 0.0594872
9: -0.0200453, 0.0195345, -0.0234168, 0.0227018, -0.0427472, 0.0429514

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285424, upper bound: 0.0274104
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 2.75 seconds

## BFS NS instance: NS_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0134938, 0.0061582, -0.0128112, 0.0048635, -0.0183573, 0.0189694
1: -0.0091146, 0.0030364, -0.0081961, 0.0024833, -0.0115979, 0.0112325
2: 0.0221042, 0.0608480, 0.0253174, 0.0588830, -0.0367788, 0.0355306
3: -0.0044170, 0.0127880, -0.0043263, 0.0114502, -0.0158671, 0.0171143
4: -0.0154055, 0.0121178, -0.0146383, 0.0105954, -0.0260009, 0.0267560
5: 0.0009976, 0.0247981, 0.0017482, 0.0238898, -0.0228922, 0.0230499
6: -0.0375742, 0.0152639, -0.0354339, 0.0136311, -0.0512053, 0.0506977
7: 0.9433712, 0.9809360, 0.9472969, 0.9807042, -0.0373330, 0.0336391
8: -0.0343666, 0.0233875, -0.0330976, 0.0202093, -0.0545759, 0.0564852
9: -0.0202131, 0.0196922, -0.0181751, 0.0177777, -0.0379908, 0.0378673

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
time: 1.46 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
time: 1.49 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0134822, 0.0061361, -0.0139148, 0.0069566, -0.0204388, 0.0200509
1: -0.0090989, 0.0030270, -0.0096810, 0.0033775, -0.0124765, 0.0127080
2: 0.0221590, 0.0608145, 0.0201227, 0.0620596, -0.0399007, 0.0406917
3: -0.0044154, 0.0127652, -0.0044729, 0.0136130, -0.0180284, 0.0172381
4: -0.0153924, 0.0120918, -0.0158786, 0.0130565, -0.0284490, 0.0279705
5: 0.0010104, 0.0247826, 0.0005348, 0.0253582, -0.0243478, 0.0242478
6: -0.0375377, 0.0152360, -0.0388941, 0.0162707, -0.0538085, 0.0541301
7: 0.9434381, 0.9809319, 0.9409504, 0.9810788, -0.0376407, 0.0399815
8: -0.0343449, 0.0233333, -0.0351491, 0.0253475, -0.0596924, 0.0584824
9: -0.0201784, 0.0196595, -0.0214699, 0.0208728, -0.0410511, 0.0411294

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
time: 1.47 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0135108, 0.0061904, -0.0132872, 0.0057663, -0.0192771, 0.0194776
1: -0.0091374, 0.0030502, -0.0088365, 0.0028690, -0.0120064, 0.0118867
2: 0.0220243, 0.0608968, 0.0230768, 0.0602532, -0.0382289, 0.0378200
3: -0.0044192, 0.0128213, -0.0043895, 0.0123830, -0.0168023, 0.0172108
4: -0.0154246, 0.0121556, -0.0151733, 0.0116569, -0.0270815, 0.0273289
5: 0.0009790, 0.0248207, 0.0012248, 0.0245232, -0.0235442, 0.0235959
6: -0.0376274, 0.0153045, -0.0369263, 0.0147696, -0.0523971, 0.0522308
7: 0.9432736, 0.9809417, 0.9445596, 0.9808658, -0.0375922, 0.0363821
8: -0.0343981, 0.0234666, -0.0339825, 0.0224254, -0.0568236, 0.0574490
9: -0.0202638, 0.0197398, -0.0195962, 0.0191127, -0.0393765, 0.0393360

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_B2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.27 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0134992, 0.0061683, -0.0143816, 0.0078421, -0.0213413, 0.0205499
1: -0.0091218, 0.0030408, -0.0103092, 0.0037558, -0.0128776, 0.0133499
2: 0.0220790, 0.0608634, 0.0179252, 0.0634035, -0.0413245, 0.0429382
3: -0.0044177, 0.0127985, -0.0045349, 0.0145280, -0.0189456, 0.0173334
4: -0.0154115, 0.0121297, -0.0164034, 0.0140977, -0.0295093, 0.0285331
5: 0.0009917, 0.0248052, 0.0000215, 0.0259794, -0.0249876, 0.0247838
6: -0.0375910, 0.0152767, -0.0403579, 0.0173874, -0.0549784, 0.0556346
7: 0.9433403, 0.9809377, 0.9382654, 0.9812371, -0.0378968, 0.0426723
8: -0.0343766, 0.0234125, -0.0360169, 0.0275211, -0.0618977, 0.0594294
9: -0.0202291, 0.0197072, -0.0228637, 0.0221821, -0.0424113, 0.0425709

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B2_B1

### Relational analysis result of NS_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
time: 1.46 seconds

## Relational analysis of NS_A1_B2_B2_B2_B2

### Relational analysis result of NS_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.35 seconds

## BFS NS instance: NS_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0128112, 0.0048635, -0.0134938, 0.0061582, -0.0189694, 0.0183573
1: -0.0081961, 0.0024833, -0.0091146, 0.0030364, -0.0112325, 0.0115979
2: 0.0253174, 0.0588830, 0.0221042, 0.0608480, -0.0355306, 0.0367788
3: -0.0043263, 0.0114502, -0.0044170, 0.0127880, -0.0171143, 0.0158671
4: -0.0146383, 0.0105954, -0.0154055, 0.0121178, -0.0267560, 0.0260009
5: 0.0017482, 0.0238898, 0.0009976, 0.0247981, -0.0230499, 0.0228922
6: -0.0354339, 0.0136311, -0.0375742, 0.0152639, -0.0506977, 0.0512053
7: 0.9472969, 0.9807042, 0.9433712, 0.9809360, -0.0336391, 0.0373330
8: -0.0330976, 0.0202093, -0.0343666, 0.0233875, -0.0564852, 0.0545759
9: -0.0181751, 0.0177777, -0.0202131, 0.0196922, -0.0378673, 0.0379908

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A1_A1

### Relational analysis result of NS_A2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A1_A1_A2

### Relational analysis result of NS_A2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.40 seconds

## BFS NS instance: NS_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0139148, 0.0069566, -0.0134822, 0.0061361, -0.0200509, 0.0204388
1: -0.0096810, 0.0033775, -0.0090989, 0.0030270, -0.0127080, 0.0124765
2: 0.0201227, 0.0620596, 0.0221590, 0.0608145, -0.0406917, 0.0399007
3: -0.0044729, 0.0136130, -0.0044154, 0.0127652, -0.0172381, 0.0180284
4: -0.0158786, 0.0130565, -0.0153924, 0.0120918, -0.0279705, 0.0284490
5: 0.0005348, 0.0253582, 0.0010104, 0.0247826, -0.0242478, 0.0243478
6: -0.0388941, 0.0162707, -0.0375377, 0.0152360, -0.0541301, 0.0538085
7: 0.9409504, 0.9810788, 0.9434381, 0.9809319, -0.0399815, 0.0376407
8: -0.0351491, 0.0253475, -0.0343449, 0.0233333, -0.0584824, 0.0596924
9: -0.0214699, 0.0208728, -0.0201784, 0.0196595, -0.0411294, 0.0410511

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A2_A1

### Relational analysis result of NS_A2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A1_A2_A2

### Relational analysis result of NS_A2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0132872, 0.0057663, -0.0135108, 0.0061904, -0.0194776, 0.0192771
1: -0.0088365, 0.0028690, -0.0091374, 0.0030502, -0.0118867, 0.0120064
2: 0.0230768, 0.0602532, 0.0220243, 0.0608968, -0.0378200, 0.0382289
3: -0.0043895, 0.0123830, -0.0044192, 0.0128213, -0.0172108, 0.0168023
4: -0.0151733, 0.0116569, -0.0154246, 0.0121556, -0.0273289, 0.0270815
5: 0.0012248, 0.0245232, 0.0009790, 0.0248207, -0.0235959, 0.0235442
6: -0.0369263, 0.0147696, -0.0376274, 0.0153045, -0.0522308, 0.0523971
7: 0.9445596, 0.9808658, 0.9432736, 0.9809417, -0.0363821, 0.0375922
8: -0.0339825, 0.0224254, -0.0343981, 0.0234666, -0.0574490, 0.0568236
9: -0.0195962, 0.0191127, -0.0202638, 0.0197398, -0.0393360, 0.0393765

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.71 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0143816, 0.0078421, -0.0134992, 0.0061683, -0.0205499, 0.0213413
1: -0.0103092, 0.0037558, -0.0091218, 0.0030408, -0.0133499, 0.0128776
2: 0.0179252, 0.0634035, 0.0220790, 0.0608634, -0.0429382, 0.0413245
3: -0.0045349, 0.0145280, -0.0044177, 0.0127985, -0.0173334, 0.0189456
4: -0.0164034, 0.0140977, -0.0154115, 0.0121297, -0.0285331, 0.0295093
5: 0.0000215, 0.0259794, 0.0009917, 0.0248052, -0.0247838, 0.0249876
6: -0.0403579, 0.0173874, -0.0375910, 0.0152767, -0.0556346, 0.0549784
7: 0.9382654, 0.9812371, 0.9433403, 0.9809377, -0.0426723, 0.0378968
8: -0.0360169, 0.0275211, -0.0343766, 0.0234125, -0.0594294, 0.0618977
9: -0.0228637, 0.0221821, -0.0202291, 0.0197072, -0.0425709, 0.0424113

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285424, upper bound: 0.0274104
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.26 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0128112, 0.0048635, -0.0134252, 0.0060280, -0.0188393, 0.0182887
1: -0.0081961, 0.0024833, -0.0090223, 0.0029809, -0.0111769, 0.0115056
2: 0.0253174, 0.0588830, 0.0224272, 0.0606504, -0.0353331, 0.0364558
3: -0.0043263, 0.0114502, -0.0044078, 0.0126535, -0.0169798, 0.0158580
4: -0.0146383, 0.0105954, -0.0153284, 0.0119647, -0.0266030, 0.0259238
5: 0.0017482, 0.0238898, 0.0010731, 0.0247068, -0.0229586, 0.0228167
6: -0.0354339, 0.0136311, -0.0373591, 0.0150997, -0.0505336, 0.0509902
7: 0.9472969, 0.9807042, 0.9437658, 0.9809126, -0.0336157, 0.0369384
8: -0.0330976, 0.0202093, -0.0342390, 0.0230681, -0.0561657, 0.0544483
9: -0.0181751, 0.0177777, -0.0200083, 0.0194997, -0.0376748, 0.0377859

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 1.48 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0139148, 0.0069566, -0.0134000, 0.0059803, -0.0198950, 0.0203566
1: -0.0096810, 0.0033775, -0.0089884, 0.0029604, -0.0126414, 0.0123659
2: 0.0201227, 0.0620596, 0.0225458, 0.0605780, -0.0404552, 0.0395139
3: -0.0044729, 0.0136130, -0.0044045, 0.0126042, -0.0170770, 0.0180175
4: -0.0158786, 0.0130565, -0.0153001, 0.0119086, -0.0277872, 0.0283566
5: 0.0005348, 0.0253582, 0.0011008, 0.0246733, -0.0241385, 0.0242574
6: -0.0388941, 0.0162707, -0.0372801, 0.0150395, -0.0539336, 0.0535508
7: 0.9409504, 0.9810788, 0.9439107, 0.9809040, -0.0399536, 0.0371681
8: -0.0351491, 0.0253475, -0.0341922, 0.0229508, -0.0580999, 0.0595396
9: -0.0214699, 0.0208728, -0.0199331, 0.0194291, -0.0408989, 0.0408058

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0133744, 0.0059317, -0.0133575, 0.0058996, -0.0192740, 0.0192892
1: -0.0089539, 0.0029397, -0.0089311, 0.0029260, -0.0118799, 0.0118708
2: 0.0226662, 0.0605043, 0.0227461, 0.0604555, -0.0377893, 0.0377582
3: -0.0044011, 0.0125540, -0.0043988, 0.0125208, -0.0169218, 0.0169528
4: -0.0152713, 0.0118515, -0.0152523, 0.0118137, -0.0270850, 0.0271038
5: 0.0011289, 0.0246392, 0.0011476, 0.0246167, -0.0234878, 0.0234917
6: -0.0371998, 0.0149783, -0.0371466, 0.0149378, -0.0521376, 0.0521250
7: 0.9440578, 0.9808955, 0.9441553, 0.9808896, -0.0368318, 0.0367401
8: -0.0341446, 0.0228316, -0.0341131, 0.0227527, -0.0568973, 0.0569447
9: -0.0198567, 0.0193573, -0.0198060, 0.0193097, -0.0391664, 0.0391633

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0133492, 0.0058838, -0.0144516, 0.0079749, -0.0213241, 0.0203355
1: -0.0089199, 0.0029192, -0.0104034, 0.0038126, -0.0127325, 0.0133227
2: 0.0227852, 0.0604315, 0.0175954, 0.0636051, -0.0408199, 0.0428361
3: -0.0043977, 0.0125045, -0.0045442, 0.0146652, -0.0190630, 0.0170487
4: -0.0152429, 0.0117951, -0.0164821, 0.0142539, -0.0294968, 0.0282772
5: 0.0011567, 0.0246056, -0.0000556, 0.0260726, -0.0249159, 0.0246612
6: -0.0371206, 0.0149178, -0.0405775, 0.0175549, -0.0546755, 0.0554954
7: 0.9442031, 0.9808869, 0.9378626, 0.9812609, -0.0370578, 0.0430242
8: -0.0340976, 0.0227140, -0.0361472, 0.0278473, -0.0619449, 0.0588611
9: -0.0197812, 0.0192864, -0.0230728, 0.0223786, -0.0421598, 0.0423593

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285427, upper bound: 0.0274104
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.16 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.79 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0285424, upper bound: 0.0274104
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
NS_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
NS_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
NS_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
NS_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
NS_A2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0285424, upper bound: 0.0274104
NS_A2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
NS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0285427, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.79
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0134507, 0.0060765, -0.0189616, 0.0184543
1: -0.0082955, 0.0025432, -0.0090566, 0.0030015, -0.0112970, 0.0115998
2: 0.0249697, 0.0590957, 0.0223071, 0.0607240, -0.0357543, 0.0367886
3: -0.0043361, 0.0115950, -0.0044112, 0.0127035, -0.0170396, 0.0160062
4: -0.0147213, 0.0107602, -0.0153571, 0.0120217, -0.0267430, 0.0261173
5: 0.0016670, 0.0239881, 0.0010450, 0.0247408, -0.0230738, 0.0229431
6: -0.0356655, 0.0138078, -0.0374392, 0.0151608, -0.0508263, 0.0512470
7: 0.9468721, 0.9807293, 0.9436190, 0.9809213, -0.0340492, 0.0371103
8: -0.0332350, 0.0205532, -0.0342865, 0.0231869, -0.0564219, 0.0548397
9: -0.0183957, 0.0179849, -0.0200845, 0.0195713, -0.0379670, 0.0380693

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0281711
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0134507, 0.0060765, -0.0201026, 0.0206185
1: -0.0098308, 0.0034677, -0.0090566, 0.0030015, -0.0128323, 0.0125243
2: 0.0195986, 0.0623801, 0.0223071, 0.0607240, -0.0411253, 0.0400731
3: -0.0044877, 0.0138312, -0.0044112, 0.0127035, -0.0171912, 0.0182424
4: -0.0160038, 0.0133048, -0.0153571, 0.0120217, -0.0280255, 0.0286619
5: 0.0004124, 0.0255063, 0.0010450, 0.0247408, -0.0243284, 0.0244613
6: -0.0392432, 0.0165370, -0.0374392, 0.0151608, -0.0544040, 0.0539762
7: 0.9403101, 0.9811166, 0.9436190, 0.9809213, -0.0406111, 0.0374976
8: -0.0353561, 0.0258659, -0.0342865, 0.0231869, -0.0585430, 0.0601524
9: -0.0218022, 0.0211850, -0.0200845, 0.0195713, -0.0413736, 0.0412695

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0281711
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128970, 0.0050262, -0.0145496, 0.0081607, -0.0210577, 0.0195757
1: -0.0083115, 0.0025528, -0.0105352, 0.0038919, -0.0122034, 0.0130880
2: 0.0249137, 0.0591299, 0.0171345, 0.0638870, -0.0389734, 0.0419955
3: -0.0043377, 0.0116183, -0.0045572, 0.0148572, -0.0191948, 0.0161755
4: -0.0147347, 0.0107867, -0.0165922, 0.0144723, -0.0292070, 0.0273789
5: 0.0016539, 0.0240039, -0.0001632, 0.0262029, -0.0245490, 0.0241672
6: -0.0357028, 0.0138363, -0.0408846, 0.0177892, -0.0534920, 0.0547209
7: 0.9468036, 0.9807334, 0.9372995, 0.9812942, -0.0344906, 0.0434339
8: -0.0332571, 0.0206086, -0.0363292, 0.0283033, -0.0615604, 0.0569379
9: -0.0184312, 0.0180182, -0.0233652, 0.0226532, -0.0410844, 0.0413834

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0148829, 0.0087929, -0.0145315, 0.0081264, -0.0230092, 0.0233243
1: -0.0109837, 0.0041620, -0.0105109, 0.0038773, -0.0148610, 0.0146729
2: 0.0155655, 0.0648465, 0.0172196, 0.0638350, -0.0482695, 0.0476269
3: -0.0046015, 0.0155104, -0.0045548, 0.0148217, -0.0194232, 0.0200653
4: -0.0169668, 0.0152157, -0.0165719, 0.0144320, -0.0313988, 0.0317876
5: -0.0005297, 0.0266464, -0.0001433, 0.0261788, -0.0267086, 0.0267897
6: -0.0419297, 0.0185864, -0.0408279, 0.0177459, -0.0596756, 0.0594144
7: 0.9353825, 0.9814073, 0.9374034, 0.9812880, -0.0459054, 0.0440039
8: -0.0369488, 0.0298552, -0.0362956, 0.0282190, -0.0651678, 0.0661508
9: -0.0243603, 0.0235881, -0.0233112, 0.0226025, -0.0469628, 0.0468993

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.24 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0134682, 0.0061095, -0.0195155, 0.0194598
1: -0.0089964, 0.0029653, -0.0090800, 0.0030156, -0.0120121, 0.0120453
2: 0.0225176, 0.0605952, 0.0222251, 0.0607741, -0.0382565, 0.0383701
3: -0.0044053, 0.0126159, -0.0044135, 0.0127377, -0.0171430, 0.0170294
4: -0.0153068, 0.0119219, -0.0153767, 0.0120605, -0.0273673, 0.0272986
5: 0.0010942, 0.0246813, 0.0010259, 0.0247639, -0.0236697, 0.0236554
6: -0.0372989, 0.0150538, -0.0374937, 0.0152025, -0.0525013, 0.0525476
7: 0.9438763, 0.9809060, 0.9435188, 0.9809272, -0.0370510, 0.0373872
8: -0.0342033, 0.0229786, -0.0343189, 0.0232680, -0.0574713, 0.0572975
9: -0.0199509, 0.0194459, -0.0201365, 0.0196202, -0.0395711, 0.0395823

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0145038, 0.0080739, -0.0134682, 0.0061095, -0.0206133, 0.0215421
1: -0.0104737, 0.0038549, -0.0090800, 0.0030156, -0.0134893, 0.0129349
2: 0.0173498, 0.0637554, 0.0222251, 0.0607741, -0.0434243, 0.0415303
3: -0.0045511, 0.0147675, -0.0044135, 0.0127377, -0.0172888, 0.0191811
4: -0.0165408, 0.0143703, -0.0153767, 0.0120605, -0.0286013, 0.0297470
5: -0.0001129, 0.0261420, 0.0010259, 0.0247639, -0.0248769, 0.0251162
6: -0.0407412, 0.0176798, -0.0374937, 0.0152025, -0.0559437, 0.0551735
7: 0.9375624, 0.9812785, 0.9435188, 0.9809272, -0.0433648, 0.0377597
8: -0.0362442, 0.0280903, -0.0343189, 0.0232680, -0.0595122, 0.0624092
9: -0.0232286, 0.0225250, -0.0201365, 0.0196202, -0.0428488, 0.0426614

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0134153, 0.0060092, -0.0145669, 0.0081935, -0.0216088, 0.0205761
1: -0.0090089, 0.0029728, -0.0105585, 0.0039060, -0.0129149, 0.0135313
2: 0.0224739, 0.0606219, 0.0170529, 0.0639369, -0.0414630, 0.0435690
3: -0.0044065, 0.0126341, -0.0045595, 0.0148911, -0.0192976, 0.0171936
4: -0.0153173, 0.0119426, -0.0166117, 0.0145109, -0.0298282, 0.0285543
5: 0.0010840, 0.0246936, -0.0001823, 0.0262259, -0.0251419, 0.0248759
6: -0.0373280, 0.0150760, -0.0409389, 0.0178306, -0.0551586, 0.0560149
7: 0.9438229, 0.9809093, 0.9371998, 0.9813001, -0.0374772, 0.0437095
8: -0.0342206, 0.0230219, -0.0363614, 0.0283839, -0.0626045, 0.0593833
9: -0.0199786, 0.0194719, -0.0234168, 0.0227018, -0.0426805, 0.0428887

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0153611, 0.0096999, -0.0145487, 0.0081590, -0.0235201, 0.0242486
1: -0.0116272, 0.0045495, -0.0105340, 0.0038912, -0.0155184, 0.0150835
2: 0.0133143, 0.0662231, 0.0171386, 0.0638845, -0.0505702, 0.0490845
3: -0.0046650, 0.0164477, -0.0045571, 0.0148555, -0.0195205, 0.0210048
4: -0.0175044, 0.0162822, -0.0165912, 0.0144704, -0.0319747, 0.0328734
5: -0.0010556, 0.0272827, -0.0001623, 0.0262017, -0.0272573, 0.0274450
6: -0.0434292, 0.0197303, -0.0408819, 0.0177871, -0.0612163, 0.0606122
7: 0.9326323, 0.9815695, 0.9373045, 0.9812939, -0.0486615, 0.0442650
8: -0.0378378, 0.0320819, -0.0363276, 0.0282992, -0.0661370, 0.0684095
9: -0.0257881, 0.0249294, -0.0233626, 0.0226508, -0.0484389, 0.0482919

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0134938, 0.0061582, -0.0127776, 0.0048254, -0.0183192, 0.0189358
1: -0.0091146, 0.0030364, -0.0081709, 0.0024561, -0.0115706, 0.0112073
2: 0.0221042, 0.0608480, 0.0253886, 0.0587862, -0.0366820, 0.0354594
3: -0.0044170, 0.0127880, -0.0043218, 0.0113997, -0.0158166, 0.0171098
4: -0.0154055, 0.0121178, -0.0146005, 0.0105412, -0.0259467, 0.0267182
5: 0.0009976, 0.0247981, 0.0017795, 0.0238451, -0.0228474, 0.0230186
6: -0.0375742, 0.0152639, -0.0353284, 0.0135622, -0.0511364, 0.0505923
7: 0.9433712, 0.9809360, 0.9474486, 0.9806929, -0.0373216, 0.0334874
8: -0.0343666, 0.0233875, -0.0330351, 0.0200919, -0.0544585, 0.0564227
9: -0.0202131, 0.0196922, -0.0180747, 0.0177138, -0.0379269, 0.0377669

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_B1_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
time: 1.94 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0134761, 0.0061245, -0.0147582, 0.0085563, -0.0220324, 0.0208827
1: -0.0090907, 0.0030221, -0.0108159, 0.0040610, -0.0131517, 0.0138380
2: 0.0221878, 0.0607969, 0.0161525, 0.0644875, -0.0422998, 0.0446444
3: -0.0044146, 0.0127532, -0.0045849, 0.0152660, -0.0196806, 0.0173381
4: -0.0153856, 0.0120782, -0.0168267, 0.0149376, -0.0303231, 0.0289048
5: 0.0010172, 0.0247745, -0.0003926, 0.0264805, -0.0254633, 0.0251671
6: -0.0375186, 0.0152214, -0.0415387, 0.0182881, -0.0558067, 0.0567602
7: 0.9434732, 0.9809299, 0.9360997, 0.9813651, -0.0378919, 0.0448302
8: -0.0343336, 0.0233048, -0.0367170, 0.0292746, -0.0636082, 0.0600219
9: -0.0201601, 0.0196424, -0.0239880, 0.0232384, -0.0433985, 0.0436304

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
time: 1.22 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
time: 1.24 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0134822, 0.0061361, -0.0138813, 0.0068931, -0.0203753, 0.0200174
1: -0.0090989, 0.0030270, -0.0096360, 0.0033504, -0.0124493, 0.0126630
2: 0.0221590, 0.0608145, 0.0202803, 0.0619633, -0.0398043, 0.0405341
3: -0.0044154, 0.0127652, -0.0044684, 0.0135474, -0.0179628, 0.0172336
4: -0.0153924, 0.0120918, -0.0158410, 0.0129819, -0.0283743, 0.0279328
5: 0.0010104, 0.0247826, 0.0005716, 0.0253137, -0.0243032, 0.0242110
6: -0.0375377, 0.0152360, -0.0387891, 0.0161906, -0.0537284, 0.0540251
7: 0.9434381, 0.9809319, 0.9411429, 0.9810674, -0.0376293, 0.0397890
8: -0.0343449, 0.0233333, -0.0350868, 0.0251916, -0.0595365, 0.0584202
9: -0.0201784, 0.0196595, -0.0213699, 0.0207789, -0.0409572, 0.0410294

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
time: 1.27 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0134644, 0.0061023, -0.0158551, 0.0106369, -0.0241013, 0.0219575
1: -0.0090750, 0.0030126, -0.0122919, 0.0049498, -0.0140248, 0.0153045
2: 0.0222428, 0.0607632, 0.0109890, 0.0676451, -0.0454023, 0.0497743
3: -0.0044130, 0.0127303, -0.0047307, 0.0174159, -0.0218290, 0.0174610
4: -0.0153724, 0.0120521, -0.0180596, 0.0173839, -0.0327564, 0.0301117
5: 0.0010300, 0.0247589, -0.0015988, 0.0279401, -0.0269100, 0.0263577
6: -0.0374819, 0.0151935, -0.0449782, 0.0209119, -0.0583938, 0.0601717
7: 0.9435405, 0.9809259, 0.9297912, 0.9817372, -0.0381967, 0.0511346
8: -0.0343119, 0.0232505, -0.0387562, 0.0343820, -0.0686939, 0.0620066
9: -0.0201252, 0.0196096, -0.0272630, 0.0263149, -0.0464401, 0.0468726

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 2.22 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.16 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0134682, 0.0061095, -0.0132872, 0.0057663, -0.0192345, 0.0193967
1: -0.0090800, 0.0030156, -0.0088365, 0.0028690, -0.0119490, 0.0118522
2: 0.0222251, 0.0607741, 0.0230768, 0.0602532, -0.0380281, 0.0376972
3: -0.0044135, 0.0127377, -0.0043895, 0.0123830, -0.0167966, 0.0171272
4: -0.0153767, 0.0120605, -0.0151733, 0.0116569, -0.0270336, 0.0272338
5: 0.0010259, 0.0247639, 0.0012248, 0.0245232, -0.0234973, 0.0235391
6: -0.0374937, 0.0152025, -0.0369263, 0.0147696, -0.0522634, 0.0521288
7: 0.9435188, 0.9809272, 0.9445596, 0.9808658, -0.0373470, 0.0363677
8: -0.0343189, 0.0232680, -0.0339825, 0.0224254, -0.0567443, 0.0572505
9: -0.0201365, 0.0196202, -0.0195962, 0.0191127, -0.0392491, 0.0392164

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
time: 1.50 seconds

## Relational analysis of NS_A1_B2_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0145669, 0.0081935, -0.0132872, 0.0057663, -0.0203332, 0.0214807
1: -0.0105585, 0.0039060, -0.0088365, 0.0028690, -0.0134275, 0.0127425
2: 0.0170529, 0.0639369, 0.0230768, 0.0602532, -0.0432002, 0.0408600
3: -0.0045595, 0.0148911, -0.0043895, 0.0123830, -0.0169426, 0.0192806
4: -0.0166117, 0.0145109, -0.0151733, 0.0116569, -0.0282686, 0.0296842
5: -0.0001823, 0.0262259, 0.0012248, 0.0245232, -0.0247054, 0.0250011
6: -0.0409389, 0.0178306, -0.0369263, 0.0147696, -0.0557085, 0.0547569
7: 0.9371998, 0.9813001, 0.9445596, 0.9808658, -0.0436659, 0.0367405
8: -0.0363614, 0.0283839, -0.0339825, 0.0224254, -0.0587868, 0.0623663
9: -0.0234168, 0.0227018, -0.0195962, 0.0191127, -0.0425295, 0.0422980

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
time: 2.55 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
time: 1.32 seconds

## BFS NS instance: NS_A1_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0134992, 0.0061683, -0.0143482, 0.0077788, -0.0212780, 0.0205166
1: -0.0091218, 0.0030408, -0.0102643, 0.0037288, -0.0128506, 0.0133051
2: 0.0220790, 0.0608634, 0.0180822, 0.0633075, -0.0412285, 0.0427812
3: -0.0044177, 0.0127985, -0.0045305, 0.0144626, -0.0188802, 0.0173290
4: -0.0154115, 0.0121297, -0.0163659, 0.0140233, -0.0294348, 0.0284956
5: 0.0009917, 0.0248052, 0.0000581, 0.0259350, -0.0249433, 0.0247471
6: -0.0375910, 0.0152767, -0.0402533, 0.0173076, -0.0548987, 0.0555300
7: 0.9433403, 0.9809377, 0.9384574, 0.9812257, -0.0378854, 0.0424803
8: -0.0343766, 0.0234125, -0.0359549, 0.0273658, -0.0617424, 0.0593674
9: -0.0202291, 0.0197072, -0.0227641, 0.0220886, -0.0423177, 0.0424713

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B2_B2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285016
time: 1.30 seconds

## Relational analysis of NS_A1_B2_B2_B2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
time: 1.27 seconds

## BFS NS instance: NS_A1_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0134813, 0.0061343, -0.0163244, 0.0115270, -0.0250083, 0.0224587
1: -0.0090977, 0.0030263, -0.0129234, 0.0053301, -0.0144278, 0.0159497
2: 0.0221635, 0.0608118, 0.0087798, 0.0689961, -0.0468326, 0.0520319
3: -0.0044153, 0.0127633, -0.0047930, 0.0183357, -0.0227510, 0.0175564
4: -0.0153914, 0.0120897, -0.0185871, 0.0184306, -0.0338220, 0.0306768
5: 0.0010115, 0.0247814, -0.0021148, 0.0285645, -0.0275530, 0.0268961
6: -0.0375348, 0.0152338, -0.0464498, 0.0220345, -0.0595693, 0.0616836
7: 0.9434436, 0.9809317, 0.9270921, 0.9818965, -0.0384529, 0.0538396
8: -0.0343432, 0.0233290, -0.0396286, 0.0365672, -0.0709103, 0.0629575
9: -0.0201755, 0.0196569, -0.0286642, 0.0276311, -0.0478067, 0.0483210

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B2_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.25 seconds

## Relational analysis of NS_A1_B2_B2_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0127776, 0.0048254, -0.0134938, 0.0061582, -0.0189358, 0.0183192
1: -0.0081709, 0.0024561, -0.0091146, 0.0030364, -0.0112073, 0.0115706
2: 0.0253886, 0.0587862, 0.0221042, 0.0608480, -0.0354594, 0.0366820
3: -0.0043218, 0.0113997, -0.0044170, 0.0127880, -0.0171098, 0.0158166
4: -0.0146005, 0.0105412, -0.0154055, 0.0121178, -0.0267182, 0.0259467
5: 0.0017795, 0.0238451, 0.0009976, 0.0247981, -0.0230186, 0.0228474
6: -0.0353284, 0.0135622, -0.0375742, 0.0152639, -0.0505923, 0.0511364
7: 0.9474486, 0.9806929, 0.9433712, 0.9809360, -0.0334874, 0.0373216
8: -0.0330351, 0.0200919, -0.0343666, 0.0233875, -0.0564227, 0.0544585
9: -0.0180747, 0.0177138, -0.0202131, 0.0196922, -0.0377669, 0.0379269

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0147582, 0.0085563, -0.0134761, 0.0061245, -0.0208827, 0.0220324
1: -0.0108159, 0.0040610, -0.0090907, 0.0030221, -0.0138380, 0.0131517
2: 0.0161525, 0.0644875, 0.0221878, 0.0607969, -0.0446444, 0.0422998
3: -0.0045849, 0.0152660, -0.0044146, 0.0127532, -0.0173381, 0.0196806
4: -0.0168267, 0.0149376, -0.0153856, 0.0120782, -0.0289048, 0.0303231
5: -0.0003926, 0.0264805, 0.0010172, 0.0247745, -0.0251671, 0.0254633
6: -0.0415387, 0.0182881, -0.0375186, 0.0152214, -0.0567602, 0.0558067
7: 0.9360997, 0.9813651, 0.9434732, 0.9809299, -0.0448302, 0.0378919
8: -0.0367170, 0.0292746, -0.0343336, 0.0233048, -0.0600219, 0.0636082
9: -0.0239880, 0.0232384, -0.0201601, 0.0196424, -0.0436304, 0.0433985

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.36 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0138813, 0.0068931, -0.0134822, 0.0061361, -0.0200174, 0.0203753
1: -0.0096360, 0.0033504, -0.0090989, 0.0030270, -0.0126630, 0.0124493
2: 0.0202803, 0.0619633, 0.0221590, 0.0608145, -0.0405341, 0.0398043
3: -0.0044684, 0.0135474, -0.0044154, 0.0127652, -0.0172336, 0.0179628
4: -0.0158410, 0.0129819, -0.0153924, 0.0120918, -0.0279328, 0.0283743
5: 0.0005716, 0.0253137, 0.0010104, 0.0247826, -0.0242110, 0.0243032
6: -0.0387891, 0.0161906, -0.0375377, 0.0152360, -0.0540251, 0.0537284
7: 0.9411429, 0.9810674, 0.9434381, 0.9809319, -0.0397890, 0.0376293
8: -0.0350868, 0.0251916, -0.0343449, 0.0233333, -0.0584202, 0.0595365
9: -0.0213699, 0.0207789, -0.0201784, 0.0196595, -0.0410294, 0.0409572

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 2.49 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0158551, 0.0106369, -0.0134644, 0.0061023, -0.0219575, 0.0241013
1: -0.0122919, 0.0049498, -0.0090750, 0.0030126, -0.0153045, 0.0140248
2: 0.0109890, 0.0676451, 0.0222428, 0.0607632, -0.0497743, 0.0454023
3: -0.0047307, 0.0174159, -0.0044130, 0.0127303, -0.0174610, 0.0218290
4: -0.0180596, 0.0173839, -0.0153724, 0.0120521, -0.0301117, 0.0327564
5: -0.0015988, 0.0279401, 0.0010300, 0.0247589, -0.0263577, 0.0269100
6: -0.0449782, 0.0209119, -0.0374819, 0.0151935, -0.0601717, 0.0583938
7: 0.9297912, 0.9817372, 0.9435405, 0.9809259, -0.0511346, 0.0381967
8: -0.0387562, 0.0343820, -0.0343119, 0.0232505, -0.0620066, 0.0686939
9: -0.0272630, 0.0263149, -0.0201252, 0.0196096, -0.0468726, 0.0464401

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0132872, 0.0057663, -0.0134682, 0.0061095, -0.0193967, 0.0192345
1: -0.0088365, 0.0028690, -0.0090800, 0.0030156, -0.0118522, 0.0119490
2: 0.0230768, 0.0602532, 0.0222251, 0.0607741, -0.0376972, 0.0380281
3: -0.0043895, 0.0123830, -0.0044135, 0.0127377, -0.0171272, 0.0167966
4: -0.0151733, 0.0116569, -0.0153767, 0.0120605, -0.0272338, 0.0270336
5: 0.0012248, 0.0245232, 0.0010259, 0.0247639, -0.0235391, 0.0234973
6: -0.0369263, 0.0147696, -0.0374937, 0.0152025, -0.0521288, 0.0522634
7: 0.9445596, 0.9808658, 0.9435188, 0.9809272, -0.0363677, 0.0373470
8: -0.0339825, 0.0224254, -0.0343189, 0.0232680, -0.0572505, 0.0567443
9: -0.0195962, 0.0191127, -0.0201365, 0.0196202, -0.0392164, 0.0392491

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289763, upper bound: 0.0274104
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281808, upper bound: 0.0274104
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0132872, 0.0057663, -0.0145669, 0.0081935, -0.0214807, 0.0203332
1: -0.0088365, 0.0028690, -0.0105585, 0.0039060, -0.0127425, 0.0134275
2: 0.0230768, 0.0602532, 0.0170529, 0.0639369, -0.0408600, 0.0432002
3: -0.0043895, 0.0123830, -0.0045595, 0.0148911, -0.0192806, 0.0169426
4: -0.0151733, 0.0116569, -0.0166117, 0.0145109, -0.0296842, 0.0282686
5: 0.0012248, 0.0245232, -0.0001823, 0.0262259, -0.0250011, 0.0247054
6: -0.0369263, 0.0147696, -0.0409389, 0.0178306, -0.0547569, 0.0557085
7: 0.9445596, 0.9808658, 0.9371998, 0.9813001, -0.0367405, 0.0436659
8: -0.0339825, 0.0224254, -0.0363614, 0.0283839, -0.0623663, 0.0587868
9: -0.0195962, 0.0191127, -0.0234168, 0.0227018, -0.0422980, 0.0425295

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289763, upper bound: 0.0274104
time: 1.30 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281808, upper bound: 0.0274104
time: 1.29 seconds

## BFS NS instance: NS_A2_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0143482, 0.0077788, -0.0134992, 0.0061683, -0.0205166, 0.0212780
1: -0.0102643, 0.0037288, -0.0091218, 0.0030408, -0.0133051, 0.0128506
2: 0.0180822, 0.0633075, 0.0220790, 0.0608634, -0.0427812, 0.0412285
3: -0.0045305, 0.0144626, -0.0044177, 0.0127985, -0.0173290, 0.0188802
4: -0.0163659, 0.0140233, -0.0154115, 0.0121297, -0.0284956, 0.0294348
5: 0.0000581, 0.0259350, 0.0009917, 0.0248052, -0.0247471, 0.0249433
6: -0.0402533, 0.0173076, -0.0375910, 0.0152767, -0.0555300, 0.0548987
7: 0.9384574, 0.9812257, 0.9433403, 0.9809377, -0.0424803, 0.0378854
8: -0.0359549, 0.0273658, -0.0343766, 0.0234125, -0.0593674, 0.0617424
9: -0.0227641, 0.0220886, -0.0202291, 0.0197072, -0.0424713, 0.0423177

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.57 seconds

## Relational analysis of NS_A2_B1_A2_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.40 seconds

## BFS NS instance: NS_A2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0163244, 0.0115270, -0.0134813, 0.0061343, -0.0224587, 0.0250083
1: -0.0129234, 0.0053301, -0.0090977, 0.0030263, -0.0159497, 0.0144278
2: 0.0087798, 0.0689961, 0.0221635, 0.0608118, -0.0520319, 0.0468326
3: -0.0047930, 0.0183357, -0.0044153, 0.0127633, -0.0175564, 0.0227510
4: -0.0185871, 0.0184306, -0.0153914, 0.0120897, -0.0306768, 0.0338220
5: -0.0021148, 0.0285645, 0.0010115, 0.0247814, -0.0268961, 0.0275530
6: -0.0464498, 0.0220345, -0.0375348, 0.0152338, -0.0616836, 0.0595693
7: 0.9270921, 0.9818965, 0.9434436, 0.9809317, -0.0538396, 0.0384529
8: -0.0396286, 0.0365672, -0.0343432, 0.0233290, -0.0629575, 0.0709103
9: -0.0286642, 0.0276311, -0.0201755, 0.0196569, -0.0483210, 0.0478067

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A2_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0128112, 0.0048635, -0.0133380, 0.0058626, -0.0186739, 0.0182015
1: -0.0081961, 0.0024833, -0.0089049, 0.0029102, -0.0111063, 0.0113882
2: 0.0253174, 0.0588830, 0.0228377, 0.0603994, -0.0350820, 0.0360453
3: -0.0043263, 0.0114502, -0.0043963, 0.0124826, -0.0168088, 0.0158464
4: -0.0146383, 0.0105954, -0.0152304, 0.0117702, -0.0264085, 0.0258258
5: 0.0017482, 0.0238898, 0.0011690, 0.0245908, -0.0228426, 0.0227209
6: -0.0354339, 0.0136311, -0.0370856, 0.0148911, -0.0503250, 0.0507167
7: 0.9472969, 0.9807042, 0.9442673, 0.9808830, -0.0335861, 0.0364369
8: -0.0330976, 0.0202093, -0.0340769, 0.0226620, -0.0557596, 0.0542862
9: -0.0181751, 0.0177777, -0.0197479, 0.0192551, -0.0374302, 0.0375256

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283019, upper bound: 0.0285657
time: 1.22 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128112, 0.0048635, -0.0144326, 0.0079387, -0.0207500, 0.0192961
1: -0.0081961, 0.0024833, -0.0103778, 0.0037971, -0.0119932, 0.0128611
2: 0.0253174, 0.0588830, 0.0176853, 0.0635502, -0.0382328, 0.0411977
3: -0.0043263, 0.0114502, -0.0045417, 0.0146279, -0.0189541, 0.0159918
4: -0.0146383, 0.0105954, -0.0164607, 0.0142113, -0.0288496, 0.0270561
5: 0.0017482, 0.0238898, -0.0000346, 0.0260472, -0.0242990, 0.0239244
6: -0.0354339, 0.0136311, -0.0405177, 0.0175093, -0.0529432, 0.0541488
7: 0.9472969, 0.9807042, 0.9379725, 0.9812545, -0.0339575, 0.0427318
8: -0.0330976, 0.0202093, -0.0361117, 0.0277584, -0.0608560, 0.0563209
9: -0.0181751, 0.0177777, -0.0230158, 0.0223250, -0.0405002, 0.0407935

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.31 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0139148, 0.0069566, -0.0133655, 0.0059147, -0.0198294, 0.0203221
1: -0.0096810, 0.0033775, -0.0089418, 0.0029324, -0.0126134, 0.0123194
2: 0.0201227, 0.0620596, 0.0227085, 0.0604784, -0.0403557, 0.0393511
3: -0.0044729, 0.0136130, -0.0043999, 0.0125364, -0.0170093, 0.0180129
4: -0.0158786, 0.0130565, -0.0152612, 0.0118314, -0.0277101, 0.0283178
5: 0.0005348, 0.0253582, 0.0011388, 0.0246273, -0.0240925, 0.0242194
6: -0.0388941, 0.0162707, -0.0371717, 0.0149568, -0.0538509, 0.0534424
7: 0.9409504, 0.9810788, 0.9441096, 0.9808923, -0.0399419, 0.0369692
8: -0.0351491, 0.0253475, -0.0341280, 0.0227898, -0.0579389, 0.0594754
9: -0.0214699, 0.0208728, -0.0198298, 0.0193321, -0.0408020, 0.0407026

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
time: 1.19 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
time: 1.34 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0138900, 0.0069096, -0.0153432, 0.0096660, -0.0235560, 0.0222528
1: -0.0096477, 0.0033575, -0.0116031, 0.0045350, -0.0141827, 0.0149606
2: 0.0202393, 0.0619884, 0.0133986, 0.0661716, -0.0459322, 0.0485898
3: -0.0044696, 0.0135645, -0.0046627, 0.0164126, -0.0208822, 0.0182271
4: -0.0158508, 0.0130013, -0.0174842, 0.0162423, -0.0320931, 0.0304855
5: 0.0005620, 0.0253253, -0.0010359, 0.0272589, -0.0266969, 0.0263612
6: -0.0388165, 0.0162115, -0.0433731, 0.0196875, -0.0585040, 0.0595846
7: 0.9410927, 0.9810704, 0.9327351, 0.9815634, -0.0404707, 0.0483353
8: -0.0351030, 0.0252322, -0.0378045, 0.0319985, -0.0671016, 0.0630367
9: -0.0213959, 0.0208033, -0.0257346, 0.0248791, -0.0462750, 0.0465379

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.25 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0133744, 0.0059317, -0.0133225, 0.0058333, -0.0192077, 0.0192543
1: -0.0089539, 0.0029397, -0.0088841, 0.0028976, -0.0118516, 0.0118238
2: 0.0226662, 0.0605043, 0.0229105, 0.0603549, -0.0376887, 0.0375937
3: -0.0044011, 0.0125540, -0.0043942, 0.0124523, -0.0168534, 0.0169482
4: -0.0152713, 0.0118515, -0.0152130, 0.0117358, -0.0270071, 0.0270645
5: 0.0011289, 0.0246392, 0.0011860, 0.0245702, -0.0234413, 0.0234533
6: -0.0371998, 0.0149783, -0.0370371, 0.0148542, -0.0520540, 0.0520154
7: 0.9440578, 0.9808955, 0.9443563, 0.9808778, -0.0368200, 0.0365392
8: -0.0341446, 0.0228316, -0.0340482, 0.0225900, -0.0567346, 0.0568798
9: -0.0198567, 0.0193573, -0.0197017, 0.0192117, -0.0390684, 0.0390590

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0133497, 0.0058848, -0.0153028, 0.0095894, -0.0229391, 0.0211877
1: -0.0089207, 0.0029197, -0.0115488, 0.0045023, -0.0134230, 0.0144684
2: 0.0227826, 0.0604331, 0.0135886, 0.0660554, -0.0432728, 0.0468445
3: -0.0043978, 0.0125056, -0.0046573, 0.0163335, -0.0207313, 0.0171629
4: -0.0152435, 0.0117964, -0.0174389, 0.0161523, -0.0313958, 0.0292353
5: 0.0011561, 0.0246063, -0.0009915, 0.0272052, -0.0260491, 0.0255978
6: -0.0371223, 0.0149192, -0.0432466, 0.0195910, -0.0567133, 0.0581657
7: 0.9442000, 0.9808871, 0.9329674, 0.9815499, -0.0373498, 0.0479196
8: -0.0340987, 0.0227165, -0.0377295, 0.0318106, -0.0659093, 0.0604461
9: -0.0197829, 0.0192880, -0.0256142, 0.0247659, -0.0445488, 0.0449021

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0133155, 0.0058200, -0.0144516, 0.0079749, -0.0212904, 0.0202716
1: -0.0088747, 0.0028920, -0.0104034, 0.0038126, -0.0126872, 0.0132954
2: 0.0229435, 0.0603347, 0.0175954, 0.0636051, -0.0406616, 0.0427393
3: -0.0043933, 0.0124385, -0.0045442, 0.0146652, -0.0190585, 0.0169827
4: -0.0152051, 0.0117201, -0.0164821, 0.0142539, -0.0294590, 0.0282022
5: 0.0011937, 0.0245608, -0.0000556, 0.0260726, -0.0248789, 0.0246164
6: -0.0370151, 0.0148374, -0.0405775, 0.0175549, -0.0545700, 0.0554149
7: 0.9443967, 0.9808753, 0.9378626, 0.9812609, -0.0368642, 0.0430127
8: -0.0340351, 0.0225573, -0.0361472, 0.0278473, -0.0618824, 0.0587045
9: -0.0196808, 0.0191921, -0.0230728, 0.0223786, -0.0420594, 0.0422649

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0152935, 0.0095718, -0.0144260, 0.0079262, -0.0232198, 0.0239977
1: -0.0115363, 0.0044948, -0.0103689, 0.0037918, -0.0153281, 0.0148637
2: 0.0136324, 0.0660286, 0.0177163, 0.0635312, -0.0498988, 0.0483123
3: -0.0046561, 0.0163153, -0.0045408, 0.0146149, -0.0192710, 0.0208561
4: -0.0174284, 0.0161315, -0.0164533, 0.0141967, -0.0316251, 0.0325848
5: -0.0009813, 0.0271928, -0.0000273, 0.0260384, -0.0270197, 0.0272202
6: -0.0432174, 0.0195687, -0.0404971, 0.0174935, -0.0607110, 0.0600658
7: 0.9330209, 0.9815466, 0.9380102, 0.9812523, -0.0482314, 0.0435364
8: -0.0377122, 0.0317672, -0.0360995, 0.0277278, -0.0654400, 0.0678667
9: -0.0255864, 0.0247398, -0.0229962, 0.0223066, -0.0478930, 0.0477360

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.07 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.16 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.73 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0281711
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0281711
NS_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
NS_A1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
NS_A1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
NS_A1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
NS_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
NS_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
NS_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
NS_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
NS_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
NS_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
NS_A1_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285016
NS_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
NS_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
NS_A2_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
NS_A2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0289763, upper bound: 0.0274104
NS_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0281808, upper bound: 0.0274104
NS_A2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0289763, upper bound: 0.0274104
NS_A2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0281808, upper bound: 0.0274104
NS_A2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0283019, upper bound: 0.0285657
NS_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
NS_A2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
NS_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
NS_A2_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
NS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
NS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
NS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0134271, 0.0060316, -0.0189168, 0.0184307
1: -0.0082955, 0.0025432, -0.0090248, 0.0029824, -0.0112778, 0.0115680
2: 0.0249697, 0.0590957, 0.0224183, 0.0606559, -0.0356863, 0.0366774
3: -0.0043361, 0.0115950, -0.0044081, 0.0126572, -0.0169933, 0.0160030
4: -0.0147213, 0.0107602, -0.0153305, 0.0119690, -0.0266903, 0.0260907
5: 0.0016670, 0.0239881, 0.0010710, 0.0247093, -0.0230424, 0.0229171
6: -0.0356655, 0.0138078, -0.0373650, 0.0151043, -0.0507698, 0.0511728
7: 0.9468721, 0.9807293, 0.9437549, 0.9809133, -0.0340413, 0.0369745
8: -0.0332350, 0.0205532, -0.0342426, 0.0230769, -0.0563119, 0.0547958
9: -0.0183957, 0.0179849, -0.0200139, 0.0195050, -0.0379007, 0.0379988

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.32 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128688, 0.0049727, -0.0153757, 0.0097276, -0.0225964, 0.0203484
1: -0.0082735, 0.0025300, -0.0116469, 0.0045614, -0.0128349, 0.0141768
2: 0.0250464, 0.0590487, 0.0132456, 0.0662652, -0.0412188, 0.0458032
3: -0.0043339, 0.0115630, -0.0046670, 0.0164764, -0.0208103, 0.0162300
4: -0.0147030, 0.0107238, -0.0175208, 0.0163148, -0.0310178, 0.0282446
5: 0.0016849, 0.0239664, -0.0010716, 0.0273022, -0.0256173, 0.0250381
6: -0.0356143, 0.0137688, -0.0434751, 0.0197653, -0.0553796, 0.0572439
7: 0.9469659, 0.9807239, 0.9325481, 0.9815744, -0.0346085, 0.0481758
8: -0.0332047, 0.0204773, -0.0378650, 0.0321499, -0.0653546, 0.0583423
9: -0.0183470, 0.0179391, -0.0258318, 0.0249703, -0.0433173, 0.0437709

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0134271, 0.0060316, -0.0200577, 0.0205949
1: -0.0098308, 0.0034677, -0.0090248, 0.0029824, -0.0128132, 0.0124925
2: 0.0195986, 0.0623801, 0.0224183, 0.0606559, -0.0410573, 0.0399619
3: -0.0044877, 0.0138312, -0.0044081, 0.0126572, -0.0171449, 0.0182393
4: -0.0160038, 0.0133048, -0.0153305, 0.0119690, -0.0279727, 0.0286354
5: 0.0004124, 0.0255063, 0.0010710, 0.0247093, -0.0242969, 0.0244353
6: -0.0392432, 0.0165370, -0.0373650, 0.0151043, -0.0543475, 0.0539020
7: 0.9403101, 0.9811166, 0.9437549, 0.9809133, -0.0406032, 0.0373617
8: -0.0353561, 0.0258659, -0.0342426, 0.0230769, -0.0584330, 0.0601084
9: -0.0218022, 0.0211850, -0.0200139, 0.0195050, -0.0413073, 0.0411989

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0140097, 0.0071367, -0.0153757, 0.0097276, -0.0237374, 0.0225124
1: -0.0098088, 0.0034545, -0.0116469, 0.0045614, -0.0143702, 0.0151014
2: 0.0196757, 0.0623331, 0.0132456, 0.0662652, -0.0465896, 0.0490875
3: -0.0044855, 0.0137991, -0.0046670, 0.0164764, -0.0209619, 0.0184661
4: -0.0159854, 0.0132684, -0.0175208, 0.0163148, -0.0323002, 0.0307892
5: 0.0004303, 0.0254846, -0.0010716, 0.0273022, -0.0268718, 0.0265562
6: -0.0391919, 0.0164979, -0.0434751, 0.0197653, -0.0589572, 0.0599730
7: 0.9404041, 0.9811110, 0.9325481, 0.9815744, -0.0411704, 0.0485629
8: -0.0353257, 0.0257897, -0.0378650, 0.0321499, -0.0674756, 0.0636547
9: -0.0217535, 0.0211392, -0.0258318, 0.0249703, -0.0467238, 0.0469710

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0281711
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0281711
time: 2.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0128634, 0.0049623, -0.0145496, 0.0081607, -0.0210240, 0.0195119
1: -0.0082662, 0.0025256, -0.0105352, 0.0038919, -0.0121581, 0.0130608
2: 0.0250721, 0.0590331, 0.0171345, 0.0638870, -0.0388150, 0.0418986
3: -0.0043332, 0.0115523, -0.0045572, 0.0148572, -0.0191904, 0.0161095
4: -0.0146968, 0.0107116, -0.0165922, 0.0144723, -0.0291692, 0.0273038
5: 0.0016909, 0.0239592, -0.0001632, 0.0262029, -0.0245120, 0.0241224
6: -0.0355973, 0.0137558, -0.0408846, 0.0177892, -0.0533864, 0.0546404
7: 0.9469972, 0.9807220, 0.9372995, 0.9812942, -0.0342970, 0.0434225
8: -0.0331945, 0.0204519, -0.0363292, 0.0283033, -0.0614978, 0.0567811
9: -0.0183307, 0.0179238, -0.0233652, 0.0226532, -0.0409840, 0.0412890

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0140043, 0.0071264, -0.0145496, 0.0081607, -0.0221650, 0.0216760
1: -0.0098015, 0.0034501, -0.0105352, 0.0038919, -0.0136934, 0.0139853
2: 0.0197014, 0.0623174, 0.0171345, 0.0638870, -0.0441857, 0.0451829
3: -0.0044848, 0.0137885, -0.0045572, 0.0148572, -0.0193420, 0.0183457
4: -0.0159793, 0.0132562, -0.0165922, 0.0144723, -0.0304516, 0.0298484
5: 0.0004364, 0.0254773, -0.0001632, 0.0262029, -0.0257665, 0.0256406
6: -0.0391748, 0.0164849, -0.0408846, 0.0177892, -0.0569640, 0.0573695
7: 0.9404354, 0.9811090, 0.9372995, 0.9812942, -0.0408588, 0.0438095
8: -0.0353155, 0.0257643, -0.0363292, 0.0283033, -0.0636188, 0.0620935
9: -0.0217371, 0.0211239, -0.0233652, 0.0226532, -0.0443904, 0.0444890

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.49 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0148829, 0.0087929, -0.0140097, 0.0071367, -0.0220196, 0.0228026
1: -0.0109837, 0.0041620, -0.0098088, 0.0034545, -0.0144382, 0.0139708
2: 0.0155655, 0.0648465, 0.0196757, 0.0623331, -0.0467676, 0.0451709
3: -0.0046015, 0.0155104, -0.0044855, 0.0137991, -0.0184007, 0.0199959
4: -0.0169668, 0.0152157, -0.0159854, 0.0132684, -0.0302352, 0.0312011
5: -0.0005297, 0.0266464, 0.0004303, 0.0254846, -0.0260143, 0.0262160
6: -0.0419297, 0.0185864, -0.0391919, 0.0164979, -0.0584276, 0.0577784
7: 0.9353825, 0.9814073, 0.9404041, 0.9811110, -0.0457284, 0.0410033
8: -0.0369488, 0.0298552, -0.0353257, 0.0257897, -0.0627385, 0.0651809
9: -0.0243603, 0.0235881, -0.0217535, 0.0211392, -0.0454995, 0.0453415

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.28 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0148829, 0.0087929, -0.0144867, 0.0080415, -0.0229244, 0.0232796
1: -0.0109837, 0.0041620, -0.0104507, 0.0038410, -0.0148247, 0.0146127
2: 0.0155655, 0.0648465, 0.0174302, 0.0637062, -0.0481407, 0.0474163
3: -0.0046015, 0.0155104, -0.0045489, 0.0147340, -0.0193356, 0.0200593
4: -0.0169668, 0.0152157, -0.0165216, 0.0143322, -0.0312990, 0.0317373
5: -0.0005297, 0.0266464, -0.0000941, 0.0261193, -0.0266490, 0.0267405
6: -0.0419297, 0.0185864, -0.0406876, 0.0176389, -0.0595686, 0.0592740
7: 0.9353825, 0.9814073, 0.9376608, 0.9812729, -0.0458904, 0.0437465
8: -0.0369488, 0.0298552, -0.0362124, 0.0280107, -0.0649595, 0.0660676
9: -0.0243603, 0.0235881, -0.0231776, 0.0224770, -0.0468373, 0.0467656

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0134442, 0.0060640, -0.0194700, 0.0194358
1: -0.0089964, 0.0029653, -0.0090478, 0.0029962, -0.0119926, 0.0120131
2: 0.0225176, 0.0605952, 0.0223379, 0.0607050, -0.0381874, 0.0382572
3: -0.0044053, 0.0126159, -0.0044104, 0.0126907, -0.0170960, 0.0170262
4: -0.0153068, 0.0119219, -0.0153497, 0.0120070, -0.0273138, 0.0272716
5: 0.0010942, 0.0246813, 0.0010522, 0.0247320, -0.0236378, 0.0236290
6: -0.0372989, 0.0150538, -0.0374185, 0.0151451, -0.0524440, 0.0524724
7: 0.9438763, 0.9809060, 0.9436567, 0.9809191, -0.0370428, 0.0372493
8: -0.0342033, 0.0229786, -0.0342743, 0.0231563, -0.0573597, 0.0572529
9: -0.0199509, 0.0194459, -0.0200649, 0.0195529, -0.0395038, 0.0395108

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0295255
time: 1.48 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0295255
time: 1.28 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0133891, 0.0059596, -0.0153929, 0.0097603, -0.0231494, 0.0213525
1: -0.0089737, 0.0029516, -0.0116700, 0.0045753, -0.0135490, 0.0146216
2: 0.0225972, 0.0605465, 0.0131645, 0.0663147, -0.0437175, 0.0473820
3: -0.0044030, 0.0125828, -0.0046693, 0.0165101, -0.0209131, 0.0172520
4: -0.0152878, 0.0118842, -0.0175401, 0.0163532, -0.0316410, 0.0294243
5: 0.0011128, 0.0246588, -0.0010906, 0.0273251, -0.0262123, 0.0257493
6: -0.0372459, 0.0150134, -0.0435291, 0.0198064, -0.0570523, 0.0585425
7: 0.9439735, 0.9809004, 0.9324492, 0.9815803, -0.0376068, 0.0484512
8: -0.0341719, 0.0228999, -0.0378970, 0.0322301, -0.0664020, 0.0607969
9: -0.0199005, 0.0193985, -0.0258831, 0.0250186, -0.0449191, 0.0452816

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0145038, 0.0080739, -0.0134442, 0.0060640, -0.0205678, 0.0215181
1: -0.0104737, 0.0038549, -0.0090478, 0.0029962, -0.0134699, 0.0129026
2: 0.0173498, 0.0637554, 0.0223379, 0.0607050, -0.0433553, 0.0414174
3: -0.0045511, 0.0147675, -0.0044104, 0.0126907, -0.0172418, 0.0191779
4: -0.0165408, 0.0143703, -0.0153497, 0.0120070, -0.0285478, 0.0297200
5: -0.0001129, 0.0261420, 0.0010522, 0.0247320, -0.0248450, 0.0250898
6: -0.0407412, 0.0176798, -0.0374185, 0.0151451, -0.0558863, 0.0550983
7: 0.9375624, 0.9812785, 0.9436567, 0.9809191, -0.0433567, 0.0376219
8: -0.0362442, 0.0280903, -0.0342743, 0.0231563, -0.0594005, 0.0623646
9: -0.0232286, 0.0225250, -0.0200649, 0.0195529, -0.0427815, 0.0425899

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0144867, 0.0080415, -0.0153929, 0.0097603, -0.0242470, 0.0234344
1: -0.0104507, 0.0038410, -0.0116700, 0.0045753, -0.0150260, 0.0155111
2: 0.0174302, 0.0637062, 0.0131645, 0.0663147, -0.0488845, 0.0505417
3: -0.0045489, 0.0147340, -0.0046693, 0.0165101, -0.0210590, 0.0194033
4: -0.0165216, 0.0143322, -0.0175401, 0.0163532, -0.0328748, 0.0318723
5: -0.0000941, 0.0261193, -0.0010906, 0.0273251, -0.0274192, 0.0272099
6: -0.0406876, 0.0176389, -0.0435291, 0.0198064, -0.0604940, 0.0611680
7: 0.9376608, 0.9812729, 0.9324492, 0.9815803, -0.0439195, 0.0488238
8: -0.0362124, 0.0280107, -0.0378970, 0.0322301, -0.0684425, 0.0659077
9: -0.0231776, 0.0224770, -0.0258831, 0.0250186, -0.0481962, 0.0483602

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0134153, 0.0060092, -0.0140261, 0.0071677, -0.0205831, 0.0200353
1: -0.0090089, 0.0029728, -0.0098308, 0.0034677, -0.0124766, 0.0128036
2: 0.0224739, 0.0606219, 0.0195986, 0.0623801, -0.0399062, 0.0410233
3: -0.0044065, 0.0126341, -0.0044877, 0.0138312, -0.0182377, 0.0171218
4: -0.0153173, 0.0119426, -0.0160038, 0.0133048, -0.0286221, 0.0279464
5: 0.0010840, 0.0246936, 0.0004124, 0.0255063, -0.0244223, 0.0242812
6: -0.0373280, 0.0150760, -0.0392432, 0.0165370, -0.0538650, 0.0543192
7: 0.9438229, 0.9809093, 0.9403101, 0.9811166, -0.0372937, 0.0405992
8: -0.0342206, 0.0230219, -0.0353561, 0.0258659, -0.0600865, 0.0583779
9: -0.0199786, 0.0194719, -0.0218022, 0.0211850, -0.0411637, 0.0412741

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0134153, 0.0060092, -0.0145038, 0.0080739, -0.0214892, 0.0205131
1: -0.0090089, 0.0029728, -0.0104737, 0.0038549, -0.0128638, 0.0134465
2: 0.0224739, 0.0606219, 0.0173498, 0.0637554, -0.0412815, 0.0432721
3: -0.0044065, 0.0126341, -0.0045511, 0.0147675, -0.0191741, 0.0171852
4: -0.0153173, 0.0119426, -0.0165408, 0.0143703, -0.0296876, 0.0284834
5: 0.0010840, 0.0246936, -0.0001129, 0.0261420, -0.0250581, 0.0248065
6: -0.0373280, 0.0150760, -0.0407412, 0.0176798, -0.0550077, 0.0558172
7: 0.9438229, 0.9809093, 0.9375624, 0.9812785, -0.0374557, 0.0433469
8: -0.0342206, 0.0230219, -0.0362442, 0.0280903, -0.0623109, 0.0592660
9: -0.0199786, 0.0194719, -0.0232286, 0.0225250, -0.0425036, 0.0427005

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0153611, 0.0096999, -0.0140097, 0.0071367, -0.0224978, 0.0237097
1: -0.0116272, 0.0045495, -0.0098088, 0.0034545, -0.0150817, 0.0143583
2: 0.0133143, 0.0662231, 0.0196757, 0.0623331, -0.0490187, 0.0465475
3: -0.0046650, 0.0164477, -0.0044855, 0.0137991, -0.0184642, 0.0209332
4: -0.0175044, 0.0162822, -0.0159854, 0.0132684, -0.0307727, 0.0322676
5: -0.0010556, 0.0272827, 0.0004303, 0.0254846, -0.0265402, 0.0268524
6: -0.0434292, 0.0197303, -0.0391919, 0.0164979, -0.0599271, 0.0589223
7: 0.9326323, 0.9815695, 0.9404041, 0.9811110, -0.0484787, 0.0411654
8: -0.0378378, 0.0320819, -0.0353257, 0.0257897, -0.0636275, 0.0674075
9: -0.0257881, 0.0249294, -0.0217535, 0.0211392, -0.0469273, 0.0466828

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0153611, 0.0096999, -0.0144867, 0.0080415, -0.0234026, 0.0241867
1: -0.0116272, 0.0045495, -0.0104507, 0.0038410, -0.0154682, 0.0150002
2: 0.0133143, 0.0662231, 0.0174302, 0.0637062, -0.0503918, 0.0487929
3: -0.0046650, 0.0164477, -0.0045489, 0.0147340, -0.0193991, 0.0209966
4: -0.0175044, 0.0162822, -0.0165216, 0.0143322, -0.0318366, 0.0328038
5: -0.0010556, 0.0272827, -0.0000941, 0.0261193, -0.0271748, 0.0273769
6: -0.0434292, 0.0197303, -0.0406876, 0.0176389, -0.0610681, 0.0604179
7: 0.9326323, 0.9815695, 0.9376608, 0.9812729, -0.0486406, 0.0439087
8: -0.0378378, 0.0320819, -0.0362124, 0.0280107, -0.0658485, 0.0682943
9: -0.0257881, 0.0249294, -0.0231776, 0.0224770, -0.0482651, 0.0481069

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.25 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0134507, 0.0060765, -0.0127776, 0.0048254, -0.0182761, 0.0188541
1: -0.0090566, 0.0030015, -0.0081709, 0.0024561, -0.0115126, 0.0111724
2: 0.0223071, 0.0607240, 0.0253886, 0.0587862, -0.0364791, 0.0353354
3: -0.0044112, 0.0127035, -0.0043218, 0.0113997, -0.0158109, 0.0170253
4: -0.0153571, 0.0120217, -0.0146005, 0.0105412, -0.0258983, 0.0266221
5: 0.0010450, 0.0247408, 0.0017795, 0.0238451, -0.0228001, 0.0229613
6: -0.0374392, 0.0151608, -0.0353284, 0.0135622, -0.0510014, 0.0504892
7: 0.9436190, 0.9809213, 0.9474486, 0.9806929, -0.0370739, 0.0334727
8: -0.0342865, 0.0231869, -0.0330351, 0.0200919, -0.0543784, 0.0562220
9: -0.0200845, 0.0195713, -0.0180747, 0.0177138, -0.0377983, 0.0376460

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
time: 1.48 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
time: 2.31 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0145496, 0.0081607, -0.0127776, 0.0048254, -0.0193750, 0.0209383
1: -0.0105352, 0.0038919, -0.0081709, 0.0024561, -0.0129913, 0.0120628
2: 0.0171345, 0.0638870, 0.0253886, 0.0587862, -0.0416517, 0.0384984
3: -0.0045572, 0.0148572, -0.0043218, 0.0113997, -0.0159569, 0.0191790
4: -0.0165922, 0.0144723, -0.0146005, 0.0105412, -0.0271334, 0.0290728
5: -0.0001632, 0.0262029, 0.0017795, 0.0238451, -0.0240083, 0.0244234
6: -0.0408846, 0.0177892, -0.0353284, 0.0135622, -0.0544468, 0.0531176
7: 0.9372995, 0.9812942, 0.9474486, 0.9806929, -0.0433934, 0.0338456
8: -0.0363292, 0.0283033, -0.0330351, 0.0200919, -0.0564212, 0.0613384
9: -0.0233652, 0.0226532, -0.0180747, 0.0177138, -0.0410790, 0.0407279

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
time: 1.40 seconds

## Relational analysis of NS_A1_B2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0134328, 0.0060425, -0.0147582, 0.0085563, -0.0219892, 0.0208007
1: -0.0090325, 0.0029870, -0.0108159, 0.0040610, -0.0130935, 0.0138030
2: 0.0223912, 0.0606724, 0.0161525, 0.0644875, -0.0420963, 0.0445200
3: -0.0044089, 0.0126685, -0.0045849, 0.0152660, -0.0196749, 0.0172534
4: -0.0153370, 0.0119818, -0.0168267, 0.0149376, -0.0302745, 0.0288084
5: 0.0010647, 0.0247169, -0.0003926, 0.0264805, -0.0254158, 0.0251096
6: -0.0373830, 0.0151180, -0.0415387, 0.0182881, -0.0556711, 0.0566567
7: 0.9437219, 0.9809152, 0.9360997, 0.9813651, -0.0376432, 0.0448155
8: -0.0342532, 0.0231036, -0.0367170, 0.0292746, -0.0635278, 0.0598206
9: -0.0200310, 0.0195211, -0.0239880, 0.0232384, -0.0432694, 0.0435092

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
time: 1.26 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
time: 1.65 seconds

## BFS NS instance: NS_A1_B2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0145315, 0.0081264, -0.0147582, 0.0085563, -0.0230878, 0.0228845
1: -0.0105109, 0.0038773, -0.0108159, 0.0040610, -0.0145718, 0.0146932
2: 0.0172196, 0.0638350, 0.0161525, 0.0644875, -0.0472679, 0.0476825
3: -0.0045548, 0.0148217, -0.0045849, 0.0152660, -0.0198209, 0.0194067
4: -0.0165719, 0.0144320, -0.0168267, 0.0149376, -0.0315094, 0.0312587
5: -0.0001433, 0.0261788, -0.0003926, 0.0264805, -0.0266238, 0.0265714
6: -0.0408279, 0.0177459, -0.0415387, 0.0182881, -0.0591161, 0.0592847
7: 0.9374034, 0.9812880, 0.9360997, 0.9813651, -0.0439616, 0.0451882
8: -0.0362956, 0.0282190, -0.0367170, 0.0292746, -0.0655702, 0.0649360
9: -0.0233112, 0.0226025, -0.0239880, 0.0232384, -0.0465496, 0.0465905

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
time: 1.34 seconds

## Relational analysis of NS_A1_B2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
time: 1.23 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0129184, 0.0050667, -0.0138813, 0.0068931, -0.0198115, 0.0189480
1: -0.0083402, 0.0025701, -0.0096360, 0.0033504, -0.0116906, 0.0122061
2: 0.0248131, 0.0591914, 0.0202803, 0.0619633, -0.0371502, 0.0389111
3: -0.0043405, 0.0116601, -0.0044684, 0.0135474, -0.0178879, 0.0161286
4: -0.0147587, 0.0108344, -0.0158410, 0.0129819, -0.0277406, 0.0266754
5: 0.0016304, 0.0240324, 0.0005716, 0.0253137, -0.0236833, 0.0234608
6: -0.0357698, 0.0138874, -0.0387891, 0.0161906, -0.0519604, 0.0526765
7: 0.9466808, 0.9807407, 0.9411429, 0.9810674, -0.0343866, 0.0395977
8: -0.0332968, 0.0207081, -0.0350868, 0.0251916, -0.0584884, 0.0557949
9: -0.0184950, 0.0180782, -0.0213699, 0.0207789, -0.0392738, 0.0394481

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
time: 1.60 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
time: 1.35 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0134376, 0.0060516, -0.0138813, 0.0068931, -0.0203307, 0.0199329
1: -0.0090389, 0.0029909, -0.0096360, 0.0033504, -0.0123894, 0.0126269
2: 0.0223688, 0.0606862, 0.0202803, 0.0619633, -0.0395945, 0.0404059
3: -0.0044095, 0.0126778, -0.0044684, 0.0135474, -0.0179569, 0.0171463
4: -0.0153423, 0.0119924, -0.0158410, 0.0129819, -0.0283242, 0.0278334
5: 0.0010594, 0.0247233, 0.0005716, 0.0253137, -0.0242542, 0.0241517
6: -0.0373980, 0.0151294, -0.0387891, 0.0161906, -0.0535887, 0.0539185
7: 0.9436943, 0.9809169, 0.9411429, 0.9810674, -0.0373731, 0.0397739
8: -0.0342621, 0.0231258, -0.0350868, 0.0251916, -0.0594537, 0.0582127
9: -0.0200453, 0.0195345, -0.0213699, 0.0207789, -0.0408242, 0.0409044

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
time: 1.42 seconds

## Relational analysis of NS_A1_B2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
time: 1.35 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0129021, 0.0050359, -0.0158551, 0.0106369, -0.0235391, 0.0208910
1: -0.0083184, 0.0025570, -0.0122919, 0.0049498, -0.0132682, 0.0148489
2: 0.0248895, 0.0591447, 0.0109890, 0.0676451, -0.0427556, 0.0481558
3: -0.0043383, 0.0116283, -0.0047307, 0.0174159, -0.0217543, 0.0163590
4: -0.0147405, 0.0107982, -0.0180596, 0.0173839, -0.0321244, 0.0288578
5: 0.0016482, 0.0240108, -0.0015988, 0.0279401, -0.0262918, 0.0256095
6: -0.0357189, 0.0138486, -0.0449782, 0.0209119, -0.0566309, 0.0588268
7: 0.9467739, 0.9807352, 0.9297912, 0.9817372, -0.0349633, 0.0509440
8: -0.0332667, 0.0206325, -0.0387562, 0.0343820, -0.0676487, 0.0593887
9: -0.0184466, 0.0180326, -0.0272630, 0.0263149, -0.0447614, 0.0452957

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.26 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.41 seconds

## BFS NS instance: NS_A1_B2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0134208, 0.0060196, -0.0158551, 0.0106369, -0.0240577, 0.0218747
1: -0.0090163, 0.0029772, -0.0122919, 0.0049498, -0.0139661, 0.0152692
2: 0.0224481, 0.0606377, 0.0109890, 0.0676451, -0.0451970, 0.0496487
3: -0.0044072, 0.0126448, -0.0047307, 0.0174159, -0.0218232, 0.0173755
4: -0.0153234, 0.0119548, -0.0180596, 0.0173839, -0.0327073, 0.0300144
5: 0.0010780, 0.0247009, -0.0015988, 0.0279401, -0.0268621, 0.0262996
6: -0.0373451, 0.0150891, -0.0449782, 0.0209119, -0.0582571, 0.0600673
7: 0.9437914, 0.9809111, 0.9297912, 0.9817372, -0.0379458, 0.0511199
8: -0.0342308, 0.0230474, -0.0387562, 0.0343820, -0.0686128, 0.0618035
9: -0.0199950, 0.0194872, -0.0272630, 0.0263149, -0.0463099, 0.0467503

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.38 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.26 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0134682, 0.0061095, -0.0132535, 0.0057022, -0.0191704, 0.0193629
1: -0.0090800, 0.0030156, -0.0087911, 0.0028416, -0.0119217, 0.0118067
2: 0.0222251, 0.0607741, 0.0232358, 0.0601560, -0.0379309, 0.0375383
3: -0.0044135, 0.0127377, -0.0043850, 0.0123169, -0.0167304, 0.0171227
4: -0.0153767, 0.0120605, -0.0151353, 0.0115816, -0.0269583, 0.0271958
5: 0.0010259, 0.0247639, 0.0012620, 0.0244782, -0.0234524, 0.0235020
6: -0.0374937, 0.0152025, -0.0368205, 0.0146889, -0.0521826, 0.0520230
7: 0.9435188, 0.9809272, 0.9447536, 0.9808543, -0.0373355, 0.0361736
8: -0.0343189, 0.0232680, -0.0339197, 0.0222682, -0.0565871, 0.0571877
9: -0.0201365, 0.0196202, -0.0194954, 0.0190179, -0.0391544, 0.0391156

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0294413
time: 1.62 seconds

## Relational analysis of NS_A1_B2_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0295255
time: 1.39 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0134501, 0.0060752, -0.0152331, 0.0094570, -0.0229072, 0.0213083
1: -0.0090557, 0.0030010, -0.0114549, 0.0044458, -0.0135015, 0.0144559
2: 0.0223100, 0.0607221, 0.0139171, 0.0658545, -0.0435445, 0.0468050
3: -0.0044111, 0.0127023, -0.0046480, 0.0161968, -0.0206079, 0.0173504
4: -0.0153564, 0.0120202, -0.0173604, 0.0159966, -0.0313530, 0.0293807
5: 0.0010457, 0.0247399, -0.0009148, 0.0271123, -0.0260666, 0.0256547
6: -0.0374371, 0.0151593, -0.0430278, 0.0194240, -0.0568611, 0.0581871
7: 0.9436226, 0.9809211, 0.9333686, 0.9815261, -0.0379035, 0.0475525
8: -0.0342853, 0.0231839, -0.0375998, 0.0314857, -0.0657710, 0.0607837
9: -0.0200826, 0.0195695, -0.0254058, 0.0245702, -0.0446528, 0.0449754

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.37 seconds

## Relational analysis of NS_A1_B2_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.42 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0145669, 0.0081935, -0.0132535, 0.0057022, -0.0202691, 0.0214470
1: -0.0105585, 0.0039060, -0.0087911, 0.0028416, -0.0134002, 0.0126971
2: 0.0170529, 0.0639369, 0.0232358, 0.0601560, -0.0431031, 0.0407011
3: -0.0045595, 0.0148911, -0.0043850, 0.0123169, -0.0168764, 0.0192761
4: -0.0166117, 0.0145109, -0.0151353, 0.0115816, -0.0281933, 0.0296462
5: -0.0001823, 0.0262259, 0.0012620, 0.0244782, -0.0246605, 0.0249640
6: -0.0409389, 0.0178306, -0.0368205, 0.0146889, -0.0556278, 0.0546511
7: 0.9371998, 0.9813001, 0.9447536, 0.9808543, -0.0436545, 0.0365464
8: -0.0363614, 0.0283839, -0.0339197, 0.0222682, -0.0586296, 0.0623036
9: -0.0234168, 0.0227018, -0.0194954, 0.0190179, -0.0424348, 0.0421972

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289034
time: 1.70 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
time: 1.38 seconds

## BFS NS instance: NS_A1_B2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0145487, 0.0081590, -0.0152331, 0.0094570, -0.0240057, 0.0233921
1: -0.0105340, 0.0038912, -0.0114549, 0.0044458, -0.0149798, 0.0153461
2: 0.0171386, 0.0638845, 0.0139171, 0.0658545, -0.0487159, 0.0499674
3: -0.0045571, 0.0148555, -0.0046480, 0.0161968, -0.0207539, 0.0195035
4: -0.0165912, 0.0144704, -0.0173604, 0.0159966, -0.0325879, 0.0318308
5: -0.0001623, 0.0262017, -0.0009148, 0.0271123, -0.0272746, 0.0271165
6: -0.0408819, 0.0177871, -0.0430278, 0.0194240, -0.0603059, 0.0608149
7: 0.9373045, 0.9812939, 0.9333686, 0.9815261, -0.0442216, 0.0479253
8: -0.0363276, 0.0282992, -0.0375998, 0.0314857, -0.0678133, 0.0658990
9: -0.0233626, 0.0226508, -0.0254058, 0.0245702, -0.0479328, 0.0480566

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281711
time: 1.34 seconds

## Relational analysis of NS_A1_B2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
time: 1.28 seconds

## BFS NS instance: NS_A1_B2_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0129184, 0.0050667, -0.0143482, 0.0077788, -0.0206972, 0.0194149
1: -0.0083402, 0.0025701, -0.0102643, 0.0037288, -0.0120690, 0.0128344
2: 0.0248131, 0.0591914, 0.0180822, 0.0633075, -0.0384944, 0.0411092
3: -0.0043405, 0.0116601, -0.0045305, 0.0144626, -0.0188031, 0.0161906
4: -0.0147587, 0.0108344, -0.0163659, 0.0140233, -0.0287820, 0.0272002
5: 0.0016304, 0.0240324, 0.0000581, 0.0259350, -0.0243046, 0.0239742
6: -0.0357698, 0.0138874, -0.0402533, 0.0173076, -0.0530774, 0.0541407
7: 0.9466808, 0.9807407, 0.9384574, 0.9812257, -0.0345449, 0.0422833
8: -0.0332968, 0.0207081, -0.0359549, 0.0273658, -0.0606627, 0.0566630
9: -0.0184950, 0.0180782, -0.0227641, 0.0220886, -0.0405836, 0.0408423

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285016
time: 1.57 seconds

## Relational analysis of NS_A1_B2_B2_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285016
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0134376, 0.0060516, -0.0143482, 0.0077788, -0.0212164, 0.0203998
1: -0.0090389, 0.0029909, -0.0102643, 0.0037288, -0.0127677, 0.0132552
2: 0.0223688, 0.0606862, 0.0180822, 0.0633075, -0.0409387, 0.0426040
3: -0.0044095, 0.0126778, -0.0045305, 0.0144626, -0.0188721, 0.0172083
4: -0.0153423, 0.0119924, -0.0163659, 0.0140233, -0.0293656, 0.0283583
5: 0.0010594, 0.0247233, 0.0000581, 0.0259350, -0.0248756, 0.0246652
6: -0.0373980, 0.0151294, -0.0402533, 0.0173076, -0.0547056, 0.0553828
7: 0.9436943, 0.9809169, 0.9384574, 0.9812257, -0.0375314, 0.0424595
8: -0.0342621, 0.0231258, -0.0359549, 0.0273658, -0.0616279, 0.0590808
9: -0.0200453, 0.0195345, -0.0227641, 0.0220886, -0.0421339, 0.0422986

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
time: 1.27 seconds

## Relational analysis of NS_A1_B2_B2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0129021, 0.0050359, -0.0163244, 0.0115270, -0.0244292, 0.0213603
1: -0.0083184, 0.0025570, -0.0129234, 0.0053301, -0.0136485, 0.0154804
2: 0.0248895, 0.0591447, 0.0087798, 0.0689961, -0.0441066, 0.0503649
3: -0.0043383, 0.0116283, -0.0047930, 0.0183357, -0.0226740, 0.0164214
4: -0.0147405, 0.0107982, -0.0185871, 0.0184306, -0.0331710, 0.0293853
5: 0.0016482, 0.0240108, -0.0021148, 0.0285645, -0.0269163, 0.0261256
6: -0.0357189, 0.0138486, -0.0464498, 0.0220345, -0.0577534, 0.0602984
7: 0.9467739, 0.9807352, 0.9270921, 0.9818965, -0.0351225, 0.0536431
8: -0.0332667, 0.0206325, -0.0396286, 0.0365672, -0.0698338, 0.0602611
9: -0.0184466, 0.0180326, -0.0286642, 0.0276311, -0.0460777, 0.0466968

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_B2_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.37 seconds

## Relational analysis of NS_A1_B2_B2_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.41 seconds

## BFS NS instance: NS_A1_B2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0134208, 0.0060196, -0.0163244, 0.0115270, -0.0249478, 0.0223440
1: -0.0090163, 0.0029772, -0.0129234, 0.0053301, -0.0143464, 0.0159007
2: 0.0224481, 0.0606377, 0.0087798, 0.0689961, -0.0465479, 0.0518579
3: -0.0044072, 0.0126448, -0.0047930, 0.0183357, -0.0227429, 0.0174378
4: -0.0153234, 0.0119548, -0.0185871, 0.0184306, -0.0337540, 0.0305419
5: 0.0010780, 0.0247009, -0.0021148, 0.0285645, -0.0274866, 0.0268157
6: -0.0373451, 0.0150891, -0.0464498, 0.0220345, -0.0593796, 0.0615389
7: 0.9437914, 0.9809111, 0.9270921, 0.9818965, -0.0381051, 0.0538191
8: -0.0342308, 0.0230474, -0.0396286, 0.0365672, -0.0707979, 0.0626760
9: -0.0199950, 0.0194872, -0.0286642, 0.0276311, -0.0476261, 0.0481514

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_B2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.40 seconds

## Relational analysis of NS_A1_B2_B2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 2.19 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0127776, 0.0048254, -0.0134507, 0.0060765, -0.0188541, 0.0182761
1: -0.0081709, 0.0024561, -0.0090566, 0.0030015, -0.0111724, 0.0115126
2: 0.0253886, 0.0587862, 0.0223071, 0.0607240, -0.0353354, 0.0364791
3: -0.0043218, 0.0113997, -0.0044112, 0.0127035, -0.0170253, 0.0158109
4: -0.0146005, 0.0105412, -0.0153571, 0.0120217, -0.0266221, 0.0258983
5: 0.0017795, 0.0238451, 0.0010450, 0.0247408, -0.0229613, 0.0228001
6: -0.0353284, 0.0135622, -0.0374392, 0.0151608, -0.0504892, 0.0510014
7: 0.9474486, 0.9806929, 0.9436190, 0.9809213, -0.0334727, 0.0370739
8: -0.0330351, 0.0200919, -0.0342865, 0.0231869, -0.0562220, 0.0543784
9: -0.0180747, 0.0177138, -0.0200845, 0.0195713, -0.0376460, 0.0377983

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_A1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
time: 1.60 seconds

## Relational analysis of NS_A2_B1_A1_A1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
time: 2.57 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0127776, 0.0048254, -0.0145496, 0.0081607, -0.0209383, 0.0193750
1: -0.0081709, 0.0024561, -0.0105352, 0.0038919, -0.0120628, 0.0129913
2: 0.0253886, 0.0587862, 0.0171345, 0.0638870, -0.0384984, 0.0416517
3: -0.0043218, 0.0113997, -0.0045572, 0.0148572, -0.0191790, 0.0159569
4: -0.0146005, 0.0105412, -0.0165922, 0.0144723, -0.0290728, 0.0271334
5: 0.0017795, 0.0238451, -0.0001632, 0.0262029, -0.0244234, 0.0240083
6: -0.0353284, 0.0135622, -0.0408846, 0.0177892, -0.0531176, 0.0544468
7: 0.9474486, 0.9806929, 0.9372995, 0.9812942, -0.0338456, 0.0433934
8: -0.0330351, 0.0200919, -0.0363292, 0.0283033, -0.0613384, 0.0564212
9: -0.0180747, 0.0177138, -0.0233652, 0.0226532, -0.0407279, 0.0410790

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
time: 1.25 seconds

## Relational analysis of NS_A2_B1_A1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0147582, 0.0085563, -0.0134328, 0.0060425, -0.0208007, 0.0219892
1: -0.0108159, 0.0040610, -0.0090325, 0.0029870, -0.0138030, 0.0130935
2: 0.0161525, 0.0644875, 0.0223912, 0.0606724, -0.0445200, 0.0420963
3: -0.0045849, 0.0152660, -0.0044089, 0.0126685, -0.0172534, 0.0196749
4: -0.0168267, 0.0149376, -0.0153370, 0.0119818, -0.0288084, 0.0302745
5: -0.0003926, 0.0264805, 0.0010647, 0.0247169, -0.0251096, 0.0254158
6: -0.0415387, 0.0182881, -0.0373830, 0.0151180, -0.0566567, 0.0556711
7: 0.9360997, 0.9813651, 0.9437219, 0.9809152, -0.0448155, 0.0376432
8: -0.0367170, 0.0292746, -0.0342532, 0.0231036, -0.0598206, 0.0635278
9: -0.0239880, 0.0232384, -0.0200310, 0.0195211, -0.0435092, 0.0432694

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_A1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A1_A1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0147582, 0.0085563, -0.0145315, 0.0081264, -0.0228845, 0.0230878
1: -0.0108159, 0.0040610, -0.0105109, 0.0038773, -0.0146932, 0.0145718
2: 0.0161525, 0.0644875, 0.0172196, 0.0638350, -0.0476825, 0.0472679
3: -0.0045849, 0.0152660, -0.0045548, 0.0148217, -0.0194067, 0.0198209
4: -0.0168267, 0.0149376, -0.0165719, 0.0144320, -0.0312587, 0.0315094
5: -0.0003926, 0.0264805, -0.0001433, 0.0261788, -0.0265714, 0.0266238
6: -0.0415387, 0.0182881, -0.0408279, 0.0177459, -0.0592847, 0.0591161
7: 0.9360997, 0.9813651, 0.9374034, 0.9812880, -0.0451882, 0.0439616
8: -0.0367170, 0.0292746, -0.0362956, 0.0282190, -0.0649360, 0.0655702
9: -0.0239880, 0.0232384, -0.0233112, 0.0226025, -0.0465905, 0.0465496

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_A1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A1_A1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0138813, 0.0068931, -0.0129184, 0.0050667, -0.0189480, 0.0198115
1: -0.0096360, 0.0033504, -0.0083402, 0.0025701, -0.0122061, 0.0116906
2: 0.0202803, 0.0619633, 0.0248131, 0.0591914, -0.0389111, 0.0371502
3: -0.0044684, 0.0135474, -0.0043405, 0.0116601, -0.0161286, 0.0178879
4: -0.0158410, 0.0129819, -0.0147587, 0.0108344, -0.0266754, 0.0277406
5: 0.0005716, 0.0253137, 0.0016304, 0.0240324, -0.0234608, 0.0236833
6: -0.0387891, 0.0161906, -0.0357698, 0.0138874, -0.0526765, 0.0519604
7: 0.9411429, 0.9810674, 0.9466808, 0.9807407, -0.0395977, 0.0343866
8: -0.0350868, 0.0251916, -0.0332968, 0.0207081, -0.0557949, 0.0584884
9: -0.0213699, 0.0207789, -0.0184950, 0.0180782, -0.0394481, 0.0392738

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A2_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_A2_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.36 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0138813, 0.0068931, -0.0134376, 0.0060516, -0.0199329, 0.0203307
1: -0.0096360, 0.0033504, -0.0090389, 0.0029909, -0.0126269, 0.0123894
2: 0.0202803, 0.0619633, 0.0223688, 0.0606862, -0.0404059, 0.0395945
3: -0.0044684, 0.0135474, -0.0044095, 0.0126778, -0.0171463, 0.0179569
4: -0.0158410, 0.0129819, -0.0153423, 0.0119924, -0.0278334, 0.0283242
5: 0.0005716, 0.0253137, 0.0010594, 0.0247233, -0.0241517, 0.0242542
6: -0.0387891, 0.0161906, -0.0373980, 0.0151294, -0.0539185, 0.0535887
7: 0.9411429, 0.9810674, 0.9436943, 0.9809169, -0.0397739, 0.0373731
8: -0.0350868, 0.0251916, -0.0342621, 0.0231258, -0.0582127, 0.0594537
9: -0.0213699, 0.0207789, -0.0200453, 0.0195345, -0.0409044, 0.0408242

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_A2_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A1_A2_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0158551, 0.0106369, -0.0129021, 0.0050359, -0.0208910, 0.0235391
1: -0.0122919, 0.0049498, -0.0083184, 0.0025570, -0.0148489, 0.0132682
2: 0.0109890, 0.0676451, 0.0248895, 0.0591447, -0.0481558, 0.0427556
3: -0.0047307, 0.0174159, -0.0043383, 0.0116283, -0.0163590, 0.0217543
4: -0.0180596, 0.0173839, -0.0147405, 0.0107982, -0.0288578, 0.0321244
5: -0.0015988, 0.0279401, 0.0016482, 0.0240108, -0.0256095, 0.0262918
6: -0.0449782, 0.0209119, -0.0357189, 0.0138486, -0.0588268, 0.0566309
7: 0.9297912, 0.9817372, 0.9467739, 0.9807352, -0.0509440, 0.0349633
8: -0.0387562, 0.0343820, -0.0332667, 0.0206325, -0.0593887, 0.0676487
9: -0.0272630, 0.0263149, -0.0184466, 0.0180326, -0.0452957, 0.0447614

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A1_A2_A2_B1_B1

### Relational analysis result of NS_A2_B1_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A1_A2_A2_B1_B2

### Relational analysis result of NS_A2_B1_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0158551, 0.0106369, -0.0134208, 0.0060196, -0.0218747, 0.0240577
1: -0.0122919, 0.0049498, -0.0090163, 0.0029772, -0.0152692, 0.0139661
2: 0.0109890, 0.0676451, 0.0224481, 0.0606377, -0.0496487, 0.0451970
3: -0.0047307, 0.0174159, -0.0044072, 0.0126448, -0.0173755, 0.0218232
4: -0.0180596, 0.0173839, -0.0153234, 0.0119548, -0.0300144, 0.0327073
5: -0.0015988, 0.0279401, 0.0010780, 0.0247009, -0.0262996, 0.0268621
6: -0.0449782, 0.0209119, -0.0373451, 0.0150891, -0.0600673, 0.0582571
7: 0.9297912, 0.9817372, 0.9437914, 0.9809111, -0.0511199, 0.0379458
8: -0.0387562, 0.0343820, -0.0342308, 0.0230474, -0.0618035, 0.0686128
9: -0.0272630, 0.0263149, -0.0199950, 0.0194872, -0.0467503, 0.0463099

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A1_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A1_A2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0132535, 0.0057022, -0.0134682, 0.0061095, -0.0193629, 0.0191704
1: -0.0087911, 0.0028416, -0.0090800, 0.0030156, -0.0118067, 0.0119217
2: 0.0232358, 0.0601560, 0.0222251, 0.0607741, -0.0375383, 0.0379309
3: -0.0043850, 0.0123169, -0.0044135, 0.0127377, -0.0171227, 0.0167304
4: -0.0151353, 0.0115816, -0.0153767, 0.0120605, -0.0271958, 0.0269583
5: 0.0012620, 0.0244782, 0.0010259, 0.0247639, -0.0235020, 0.0234524
6: -0.0368205, 0.0146889, -0.0374937, 0.0152025, -0.0520230, 0.0521826
7: 0.9447536, 0.9808543, 0.9435188, 0.9809272, -0.0361736, 0.0373355
8: -0.0339197, 0.0222682, -0.0343189, 0.0232680, -0.0571877, 0.0565871
9: -0.0194954, 0.0190179, -0.0201365, 0.0196202, -0.0391156, 0.0391544

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
time: 1.63 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0152331, 0.0094570, -0.0134501, 0.0060752, -0.0213083, 0.0229072
1: -0.0114549, 0.0044458, -0.0090557, 0.0030010, -0.0144559, 0.0135015
2: 0.0139171, 0.0658545, 0.0223100, 0.0607221, -0.0468050, 0.0435445
3: -0.0046480, 0.0161968, -0.0044111, 0.0127023, -0.0173504, 0.0206079
4: -0.0173604, 0.0159966, -0.0153564, 0.0120202, -0.0293807, 0.0313530
5: -0.0009148, 0.0271123, 0.0010457, 0.0247399, -0.0256547, 0.0260666
6: -0.0430278, 0.0194240, -0.0374371, 0.0151593, -0.0581871, 0.0568611
7: 0.9333686, 0.9815261, 0.9436226, 0.9809211, -0.0475525, 0.0379035
8: -0.0375998, 0.0314857, -0.0342853, 0.0231839, -0.0607837, 0.0657710
9: -0.0254058, 0.0245702, -0.0200826, 0.0195695, -0.0449754, 0.0446528

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.40 seconds

## Relational analysis of NS_A2_B1_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0132535, 0.0057022, -0.0145669, 0.0081935, -0.0214470, 0.0202691
1: -0.0087911, 0.0028416, -0.0105585, 0.0039060, -0.0126971, 0.0134002
2: 0.0232358, 0.0601560, 0.0170529, 0.0639369, -0.0407011, 0.0431031
3: -0.0043850, 0.0123169, -0.0045595, 0.0148911, -0.0192761, 0.0168764
4: -0.0151353, 0.0115816, -0.0166117, 0.0145109, -0.0296462, 0.0281933
5: 0.0012620, 0.0244782, -0.0001823, 0.0262259, -0.0249640, 0.0246605
6: -0.0368205, 0.0146889, -0.0409389, 0.0178306, -0.0546511, 0.0556278
7: 0.9447536, 0.9808543, 0.9371998, 0.9813001, -0.0365464, 0.0436545
8: -0.0339197, 0.0222682, -0.0363614, 0.0283839, -0.0623036, 0.0586296
9: -0.0194954, 0.0190179, -0.0234168, 0.0227018, -0.0421972, 0.0424348

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274104
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274104
time: 1.44 seconds

## BFS NS instance: NS_A2_B1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0152331, 0.0094570, -0.0145487, 0.0081590, -0.0233921, 0.0240057
1: -0.0114549, 0.0044458, -0.0105340, 0.0038912, -0.0153461, 0.0149798
2: 0.0139171, 0.0658545, 0.0171386, 0.0638845, -0.0499674, 0.0487159
3: -0.0046480, 0.0161968, -0.0045571, 0.0148555, -0.0195035, 0.0207539
4: -0.0173604, 0.0159966, -0.0165912, 0.0144704, -0.0318308, 0.0325879
5: -0.0009148, 0.0271123, -0.0001623, 0.0262017, -0.0271165, 0.0272746
6: -0.0430278, 0.0194240, -0.0408819, 0.0177871, -0.0608149, 0.0603059
7: 0.9333686, 0.9815261, 0.9373045, 0.9812939, -0.0479253, 0.0442216
8: -0.0375998, 0.0314857, -0.0363276, 0.0282992, -0.0658990, 0.0678133
9: -0.0254058, 0.0245702, -0.0233626, 0.0226508, -0.0480566, 0.0479328

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274104
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274104
time: 1.42 seconds

## BFS NS instance: NS_A2_B1_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0143482, 0.0077788, -0.0129184, 0.0050667, -0.0194149, 0.0206972
1: -0.0102643, 0.0037288, -0.0083402, 0.0025701, -0.0128344, 0.0120690
2: 0.0180822, 0.0633075, 0.0248131, 0.0591914, -0.0411092, 0.0384944
3: -0.0045305, 0.0144626, -0.0043405, 0.0116601, -0.0161906, 0.0188031
4: -0.0163659, 0.0140233, -0.0147587, 0.0108344, -0.0272002, 0.0287820
5: 0.0000581, 0.0259350, 0.0016304, 0.0240324, -0.0239742, 0.0243046
6: -0.0402533, 0.0173076, -0.0357698, 0.0138874, -0.0541407, 0.0530774
7: 0.9384574, 0.9812257, 0.9466808, 0.9807407, -0.0422833, 0.0345449
8: -0.0359549, 0.0273658, -0.0332968, 0.0207081, -0.0566630, 0.0606627
9: -0.0227641, 0.0220886, -0.0184950, 0.0180782, -0.0408423, 0.0405836

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A2_A1_B1_B1

### Relational analysis result of NS_A2_B1_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.32 seconds

## Relational analysis of NS_A2_B1_A2_A2_A1_B1_B2

### Relational analysis result of NS_A2_B1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.22 seconds

## BFS NS instance: NS_A2_B1_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0143482, 0.0077788, -0.0134376, 0.0060516, -0.0203998, 0.0212164
1: -0.0102643, 0.0037288, -0.0090389, 0.0029909, -0.0132552, 0.0127677
2: 0.0180822, 0.0633075, 0.0223688, 0.0606862, -0.0426040, 0.0409387
3: -0.0045305, 0.0144626, -0.0044095, 0.0126778, -0.0172083, 0.0188721
4: -0.0163659, 0.0140233, -0.0153423, 0.0119924, -0.0283583, 0.0293656
5: 0.0000581, 0.0259350, 0.0010594, 0.0247233, -0.0246652, 0.0248756
6: -0.0402533, 0.0173076, -0.0373980, 0.0151294, -0.0553828, 0.0547056
7: 0.9384574, 0.9812257, 0.9436943, 0.9809169, -0.0424595, 0.0375314
8: -0.0359549, 0.0273658, -0.0342621, 0.0231258, -0.0590808, 0.0616279
9: -0.0227641, 0.0220886, -0.0200453, 0.0195345, -0.0422986, 0.0421339

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_A2_A1_B2_B1

### Relational analysis result of NS_A2_B1_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.52 seconds

## Relational analysis of NS_A2_B1_A2_A2_A1_B2_B2

### Relational analysis result of NS_A2_B1_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.37 seconds

## BFS NS instance: NS_A2_B1_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0163244, 0.0115270, -0.0129021, 0.0050359, -0.0213603, 0.0244292
1: -0.0129234, 0.0053301, -0.0083184, 0.0025570, -0.0154804, 0.0136485
2: 0.0087798, 0.0689961, 0.0248895, 0.0591447, -0.0503649, 0.0441066
3: -0.0047930, 0.0183357, -0.0043383, 0.0116283, -0.0164214, 0.0226740
4: -0.0185871, 0.0184306, -0.0147405, 0.0107982, -0.0293853, 0.0331710
5: -0.0021148, 0.0285645, 0.0016482, 0.0240108, -0.0261256, 0.0269163
6: -0.0464498, 0.0220345, -0.0357189, 0.0138486, -0.0602984, 0.0577534
7: 0.9270921, 0.9818965, 0.9467739, 0.9807352, -0.0536431, 0.0351225
8: -0.0396286, 0.0365672, -0.0332667, 0.0206325, -0.0602611, 0.0698338
9: -0.0286642, 0.0276311, -0.0184466, 0.0180326, -0.0466968, 0.0460777

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_A2_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_A2_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0163244, 0.0115270, -0.0134208, 0.0060196, -0.0223440, 0.0249478
1: -0.0129234, 0.0053301, -0.0090163, 0.0029772, -0.0159007, 0.0143464
2: 0.0087798, 0.0689961, 0.0224481, 0.0606377, -0.0518579, 0.0465479
3: -0.0047930, 0.0183357, -0.0044072, 0.0126448, -0.0174378, 0.0227429
4: -0.0185871, 0.0184306, -0.0153234, 0.0119548, -0.0305419, 0.0337540
5: -0.0021148, 0.0285645, 0.0010780, 0.0247009, -0.0268157, 0.0274866
6: -0.0464498, 0.0220345, -0.0373451, 0.0150891, -0.0615389, 0.0593796
7: 0.9270921, 0.9818965, 0.9437914, 0.9809111, -0.0538191, 0.0381051
8: -0.0396286, 0.0365672, -0.0342308, 0.0230474, -0.0626760, 0.0707979
9: -0.0286642, 0.0276311, -0.0199950, 0.0194872, -0.0481514, 0.0476261

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.37 seconds

## Relational analysis of NS_A2_B1_A2_A2_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.43 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0128112, 0.0048635, -0.0133033, 0.0057968, -0.0186081, 0.0181668
1: -0.0081961, 0.0024833, -0.0088582, 0.0028821, -0.0110781, 0.0113415
2: 0.0253174, 0.0588830, 0.0230010, 0.0602995, -0.0349821, 0.0358820
3: -0.0043263, 0.0114502, -0.0043916, 0.0124146, -0.0167409, 0.0158418
4: -0.0146383, 0.0105954, -0.0151914, 0.0116929, -0.0263311, 0.0257868
5: 0.0017482, 0.0238898, 0.0012071, 0.0245446, -0.0227964, 0.0226827
6: -0.0354339, 0.0136311, -0.0369768, 0.0148082, -0.0502420, 0.0506079
7: 0.9472969, 0.9807042, 0.9444669, 0.9808712, -0.0335743, 0.0362373
8: -0.0330976, 0.0202093, -0.0340124, 0.0225004, -0.0555981, 0.0542217
9: -0.0181751, 0.0177777, -0.0196443, 0.0191578, -0.0373329, 0.0374219

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.57 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.45 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0127861, 0.0048312, -0.0152825, 0.0095509, -0.0223371, 0.0201137
1: -0.0081742, 0.0024630, -0.0115215, 0.0044859, -0.0126601, 0.0139845
2: 0.0253837, 0.0588108, 0.0136841, 0.0659970, -0.0406133, 0.0451266
3: -0.0043229, 0.0114101, -0.0046546, 0.0162938, -0.0206167, 0.0160648
4: -0.0146100, 0.0105518, -0.0174161, 0.0161070, -0.0307171, 0.0279679
5: 0.0017724, 0.0238564, -0.0009692, 0.0271782, -0.0254058, 0.0248256
6: -0.0353551, 0.0135780, -0.0431829, 0.0195424, -0.0548975, 0.0567609
7: 0.9474165, 0.9806957, 0.9330840, 0.9815429, -0.0341264, 0.0476118
8: -0.0330509, 0.0201158, -0.0376918, 0.0317161, -0.0647671, 0.0578076
9: -0.0181002, 0.0177254, -0.0255536, 0.0247090, -0.0428092, 0.0432790

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_A1_B1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.55 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.37 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0127776, 0.0048254, -0.0144326, 0.0079387, -0.0207163, 0.0192579
1: -0.0081709, 0.0024561, -0.0103778, 0.0037971, -0.0119680, 0.0128338
2: 0.0253886, 0.0587862, 0.0176853, 0.0635502, -0.0381616, 0.0411009
3: -0.0043218, 0.0113997, -0.0045417, 0.0146279, -0.0189496, 0.0159414
4: -0.0146005, 0.0105412, -0.0164607, 0.0142113, -0.0288118, 0.0270019
5: 0.0017795, 0.0238451, -0.0000346, 0.0260472, -0.0242677, 0.0238796
6: -0.0353284, 0.0135622, -0.0405177, 0.0175093, -0.0528377, 0.0540799
7: 0.9474486, 0.9806929, 0.9379725, 0.9812545, -0.0338058, 0.0427204
8: -0.0330351, 0.0200919, -0.0361117, 0.0277584, -0.0607935, 0.0562036
9: -0.0180747, 0.0177138, -0.0230158, 0.0223250, -0.0403997, 0.0407296

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.30 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0147582, 0.0085563, -0.0144070, 0.0078902, -0.0226484, 0.0229633
1: -0.0108159, 0.0040610, -0.0103433, 0.0037764, -0.0145923, 0.0144043
2: 0.0161525, 0.0644875, 0.0178056, 0.0634766, -0.0473241, 0.0466819
3: -0.0045849, 0.0152660, -0.0045383, 0.0145778, -0.0191627, 0.0198043
4: -0.0168267, 0.0149376, -0.0164319, 0.0141543, -0.0309810, 0.0313695
5: -0.0003926, 0.0264805, -0.0000065, 0.0260132, -0.0264058, 0.0264869
6: -0.0415387, 0.0182881, -0.0404375, 0.0174481, -0.0589869, 0.0587257
7: 0.9360997, 0.9813651, 0.9381194, 0.9812458, -0.0451461, 0.0432457
8: -0.0367170, 0.0292746, -0.0360642, 0.0276394, -0.0643564, 0.0653388
9: -0.0239880, 0.0232384, -0.0229395, 0.0222534, -0.0462414, 0.0461779

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0139148, 0.0069566, -0.0128419, 0.0049216, -0.0188364, 0.0197985
1: -0.0096810, 0.0033775, -0.0082373, 0.0025082, -0.0121892, 0.0116149
2: 0.0201227, 0.0620596, 0.0251730, 0.0589713, -0.0388486, 0.0368866
3: -0.0044729, 0.0136130, -0.0043303, 0.0115103, -0.0159831, 0.0179433
4: -0.0158786, 0.0130565, -0.0146727, 0.0106638, -0.0265425, 0.0277293
5: 0.0005348, 0.0253582, 0.0017145, 0.0239306, -0.0233958, 0.0236437
6: -0.0388941, 0.0162707, -0.0355300, 0.0137045, -0.0525986, 0.0518008
7: 0.9409504, 0.9810788, 0.9471205, 0.9807147, -0.0397643, 0.0339583
8: -0.0351491, 0.0253475, -0.0331546, 0.0203520, -0.0555011, 0.0585021
9: -0.0214699, 0.0208728, -0.0182667, 0.0178637, -0.0393336, 0.0391395

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
time: 1.33 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0139148, 0.0069566, -0.0133155, 0.0058200, -0.0197348, 0.0202721
1: -0.0096810, 0.0033775, -0.0088747, 0.0028920, -0.0125730, 0.0122522
2: 0.0201227, 0.0620596, 0.0229435, 0.0603347, -0.0402120, 0.0391161
3: -0.0044729, 0.0136130, -0.0043933, 0.0124385, -0.0169114, 0.0180063
4: -0.0158786, 0.0130565, -0.0152051, 0.0117201, -0.0275987, 0.0282616
5: 0.0005348, 0.0253582, 0.0011937, 0.0245608, -0.0240261, 0.0241645
6: -0.0388941, 0.0162707, -0.0370151, 0.0148374, -0.0537315, 0.0532858
7: 0.9409504, 0.9810788, 0.9443967, 0.9808753, -0.0399249, 0.0366821
8: -0.0351491, 0.0253475, -0.0340351, 0.0225573, -0.0577064, 0.0593826
9: -0.0214699, 0.0208728, -0.0196808, 0.0191921, -0.0406619, 0.0405535

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
time: 1.37 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
time: 2.06 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0138900, 0.0069096, -0.0148214, 0.0086762, -0.0225662, 0.0217310
1: -0.0096477, 0.0033575, -0.0109010, 0.0041122, -0.0137599, 0.0142584
2: 0.0202393, 0.0619884, 0.0158549, 0.0646695, -0.0444302, 0.0461335
3: -0.0044696, 0.0135645, -0.0045933, 0.0153899, -0.0198595, 0.0181578
4: -0.0158508, 0.0130013, -0.0168977, 0.0150786, -0.0309294, 0.0298990
5: 0.0005620, 0.0253253, -0.0004621, 0.0265646, -0.0260026, 0.0257874
6: -0.0388165, 0.0162115, -0.0417370, 0.0184394, -0.0572558, 0.0579484
7: 0.9410927, 0.9810704, 0.9357362, 0.9813864, -0.0402936, 0.0453342
8: -0.0351030, 0.0252322, -0.0368345, 0.0295690, -0.0646720, 0.0620667
9: -0.0213959, 0.0208033, -0.0241768, 0.0234156, -0.0448116, 0.0449801

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0138900, 0.0069096, -0.0152935, 0.0095718, -0.0234618, 0.0222031
1: -0.0096477, 0.0033575, -0.0115363, 0.0044948, -0.0141424, 0.0148938
2: 0.0202393, 0.0619884, 0.0136324, 0.0660286, -0.0457893, 0.0483560
3: -0.0044696, 0.0135645, -0.0046561, 0.0163153, -0.0207849, 0.0182205
4: -0.0158508, 0.0130013, -0.0174284, 0.0161315, -0.0319823, 0.0304297
5: 0.0005620, 0.0253253, -0.0009813, 0.0271928, -0.0266308, 0.0263065
6: -0.0388165, 0.0162115, -0.0432174, 0.0195687, -0.0583852, 0.0594289
7: 0.9410927, 0.9810704, 0.9330209, 0.9815466, -0.0404539, 0.0480495
8: -0.0351030, 0.0252322, -0.0377122, 0.0317672, -0.0668703, 0.0629444
9: -0.0213959, 0.0208033, -0.0255864, 0.0247398, -0.0461358, 0.0463897

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.39 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.24 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0132872, 0.0057663, -0.0133225, 0.0058333, -0.0191205, 0.0190888
1: -0.0088365, 0.0028690, -0.0088841, 0.0028976, -0.0117342, 0.0117531
2: 0.0230768, 0.0602532, 0.0229105, 0.0603549, -0.0372781, 0.0373426
3: -0.0043895, 0.0123830, -0.0043942, 0.0124523, -0.0168418, 0.0167772
4: -0.0151733, 0.0116569, -0.0152130, 0.0117358, -0.0269090, 0.0268699
5: 0.0012248, 0.0245232, 0.0011860, 0.0245702, -0.0233454, 0.0233372
6: -0.0369263, 0.0147696, -0.0370371, 0.0148542, -0.0517805, 0.0518068
7: 0.9445596, 0.9808658, 0.9443563, 0.9808778, -0.0363182, 0.0365095
8: -0.0339825, 0.0224254, -0.0340482, 0.0225900, -0.0565724, 0.0564736
9: -0.0195962, 0.0191127, -0.0197017, 0.0192117, -0.0388079, 0.0388144

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0143816, 0.0078421, -0.0133225, 0.0058333, -0.0202149, 0.0211646
1: -0.0103092, 0.0037558, -0.0088841, 0.0028976, -0.0132068, 0.0126399
2: 0.0179252, 0.0634035, 0.0229105, 0.0603549, -0.0424297, 0.0404930
3: -0.0045349, 0.0145280, -0.0043942, 0.0124523, -0.0169872, 0.0189222
4: -0.0164034, 0.0140977, -0.0152130, 0.0117358, -0.0281391, 0.0293107
5: 0.0000215, 0.0259794, 0.0011860, 0.0245702, -0.0245487, 0.0247934
6: -0.0403579, 0.0173874, -0.0370371, 0.0148542, -0.0552121, 0.0544245
7: 0.9382654, 0.9812371, 0.9443563, 0.9808778, -0.0426124, 0.0368809
8: -0.0360169, 0.0275211, -0.0340482, 0.0225900, -0.0586069, 0.0615693
9: -0.0228637, 0.0221821, -0.0197017, 0.0192117, -0.0420754, 0.0418838

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
time: 1.40 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0132623, 0.0057190, -0.0153028, 0.0095894, -0.0228517, 0.0210219
1: -0.0088030, 0.0028488, -0.0115488, 0.0045023, -0.0133053, 0.0143976
2: 0.0231941, 0.0601815, 0.0135886, 0.0660554, -0.0428613, 0.0465929
3: -0.0043862, 0.0123342, -0.0046573, 0.0163335, -0.0207197, 0.0169915
4: -0.0151453, 0.0116014, -0.0174389, 0.0161523, -0.0312976, 0.0290403
5: 0.0012522, 0.0244900, -0.0009915, 0.0272052, -0.0259530, 0.0254815
6: -0.0368482, 0.0147101, -0.0432466, 0.0195910, -0.0564392, 0.0579566
7: 0.9447028, 0.9808573, 0.9329674, 0.9815499, -0.0368471, 0.0478899
8: -0.0339362, 0.0223095, -0.0377295, 0.0318106, -0.0657468, 0.0600390
9: -0.0195218, 0.0190428, -0.0256142, 0.0247659, -0.0442878, 0.0446569

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
time: 2.24 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0143569, 0.0077952, -0.0153028, 0.0095894, -0.0239463, 0.0230981
1: -0.0102759, 0.0037358, -0.0115488, 0.0045023, -0.0147783, 0.0152846
2: 0.0180414, 0.0633324, 0.0135886, 0.0660554, -0.0480140, 0.0497437
3: -0.0045316, 0.0144796, -0.0046573, 0.0163335, -0.0208651, 0.0191369
4: -0.0163756, 0.0140426, -0.0174389, 0.0161523, -0.0325279, 0.0314815
5: 0.0000486, 0.0259465, -0.0009915, 0.0272052, -0.0271566, 0.0269380
6: -0.0402805, 0.0173283, -0.0432466, 0.0195910, -0.0598715, 0.0605749
7: 0.9384076, 0.9812287, 0.9329674, 0.9815499, -0.0431423, 0.0482613
8: -0.0359710, 0.0274061, -0.0377295, 0.0318106, -0.0677816, 0.0651357
9: -0.0227899, 0.0221129, -0.0256142, 0.0247659, -0.0475558, 0.0477270

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
time: 1.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
time: 1.32 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0133155, 0.0058200, -0.0139148, 0.0069566, -0.0202721, 0.0197348
1: -0.0088747, 0.0028920, -0.0096810, 0.0033775, -0.0122522, 0.0125730
2: 0.0229435, 0.0603347, 0.0201227, 0.0620596, -0.0391161, 0.0402120
3: -0.0043933, 0.0124385, -0.0044729, 0.0136130, -0.0180063, 0.0169114
4: -0.0152051, 0.0117201, -0.0158786, 0.0130565, -0.0282616, 0.0275987
5: 0.0011937, 0.0245608, 0.0005348, 0.0253582, -0.0241645, 0.0240261
6: -0.0370151, 0.0148374, -0.0388941, 0.0162707, -0.0532858, 0.0537315
7: 0.9443967, 0.9808753, 0.9409504, 0.9810788, -0.0366821, 0.0399249
8: -0.0340351, 0.0225573, -0.0351491, 0.0253475, -0.0593826, 0.0577064
9: -0.0196808, 0.0191921, -0.0214699, 0.0208728, -0.0405535, 0.0406619

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0133155, 0.0058200, -0.0143816, 0.0078421, -0.0211576, 0.0202016
1: -0.0088747, 0.0028920, -0.0103092, 0.0037558, -0.0126305, 0.0132011
2: 0.0229435, 0.0603347, 0.0179252, 0.0634035, -0.0404600, 0.0424095
3: -0.0043933, 0.0124385, -0.0045349, 0.0145280, -0.0189212, 0.0169734
4: -0.0152051, 0.0117201, -0.0164034, 0.0140977, -0.0293028, 0.0281235
5: 0.0011937, 0.0245608, 0.0000215, 0.0259794, -0.0247857, 0.0245394
6: -0.0370151, 0.0148374, -0.0403579, 0.0173874, -0.0544025, 0.0551953
7: 0.9443967, 0.9808753, 0.9382654, 0.9812371, -0.0368404, 0.0426099
8: -0.0340351, 0.0225573, -0.0360169, 0.0275211, -0.0615563, 0.0585742
9: -0.0196808, 0.0191921, -0.0228637, 0.0221821, -0.0418629, 0.0420557

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0152935, 0.0095718, -0.0138900, 0.0069096, -0.0222031, 0.0234618
1: -0.0115363, 0.0044948, -0.0096477, 0.0033575, -0.0148938, 0.0141424
2: 0.0136324, 0.0660286, 0.0202393, 0.0619884, -0.0483560, 0.0457893
3: -0.0046561, 0.0163153, -0.0044696, 0.0135645, -0.0182205, 0.0207849
4: -0.0174284, 0.0161315, -0.0158508, 0.0130013, -0.0304297, 0.0319823
5: -0.0009813, 0.0271928, 0.0005620, 0.0253253, -0.0263065, 0.0266308
6: -0.0432174, 0.0195687, -0.0388165, 0.0162115, -0.0594289, 0.0583852
7: 0.9330209, 0.9815466, 0.9410927, 0.9810704, -0.0480495, 0.0404539
8: -0.0377122, 0.0317672, -0.0351030, 0.0252322, -0.0629444, 0.0668703
9: -0.0255864, 0.0247398, -0.0213959, 0.0208033, -0.0463897, 0.0461358

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0152935, 0.0095718, -0.0143569, 0.0077952, -0.0230887, 0.0239286
1: -0.0115363, 0.0044948, -0.0102759, 0.0037358, -0.0152721, 0.0147707
2: 0.0136324, 0.0660286, 0.0180414, 0.0633324, -0.0497000, 0.0479872
3: -0.0046561, 0.0163153, -0.0045316, 0.0144796, -0.0191356, 0.0208469
4: -0.0174284, 0.0161315, -0.0163756, 0.0140426, -0.0314710, 0.0325072
5: -0.0009813, 0.0271928, 0.0000486, 0.0259465, -0.0269278, 0.0271442
6: -0.0432174, 0.0195687, -0.0402805, 0.0173283, -0.0605457, 0.0598492
7: 0.9330209, 0.9815466, 0.9384076, 0.9812287, -0.0482078, 0.0431390
8: -0.0377122, 0.0317672, -0.0359710, 0.0274061, -0.0651184, 0.0677383
9: -0.0255864, 0.0247398, -0.0227899, 0.0221129, -0.0476992, 0.0475297

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.12 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.25 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A1_B1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A1_B1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A1_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
NS_A1_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
NS_A1_B1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0281711
NS_A1_B1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0281711
NS_A1_B1_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A1_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A1_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A1_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A1_B1_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0295255
NS_A1_B1_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0295255
NS_A1_B1_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
NS_A1_B1_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
NS_A1_B1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
NS_A1_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
NS_A1_B1_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
NS_A1_B1_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
NS_A1_B2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
NS_A1_B2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
NS_A1_B2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290693
NS_A1_B2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
NS_A1_B2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
NS_A1_B2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
NS_A1_B2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282347
NS_A1_B2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
NS_A1_B2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
NS_A1_B2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
NS_A1_B2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
NS_A1_B2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0294413
NS_A1_B2_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0295255
NS_A1_B2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
NS_A1_B2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
NS_A1_B2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289034
NS_A1_B2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0289763
NS_A1_B2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281711
NS_A1_B2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0281808
NS_A1_B2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285016
NS_A1_B2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285016
NS_A1_B2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
NS_A1_B2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
NS_A1_B2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
NS_A2_B1_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
NS_A2_B1_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
NS_A2_B1_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
NS_A2_B1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B1_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B1_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
NS_A2_B1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
NS_A2_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
NS_A2_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
NS_A2_B1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274104
NS_A2_B1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274104
NS_A2_B1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274104
NS_A2_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274104
NS_A2_B1_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A2_B2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A2_B2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A2_B2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A2_B2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A2_B2_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
NS_A2_B2_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
NS_A2_B2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
NS_A2_B2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0285328
NS_A2_B2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
NS_A2_B2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
NS_A2_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
NS_A2_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0290218
NS_A2_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
NS_A2_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
NS_A2_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
NS_A2_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
NS_A2_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.25
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0128634, 0.0049623, -0.0134271, 0.0060316, -0.0188950, 0.0183894
1: -0.0082662, 0.0025256, -0.0090248, 0.0029824, -0.0112486, 0.0115504
2: 0.0250721, 0.0590331, 0.0224183, 0.0606559, -0.0355838, 0.0366148
3: -0.0043332, 0.0115523, -0.0044081, 0.0126572, -0.0169904, 0.0159604
4: -0.0146968, 0.0107116, -0.0153305, 0.0119690, -0.0266658, 0.0260422
5: 0.0016909, 0.0239592, 0.0010710, 0.0247093, -0.0230184, 0.0228882
6: -0.0355973, 0.0137558, -0.0373650, 0.0151043, -0.0507016, 0.0511208
7: 0.9469972, 0.9807220, 0.9437549, 0.9809133, -0.0339162, 0.0369671
8: -0.0331945, 0.0204519, -0.0342426, 0.0230769, -0.0562715, 0.0546945
9: -0.0183307, 0.0179238, -0.0200139, 0.0195050, -0.0378358, 0.0379378

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0294413
time: 1.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0294413
time: 1.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0148500, 0.0087305, -0.0134271, 0.0060316, -0.0208816, 0.0221576
1: -0.0109395, 0.0041354, -0.0090248, 0.0029824, -0.0139218, 0.0131602
2: 0.0157203, 0.0647518, 0.0224183, 0.0606559, -0.0449356, 0.0423336
3: -0.0045971, 0.0154460, -0.0044081, 0.0126572, -0.0172544, 0.0198541
4: -0.0169299, 0.0151423, -0.0153305, 0.0119690, -0.0288988, 0.0304729
5: -0.0004936, 0.0266026, 0.0010710, 0.0247093, -0.0252029, 0.0255317
6: -0.0418266, 0.0185078, -0.0373650, 0.0151043, -0.0569309, 0.0558728
7: 0.9355717, 0.9813961, 0.9437549, 0.9809133, -0.0453417, 0.0376413
8: -0.0368877, 0.0297021, -0.0342426, 0.0230769, -0.0599646, 0.0639446
9: -0.0242621, 0.0234958, -0.0200139, 0.0195050, -0.0437672, 0.0435098

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0294413
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0294413
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0128688, 0.0049727, -0.0148500, 0.0087305, -0.0215993, 0.0198227
1: -0.0082735, 0.0025300, -0.0109395, 0.0041354, -0.0124089, 0.0134694
2: 0.0250464, 0.0590487, 0.0157203, 0.0647518, -0.0397054, 0.0433285
3: -0.0043339, 0.0115630, -0.0045971, 0.0154460, -0.0197799, 0.0161601
4: -0.0147030, 0.0107238, -0.0169299, 0.0151423, -0.0298453, 0.0276537
5: 0.0016849, 0.0239664, -0.0004936, 0.0266026, -0.0249177, 0.0244600
6: -0.0356143, 0.0137688, -0.0418266, 0.0185078, -0.0541221, 0.0555954
7: 0.9469659, 0.9807239, 0.9355717, 0.9813961, -0.0344302, 0.0451522
8: -0.0332047, 0.0204773, -0.0368877, 0.0297021, -0.0629067, 0.0573650
9: -0.0183470, 0.0179391, -0.0242621, 0.0234958, -0.0418428, 0.0422013

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273865, upper bound: 0.0279192
time: 1.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273865, upper bound: 0.0273688
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0128688, 0.0049727, -0.0153304, 0.0096417, -0.0225105, 0.0203031
1: -0.0082735, 0.0025300, -0.0115859, 0.0045246, -0.0127982, 0.0141159
2: 0.0250464, 0.0590487, 0.0134588, 0.0661348, -0.0410883, 0.0455900
3: -0.0043339, 0.0115630, -0.0046610, 0.0163876, -0.0207215, 0.0162240
4: -0.0147030, 0.0107238, -0.0174699, 0.0162138, -0.0309167, 0.0281937
5: 0.0016849, 0.0239664, -0.0010218, 0.0272419, -0.0255570, 0.0249883
6: -0.0356143, 0.0137688, -0.0433330, 0.0196569, -0.0552713, 0.0571019
7: 0.9469659, 0.9807239, 0.9328088, 0.9815592, -0.0345933, 0.0479151
8: -0.0332047, 0.0204773, -0.0377808, 0.0319390, -0.0651437, 0.0582581
9: -0.0183470, 0.0179391, -0.0256965, 0.0248433, -0.0431903, 0.0436356

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273865, upper bound: 0.0279192
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273865, upper bound: 0.0273688
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0128634, 0.0049623, -0.0189884, 0.0200311
1: -0.0098308, 0.0034677, -0.0082662, 0.0025256, -0.0123564, 0.0117339
2: 0.0195986, 0.0623801, 0.0250721, 0.0590331, -0.0394344, 0.0373080
3: -0.0044877, 0.0138312, -0.0043332, 0.0115523, -0.0160400, 0.0181644
4: -0.0160038, 0.0133048, -0.0146968, 0.0107116, -0.0267154, 0.0280017
5: 0.0004124, 0.0255063, 0.0016909, 0.0239592, -0.0235468, 0.0238154
6: -0.0392432, 0.0165370, -0.0355973, 0.0137558, -0.0529990, 0.0521343
7: 0.9403101, 0.9811166, 0.9469972, 0.9807220, -0.0404118, 0.0341194
8: -0.0353561, 0.0258659, -0.0331945, 0.0204519, -0.0558080, 0.0590604
9: -0.0218022, 0.0211850, -0.0183307, 0.0179238, -0.0397261, 0.0395157

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
time: 1.41 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0133834, 0.0059487, -0.0199748, 0.0205511
1: -0.0098308, 0.0034677, -0.0089659, 0.0029469, -0.0127777, 0.0124337
2: 0.0195986, 0.0623801, 0.0226242, 0.0605300, -0.0409314, 0.0397559
3: -0.0044877, 0.0138312, -0.0044023, 0.0125715, -0.0170592, 0.0182335
4: -0.0160038, 0.0133048, -0.0152814, 0.0118714, -0.0278752, 0.0285862
5: 0.0004124, 0.0255063, 0.0011191, 0.0246511, -0.0242387, 0.0243872
6: -0.0392432, 0.0165370, -0.0372279, 0.0149997, -0.0542428, 0.0537649
7: 0.9403101, 0.9811166, 0.9440065, 0.9808984, -0.0405883, 0.0371101
8: -0.0353561, 0.0258659, -0.0341612, 0.0228733, -0.0582293, 0.0600271
9: -0.0218022, 0.0211850, -0.0198833, 0.0193824, -0.0411846, 0.0410683

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289034
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0140097, 0.0071367, -0.0148500, 0.0087305, -0.0227402, 0.0219867
1: -0.0098088, 0.0034545, -0.0109395, 0.0041354, -0.0139442, 0.0143940
2: 0.0196757, 0.0623331, 0.0157203, 0.0647518, -0.0450762, 0.0466128
3: -0.0044855, 0.0137991, -0.0045971, 0.0154460, -0.0199315, 0.0183963
4: -0.0159854, 0.0132684, -0.0169299, 0.0151423, -0.0311277, 0.0301982
5: 0.0004303, 0.0254846, -0.0004936, 0.0266026, -0.0261723, 0.0259782
6: -0.0391919, 0.0164979, -0.0418266, 0.0185078, -0.0576997, 0.0583245
7: 0.9404041, 0.9811110, 0.9355717, 0.9813961, -0.0409921, 0.0455393
8: -0.0353257, 0.0257897, -0.0368877, 0.0297021, -0.0650277, 0.0626774
9: -0.0217535, 0.0211392, -0.0242621, 0.0234958, -0.0452493, 0.0454013

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0274582
time: 2.53 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0271855
time: 1.45 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0140097, 0.0071367, -0.0153304, 0.0096417, -0.0236515, 0.0224671
1: -0.0098088, 0.0034545, -0.0115859, 0.0045246, -0.0143334, 0.0150404
2: 0.0196757, 0.0623331, 0.0134588, 0.0661348, -0.0464591, 0.0488743
3: -0.0044855, 0.0137991, -0.0046610, 0.0163876, -0.0208731, 0.0184601
4: -0.0159854, 0.0132684, -0.0174699, 0.0162138, -0.0321992, 0.0307382
5: 0.0004303, 0.0254846, -0.0010218, 0.0272419, -0.0268116, 0.0265064
6: -0.0391919, 0.0164979, -0.0433330, 0.0196569, -0.0588489, 0.0598310
7: 0.9404041, 0.9811110, 0.9328088, 0.9815592, -0.0411552, 0.0483022
8: -0.0353257, 0.0257897, -0.0377808, 0.0319390, -0.0672647, 0.0635705
9: -0.0217535, 0.0211392, -0.0256965, 0.0248433, -0.0465968, 0.0468357

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: A, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0274582
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0271855
time: 1.24 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.55 + 597.54 = 601.09 seconds
