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
execution time: IAR + RelationalAnalysis = 1.49 + 2.16 = 3.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
time: 1.24 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
time: 1.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.71 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.71
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.71
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

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
time: 1.48 seconds

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

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
time: 1.24 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
time: 1.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.73 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0304973
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.73
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

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0303291, upper bound: 0.0294948
time: 2.09 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0294948
time: 1.25 seconds

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

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0303291, upper bound: 0.0294948
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0294948
time: 1.13 seconds

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

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0302991
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
time: 1.33 seconds

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

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0302991
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
time: 1.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.38 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 7, lower bound: -0.0303291, upper bound: 0.0294948
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0294948
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 7, lower bound: -0.0303291, upper bound: 0.0294948
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0294948
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0302991
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 7, lower bound: -0.0304973, upper bound: 0.0302991
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 7, lower bound: -0.0302991, upper bound: 0.0302991

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0134855, 0.0061423, -0.0135281, 0.0062231, -0.0197086, 0.0196703
1: -0.0091033, 0.0030297, -0.0091606, 0.0030642, -0.0121675, 0.0121903
2: 0.0221437, 0.0608238, 0.0219431, 0.0609465, -0.0388028, 0.0388808
3: -0.0044158, 0.0127716, -0.0044215, 0.0128551, -0.0172709, 0.0171931
4: -0.0153961, 0.0120991, -0.0154440, 0.0121941, -0.0275902, 0.0275431
5: 0.0010068, 0.0247869, 0.0009600, 0.0248436, -0.0238368, 0.0238269
6: -0.0375479, 0.0152438, -0.0376816, 0.0153457, -0.0528937, 0.0529254
7: 0.9434195, 0.9809331, 0.9431743, 0.9809474, -0.0375280, 0.0377588
8: -0.0343510, 0.0233485, -0.0344302, 0.0235469, -0.0578979, 0.0577787
9: -0.0201881, 0.0196686, -0.0203153, 0.0197882, -0.0399763, 0.0399839

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0294948
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301192, upper bound: 0.0293418
time: 1.80 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0145845, 0.0082270, -0.0135165, 0.0062011, -0.0207856, 0.0217435
1: -0.0105822, 0.0039202, -0.0091450, 0.0030548, -0.0136370, 0.0130653
2: 0.0169699, 0.0639877, 0.0219977, 0.0609131, -0.0439432, 0.0419900
3: -0.0045619, 0.0149257, -0.0044200, 0.0128324, -0.0173942, 0.0193457
4: -0.0166315, 0.0145503, -0.0154309, 0.0121682, -0.0287997, 0.0299812
5: -0.0002017, 0.0262494, 0.0009727, 0.0248282, -0.0250299, 0.0252767
6: -0.0409942, 0.0178728, -0.0376452, 0.0153180, -0.0563122, 0.0555180
7: 0.9370984, 0.9813061, 0.9432410, 0.9809435, -0.0438451, 0.0380651
8: -0.0363942, 0.0284660, -0.0344086, 0.0234929, -0.0598871, 0.0628747
9: -0.0234695, 0.0227513, -0.0202807, 0.0197556, -0.0432252, 0.0430320

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0294948
time: 1.50 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.50 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0134855, 0.0061423, -0.0134634, 0.0061005, -0.0195860, 0.0196057
1: -0.0091033, 0.0030297, -0.0090737, 0.0030118, -0.0121151, 0.0121033
2: 0.0221437, 0.0608238, 0.0222472, 0.0607605, -0.0386168, 0.0385766
3: -0.0044158, 0.0127716, -0.0044129, 0.0127284, -0.0171443, 0.0171845
4: -0.0153961, 0.0120991, -0.0153714, 0.0120500, -0.0274461, 0.0274704
5: 0.0010068, 0.0247869, 0.0010310, 0.0247577, -0.0237508, 0.0237559
6: -0.0375479, 0.0152438, -0.0374789, 0.0151912, -0.0527391, 0.0527228
7: 0.9434195, 0.9809331, 0.9435460, 0.9809256, -0.0375061, 0.0373871
8: -0.0343510, 0.0233485, -0.0343101, 0.0232460, -0.0575971, 0.0576586
9: -0.0201881, 0.0196686, -0.0201224, 0.0196069, -0.0397951, 0.0397910

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0294948
time: 1.71 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301667, upper bound: 0.0293418
time: 1.80 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0145845, 0.0082270, -0.0134383, 0.0060529, -0.0206374, 0.0216653
1: -0.0105822, 0.0039202, -0.0090399, 0.0029915, -0.0135737, 0.0129601
2: 0.0169699, 0.0639877, 0.0223655, 0.0606881, -0.0437182, 0.0416221
3: -0.0045619, 0.0149257, -0.0044096, 0.0126792, -0.0172411, 0.0193353
4: -0.0166315, 0.0145503, -0.0153431, 0.0119939, -0.0286254, 0.0298934
5: -0.0002017, 0.0262494, 0.0010587, 0.0247242, -0.0249259, 0.0251907
6: -0.0409942, 0.0178728, -0.0374001, 0.0151311, -0.0561253, 0.0552729
7: 0.9370984, 0.9813061, 0.9436905, 0.9809170, -0.0438186, 0.0376156
8: -0.0363942, 0.0284660, -0.0342634, 0.0231291, -0.0595232, 0.0627294
9: -0.0234695, 0.0227513, -0.0200474, 0.0195364, -0.0430060, 0.0427986

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0294948
time: 1.38 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.59 seconds

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

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
time: 1.51 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
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

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
time: 1.43 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.33 seconds

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

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
time: 1.30 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 3.70 seconds

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

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
time: 1.30 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.06 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0294948
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0301192, upper bound: 0.0293418
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0294948
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0294948
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0301667, upper bound: 0.0293418
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0294948
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.06
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0134507, 0.0060765, -0.0129301, 0.0050889, -0.0185397, 0.0190066
1: -0.0090566, 0.0030015, -0.0083560, 0.0025796, -0.0116362, 0.0113575
2: 0.0223071, 0.0607240, 0.0247579, 0.0592252, -0.0369181, 0.0359660
3: -0.0044112, 0.0127035, -0.0043421, 0.0116831, -0.0160943, 0.0170456
4: -0.0153571, 0.0120217, -0.0147719, 0.0108605, -0.0262176, 0.0267935
5: 0.0010450, 0.0247408, 0.0016175, 0.0240480, -0.0230030, 0.0231233
6: -0.0374392, 0.0151608, -0.0358065, 0.0139154, -0.0513546, 0.0509674
7: 0.9436190, 0.9809213, 0.9466134, 0.9807446, -0.0371256, 0.0343078
8: -0.0342865, 0.0231869, -0.0333186, 0.0207626, -0.0550491, 0.0565055
9: -0.0200845, 0.0195713, -0.0185300, 0.0181110, -0.0381955, 0.0381013

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
time: 3.01 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0134682, 0.0061095, -0.0134493, 0.0060737, -0.0195419, 0.0195588
1: -0.0090800, 0.0030156, -0.0090546, 0.0030003, -0.0120804, 0.0120703
2: 0.0222251, 0.0607741, 0.0223139, 0.0607197, -0.0384947, 0.0384602
3: -0.0044135, 0.0127377, -0.0044110, 0.0127007, -0.0171142, 0.0171487
4: -0.0153767, 0.0120605, -0.0153554, 0.0120184, -0.0273951, 0.0274159
5: 0.0010259, 0.0247639, 0.0010466, 0.0247388, -0.0237130, 0.0237173
6: -0.0374937, 0.0152025, -0.0374345, 0.0151573, -0.0526511, 0.0526370
7: 0.9435188, 0.9809272, 0.9436274, 0.9809207, -0.0374019, 0.0372999
8: -0.0343189, 0.0232680, -0.0342838, 0.0231801, -0.0574990, 0.0575518
9: -0.0201365, 0.0196202, -0.0200801, 0.0195672, -0.0397037, 0.0397003

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301192, upper bound: 0.0293418
time: 1.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301192, upper bound: 0.0293418
time: 1.39 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0145496, 0.0081607, -0.0129184, 0.0050667, -0.0196163, 0.0210790
1: -0.0105352, 0.0038919, -0.0083402, 0.0025701, -0.0131053, 0.0122322
2: 0.0171345, 0.0638870, 0.0248131, 0.0591914, -0.0420570, 0.0390740
3: -0.0045572, 0.0148572, -0.0043405, 0.0116601, -0.0162174, 0.0191977
4: -0.0165922, 0.0144723, -0.0147587, 0.0108344, -0.0274266, 0.0292310
5: -0.0001632, 0.0262029, 0.0016304, 0.0240324, -0.0241956, 0.0245725
6: -0.0408846, 0.0177892, -0.0357698, 0.0138874, -0.0547720, 0.0535590
7: 0.9372995, 0.9812942, 0.9466808, 0.9807407, -0.0434412, 0.0346134
8: -0.0363292, 0.0283033, -0.0332968, 0.0207081, -0.0570373, 0.0616001
9: -0.0233652, 0.0226532, -0.0184950, 0.0180782, -0.0414433, 0.0411482

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.41 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0145669, 0.0081935, -0.0134376, 0.0060516, -0.0206185, 0.0216311
1: -0.0105585, 0.0039060, -0.0090389, 0.0029909, -0.0135494, 0.0129449
2: 0.0170529, 0.0639369, 0.0223688, 0.0606862, -0.0436332, 0.0415681
3: -0.0045595, 0.0148911, -0.0044095, 0.0126778, -0.0172374, 0.0193006
4: -0.0166117, 0.0145109, -0.0153423, 0.0119924, -0.0286041, 0.0298532
5: -0.0001823, 0.0262259, 0.0010594, 0.0247233, -0.0249056, 0.0251665
6: -0.0409389, 0.0178306, -0.0373980, 0.0151294, -0.0560683, 0.0552286
7: 0.9371998, 0.9813001, 0.9436943, 0.9809169, -0.0437170, 0.0376058
8: -0.0363614, 0.0283839, -0.0342621, 0.0231258, -0.0594872, 0.0626460
9: -0.0234168, 0.0227018, -0.0200453, 0.0195345, -0.0429514, 0.0427472

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.35 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0134507, 0.0060765, -0.0129003, 0.0050324, -0.0184831, 0.0189768
1: -0.0090566, 0.0030015, -0.0083159, 0.0025555, -0.0116121, 0.0113174
2: 0.0223071, 0.0607240, 0.0248983, 0.0591394, -0.0368323, 0.0358257
3: -0.0044112, 0.0127035, -0.0043381, 0.0116247, -0.0160359, 0.0170416
4: -0.0153571, 0.0120217, -0.0147383, 0.0107940, -0.0261511, 0.0267600
5: 0.0010450, 0.0247408, 0.0016503, 0.0240083, -0.0229633, 0.0230905
6: -0.0374392, 0.0151608, -0.0357131, 0.0138441, -0.0512833, 0.0508739
7: 0.9436190, 0.9809213, 0.9467847, 0.9807345, -0.0371155, 0.0341365
8: -0.0342865, 0.0231869, -0.0332632, 0.0206238, -0.0549103, 0.0564501
9: -0.0200845, 0.0195713, -0.0184410, 0.0180274, -0.0381119, 0.0380123

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0134682, 0.0061095, -0.0133744, 0.0059317, -0.0193999, 0.0194839
1: -0.0090800, 0.0030156, -0.0089539, 0.0029397, -0.0120197, 0.0119696
2: 0.0222251, 0.0607741, 0.0226662, 0.0605043, -0.0382792, 0.0381078
3: -0.0044135, 0.0127377, -0.0044011, 0.0125540, -0.0169675, 0.0171388
4: -0.0153767, 0.0120605, -0.0152713, 0.0118515, -0.0272282, 0.0273318
5: 0.0010259, 0.0247639, 0.0011289, 0.0246392, -0.0236134, 0.0236350
6: -0.0374937, 0.0152025, -0.0371998, 0.0149783, -0.0524721, 0.0524023
7: 0.9435188, 0.9809272, 0.9440578, 0.9808955, -0.0373766, 0.0368694
8: -0.0343189, 0.0232680, -0.0341446, 0.0228316, -0.0571505, 0.0574126
9: -0.0201365, 0.0196202, -0.0198567, 0.0193573, -0.0394938, 0.0394768

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301667, upper bound: 0.0293418
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301667, upper bound: 0.0293418
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0145496, 0.0081607, -0.0128753, 0.0049850, -0.0195346, 0.0210360
1: -0.0105352, 0.0038919, -0.0082823, 0.0025353, -0.0130705, 0.0121743
2: 0.0171345, 0.0638870, 0.0250157, 0.0590676, -0.0419331, 0.0388714
3: -0.0045572, 0.0148572, -0.0043348, 0.0115758, -0.0161330, 0.0191920
4: -0.0165922, 0.0144723, -0.0147103, 0.0107384, -0.0273306, 0.0291826
5: -0.0001632, 0.0262029, 0.0016777, 0.0239751, -0.0241384, 0.0245252
6: -0.0408846, 0.0177892, -0.0356349, 0.0137845, -0.0546691, 0.0534240
7: 0.9372995, 0.9812942, 0.9469282, 0.9807261, -0.0434266, 0.0343660
8: -0.0363292, 0.0283033, -0.0332168, 0.0205077, -0.0568369, 0.0615201
9: -0.0233652, 0.0226532, -0.0183665, 0.0179574, -0.0413226, 0.0410197

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0145669, 0.0081935, -0.0133492, 0.0058838, -0.0204507, 0.0215427
1: -0.0105585, 0.0039060, -0.0089199, 0.0029192, -0.0134777, 0.0128259
2: 0.0170529, 0.0639369, 0.0227852, 0.0604315, -0.0433786, 0.0411517
3: -0.0045595, 0.0148911, -0.0043977, 0.0125045, -0.0170640, 0.0192889
4: -0.0166117, 0.0145109, -0.0152429, 0.0117951, -0.0284068, 0.0297538
5: -0.0001823, 0.0262259, 0.0011567, 0.0246056, -0.0247879, 0.0250692
6: -0.0409389, 0.0178306, -0.0371206, 0.0149178, -0.0558567, 0.0549512
7: 0.9371998, 0.9813001, 0.9442031, 0.9808869, -0.0436870, 0.0370969
8: -0.0363614, 0.0283839, -0.0340976, 0.0227140, -0.0590753, 0.0624815
9: -0.0234168, 0.0227018, -0.0197812, 0.0192864, -0.0427033, 0.0424830

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.62 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0129003, 0.0050324, -0.0134507, 0.0060765, -0.0189768, 0.0184831
1: -0.0083159, 0.0025555, -0.0090566, 0.0030015, -0.0113174, 0.0116121
2: 0.0248983, 0.0591394, 0.0223071, 0.0607240, -0.0358257, 0.0368323
3: -0.0043381, 0.0116247, -0.0044112, 0.0127035, -0.0170416, 0.0160359
4: -0.0147383, 0.0107940, -0.0153571, 0.0120217, -0.0267600, 0.0261511
5: 0.0016503, 0.0240083, 0.0010450, 0.0247408, -0.0230905, 0.0229633
6: -0.0357131, 0.0138441, -0.0374392, 0.0151608, -0.0508739, 0.0512833
7: 0.9467847, 0.9807345, 0.9436190, 0.9809213, -0.0341365, 0.0371155
8: -0.0332632, 0.0206238, -0.0342865, 0.0231869, -0.0564501, 0.0549103
9: -0.0184410, 0.0180274, -0.0200845, 0.0195713, -0.0380123, 0.0381119

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 1.52 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 1.49 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128753, 0.0049850, -0.0145496, 0.0081607, -0.0210360, 0.0195346
1: -0.0082823, 0.0025353, -0.0105352, 0.0038919, -0.0121743, 0.0130705
2: 0.0250157, 0.0590676, 0.0171345, 0.0638870, -0.0388714, 0.0419331
3: -0.0043348, 0.0115758, -0.0045572, 0.0148572, -0.0191920, 0.0161330
4: -0.0147103, 0.0107384, -0.0165922, 0.0144723, -0.0291826, 0.0273306
5: 0.0016777, 0.0239751, -0.0001632, 0.0262029, -0.0245252, 0.0241384
6: -0.0356349, 0.0137845, -0.0408846, 0.0177892, -0.0534240, 0.0546691
7: 0.9469282, 0.9807261, 0.9372995, 0.9812942, -0.0343660, 0.0434266
8: -0.0332168, 0.0205077, -0.0363292, 0.0283033, -0.0615201, 0.0568369
9: -0.0183665, 0.0179574, -0.0233652, 0.0226532, -0.0410197, 0.0413226

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0133744, 0.0059317, -0.0134682, 0.0061095, -0.0194839, 0.0193999
1: -0.0089539, 0.0029397, -0.0090800, 0.0030156, -0.0119696, 0.0120197
2: 0.0226662, 0.0605043, 0.0222251, 0.0607741, -0.0381078, 0.0382792
3: -0.0044011, 0.0125540, -0.0044135, 0.0127377, -0.0171388, 0.0169675
4: -0.0152713, 0.0118515, -0.0153767, 0.0120605, -0.0273318, 0.0272282
5: 0.0011289, 0.0246392, 0.0010259, 0.0247639, -0.0236350, 0.0236134
6: -0.0371998, 0.0149783, -0.0374937, 0.0152025, -0.0524023, 0.0524721
7: 0.9440578, 0.9808955, 0.9435188, 0.9809272, -0.0368694, 0.0373766
8: -0.0341446, 0.0228316, -0.0343189, 0.0232680, -0.0574126, 0.0571505
9: -0.0198567, 0.0193573, -0.0201365, 0.0196202, -0.0394768, 0.0394938

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.68 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.50 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0133492, 0.0058838, -0.0145669, 0.0081935, -0.0215427, 0.0204507
1: -0.0089199, 0.0029192, -0.0105585, 0.0039060, -0.0128259, 0.0134777
2: 0.0227852, 0.0604315, 0.0170529, 0.0639369, -0.0411517, 0.0433786
3: -0.0043977, 0.0125045, -0.0045595, 0.0148911, -0.0192889, 0.0170640
4: -0.0152429, 0.0117951, -0.0166117, 0.0145109, -0.0297538, 0.0284068
5: 0.0011567, 0.0246056, -0.0001823, 0.0262259, -0.0250692, 0.0247879
6: -0.0371206, 0.0149178, -0.0409389, 0.0178306, -0.0549512, 0.0558567
7: 0.9442031, 0.9808869, 0.9371998, 0.9813001, -0.0370969, 0.0436870
8: -0.0340976, 0.0227140, -0.0363614, 0.0283839, -0.0624815, 0.0590753
9: -0.0197812, 0.0192864, -0.0234168, 0.0227018, -0.0424830, 0.0427033

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285424, upper bound: 0.0274104
time: 1.27 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0129003, 0.0050324, -0.0133380, 0.0058626, -0.0187629, 0.0183704
1: -0.0083159, 0.0025555, -0.0089049, 0.0029102, -0.0112261, 0.0114604
2: 0.0248983, 0.0591394, 0.0228377, 0.0603994, -0.0355011, 0.0363016
3: -0.0043381, 0.0116247, -0.0043963, 0.0124826, -0.0168207, 0.0160209
4: -0.0147383, 0.0107940, -0.0152304, 0.0117702, -0.0265086, 0.0260244
5: 0.0016503, 0.0240083, 0.0011690, 0.0245908, -0.0229405, 0.0228394
6: -0.0357131, 0.0138441, -0.0370856, 0.0148911, -0.0506042, 0.0509297
7: 0.9467847, 0.9807345, 0.9442673, 0.9808830, -0.0340983, 0.0364671
8: -0.0332632, 0.0206238, -0.0340769, 0.0226620, -0.0559252, 0.0547007
9: -0.0184410, 0.0180274, -0.0197479, 0.0192551, -0.0376961, 0.0377753

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
time: 1.46 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128753, 0.0049850, -0.0144326, 0.0079387, -0.0208141, 0.0194176
1: -0.0082823, 0.0025353, -0.0103778, 0.0037971, -0.0120794, 0.0129130
2: 0.0250157, 0.0590676, 0.0176853, 0.0635502, -0.0385345, 0.0413823
3: -0.0043348, 0.0115758, -0.0045417, 0.0146279, -0.0189626, 0.0161175
4: -0.0147103, 0.0107384, -0.0164607, 0.0142113, -0.0289217, 0.0271990
5: 0.0016777, 0.0239751, -0.0000346, 0.0260472, -0.0243695, 0.0240097
6: -0.0356349, 0.0137845, -0.0405177, 0.0175093, -0.0531442, 0.0543022
7: 0.9469282, 0.9807261, 0.9379725, 0.9812545, -0.0343263, 0.0427536
8: -0.0332168, 0.0205077, -0.0361117, 0.0277584, -0.0609752, 0.0566194
9: -0.0183665, 0.0179574, -0.0230158, 0.0223250, -0.0406915, 0.0409733

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.07 seconds

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

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
time: 1.57 seconds

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
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285427, upper bound: 0.0274104
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.15 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.76 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0301192, upper bound: 0.0293418
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0301192, upper bound: 0.0293418
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0301667, upper bound: 0.0293418
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0301667, upper bound: 0.0293418
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0285424, upper bound: 0.0274104
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0293418
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0293418
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0285427, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.76
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0129301, 0.0050889, -0.0179740, 0.0179337
1: -0.0082955, 0.0025432, -0.0083560, 0.0025796, -0.0108751, 0.0108992
2: 0.0249697, 0.0590957, 0.0247579, 0.0592252, -0.0342555, 0.0343378
3: -0.0043361, 0.0115950, -0.0043421, 0.0116831, -0.0160192, 0.0159370
4: -0.0147213, 0.0107602, -0.0147719, 0.0108605, -0.0255818, 0.0255320
5: 0.0016670, 0.0239881, 0.0016175, 0.0240480, -0.0223810, 0.0223706
6: -0.0356655, 0.0138078, -0.0358065, 0.0139154, -0.0495809, 0.0496144
7: 0.9468721, 0.9807293, 0.9466134, 0.9807446, -0.0338725, 0.0341159
8: -0.0332350, 0.0205532, -0.0333186, 0.0207626, -0.0539976, 0.0538718
9: -0.0183957, 0.0179849, -0.0185300, 0.0181110, -0.0365067, 0.0365148

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0294948
time: 1.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0294948
time: 1.65 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0129301, 0.0050889, -0.0184949, 0.0189217
1: -0.0089964, 0.0029653, -0.0083560, 0.0025796, -0.0115760, 0.0113213
2: 0.0225176, 0.0605952, 0.0247579, 0.0592252, -0.0367076, 0.0358373
3: -0.0044053, 0.0126159, -0.0043421, 0.0116831, -0.0160884, 0.0169579
4: -0.0153068, 0.0119219, -0.0147719, 0.0108605, -0.0261673, 0.0266938
5: 0.0010942, 0.0246813, 0.0016175, 0.0240480, -0.0229538, 0.0230638
6: -0.0372989, 0.0150538, -0.0358065, 0.0139154, -0.0512143, 0.0508604
7: 0.9438763, 0.9809060, 0.9466134, 0.9807446, -0.0368683, 0.0342926
8: -0.0342033, 0.0229786, -0.0333186, 0.0207626, -0.0549660, 0.0562972
9: -0.0199509, 0.0194459, -0.0185300, 0.0181110, -0.0380620, 0.0379758

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0294948
time: 1.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0294948
time: 2.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0134493, 0.0060737, -0.0189588, 0.0184529
1: -0.0082955, 0.0025432, -0.0090546, 0.0030003, -0.0112958, 0.0115978
2: 0.0249697, 0.0590957, 0.0223139, 0.0607197, -0.0357501, 0.0367818
3: -0.0043361, 0.0115950, -0.0044110, 0.0127007, -0.0170368, 0.0160060
4: -0.0147213, 0.0107602, -0.0153554, 0.0120184, -0.0267397, 0.0261156
5: 0.0016670, 0.0239881, 0.0010466, 0.0247388, -0.0230719, 0.0229415
6: -0.0356655, 0.0138078, -0.0374345, 0.0151573, -0.0508228, 0.0512424
7: 0.9468721, 0.9807293, 0.9436274, 0.9809207, -0.0340487, 0.0371020
8: -0.0332350, 0.0205532, -0.0342838, 0.0231801, -0.0564151, 0.0548370
9: -0.0183957, 0.0179849, -0.0200801, 0.0195672, -0.0379629, 0.0380650

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
time: 1.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0134493, 0.0060737, -0.0194797, 0.0194409
1: -0.0089964, 0.0029653, -0.0090546, 0.0030003, -0.0119968, 0.0120199
2: 0.0225176, 0.0605952, 0.0223139, 0.0607197, -0.0382021, 0.0382813
3: -0.0044053, 0.0126159, -0.0044110, 0.0127007, -0.0171060, 0.0170269
4: -0.0153068, 0.0119219, -0.0153554, 0.0120184, -0.0273252, 0.0272774
5: 0.0010942, 0.0246813, 0.0010466, 0.0247388, -0.0236446, 0.0236346
6: -0.0372989, 0.0150538, -0.0374345, 0.0151573, -0.0524562, 0.0524884
7: 0.9438763, 0.9809060, 0.9436274, 0.9809207, -0.0370445, 0.0372787
8: -0.0342033, 0.0229786, -0.0342838, 0.0231801, -0.0573834, 0.0572624
9: -0.0199509, 0.0194459, -0.0200801, 0.0195672, -0.0395182, 0.0395260

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
time: 1.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0129184, 0.0050667, -0.0190928, 0.0200861
1: -0.0098308, 0.0034677, -0.0083402, 0.0025701, -0.0124009, 0.0118080
2: 0.0195986, 0.0623801, 0.0248131, 0.0591914, -0.0395928, 0.0375670
3: -0.0044877, 0.0138312, -0.0043405, 0.0116601, -0.0161478, 0.0181717
4: -0.0160038, 0.0133048, -0.0147587, 0.0108344, -0.0268381, 0.0280635
5: 0.0004124, 0.0255063, 0.0016304, 0.0240324, -0.0236200, 0.0238759
6: -0.0392432, 0.0165370, -0.0357698, 0.0138874, -0.0531306, 0.0523068
7: 0.9403101, 0.9811166, 0.9466808, 0.9807407, -0.0404305, 0.0344358
8: -0.0353561, 0.0258659, -0.0332968, 0.0207081, -0.0560641, 0.0591627
9: -0.0218022, 0.0211850, -0.0184950, 0.0180782, -0.0398804, 0.0396800

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0145038, 0.0080739, -0.0129184, 0.0050667, -0.0195705, 0.0209923
1: -0.0104737, 0.0038549, -0.0083402, 0.0025701, -0.0130438, 0.0121951
2: 0.0173498, 0.0637554, 0.0248131, 0.0591914, -0.0418417, 0.0389423
3: -0.0045511, 0.0147675, -0.0043405, 0.0116601, -0.0162113, 0.0191080
4: -0.0165408, 0.0143703, -0.0147587, 0.0108344, -0.0273751, 0.0291290
5: -0.0001129, 0.0261420, 0.0016304, 0.0240324, -0.0241453, 0.0245117
6: -0.0407412, 0.0176798, -0.0357698, 0.0138874, -0.0546286, 0.0534495
7: 0.9375624, 0.9812785, 0.9466808, 0.9807407, -0.0431783, 0.0345978
8: -0.0362442, 0.0280903, -0.0332968, 0.0207081, -0.0569522, 0.0613871
9: -0.0232286, 0.0225250, -0.0184950, 0.0180782, -0.0413068, 0.0410200

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.29 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0134376, 0.0060516, -0.0200777, 0.0206054
1: -0.0098308, 0.0034677, -0.0090389, 0.0029909, -0.0128217, 0.0125067
2: 0.0195986, 0.0623801, 0.0223688, 0.0606862, -0.0410875, 0.0400113
3: -0.0044877, 0.0138312, -0.0044095, 0.0126778, -0.0171655, 0.0182407
4: -0.0160038, 0.0133048, -0.0153423, 0.0119924, -0.0279962, 0.0286472
5: 0.0004124, 0.0255063, 0.0010594, 0.0247233, -0.0243109, 0.0244469
6: -0.0392432, 0.0165370, -0.0373980, 0.0151294, -0.0543726, 0.0539350
7: 0.9403101, 0.9811166, 0.9436943, 0.9809169, -0.0406067, 0.0374223
8: -0.0353561, 0.0258659, -0.0342621, 0.0231258, -0.0584819, 0.0601279
9: -0.0218022, 0.0211850, -0.0200453, 0.0195345, -0.0413368, 0.0412303

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285016
time: 1.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0145038, 0.0080739, -0.0134376, 0.0060516, -0.0205554, 0.0215115
1: -0.0104737, 0.0038549, -0.0090389, 0.0029909, -0.0134646, 0.0128938
2: 0.0173498, 0.0637554, 0.0223688, 0.0606862, -0.0433364, 0.0413866
3: -0.0045511, 0.0147675, -0.0044095, 0.0126778, -0.0172290, 0.0191770
4: -0.0165408, 0.0143703, -0.0153423, 0.0119924, -0.0285332, 0.0297126
5: -0.0001129, 0.0261420, 0.0010594, 0.0247233, -0.0248363, 0.0250826
6: -0.0407412, 0.0176798, -0.0373980, 0.0151294, -0.0558706, 0.0550778
7: 0.9375624, 0.9812785, 0.9436943, 0.9809169, -0.0433545, 0.0375842
8: -0.0362442, 0.0280903, -0.0342621, 0.0231258, -0.0593700, 0.0623524
9: -0.0232286, 0.0225250, -0.0200453, 0.0195345, -0.0427631, 0.0425703

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.26 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0129003, 0.0050324, -0.0179175, 0.0179039
1: -0.0082955, 0.0025432, -0.0083159, 0.0025555, -0.0108510, 0.0108591
2: 0.0249697, 0.0590957, 0.0248983, 0.0591394, -0.0341697, 0.0341974
3: -0.0043361, 0.0115950, -0.0043381, 0.0116247, -0.0159608, 0.0159330
4: -0.0147213, 0.0107602, -0.0147383, 0.0107940, -0.0255153, 0.0254985
5: 0.0016670, 0.0239881, 0.0016503, 0.0240083, -0.0223414, 0.0223378
6: -0.0356655, 0.0138078, -0.0357131, 0.0138441, -0.0495096, 0.0495209
7: 0.9468721, 0.9807293, 0.9467847, 0.9807345, -0.0338624, 0.0339446
8: -0.0332350, 0.0205532, -0.0332632, 0.0206238, -0.0538588, 0.0538164
9: -0.0183957, 0.0179849, -0.0184410, 0.0180274, -0.0364231, 0.0364259

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0294948
time: 1.37 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0294948
time: 2.32 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0129003, 0.0050324, -0.0184384, 0.0188919
1: -0.0089964, 0.0029653, -0.0083159, 0.0025555, -0.0115519, 0.0112812
2: 0.0225176, 0.0605952, 0.0248983, 0.0591394, -0.0366218, 0.0356969
3: -0.0044053, 0.0126159, -0.0043381, 0.0116247, -0.0160300, 0.0169540
4: -0.0153068, 0.0119219, -0.0147383, 0.0107940, -0.0261008, 0.0266602
5: 0.0010942, 0.0246813, 0.0016503, 0.0240083, -0.0229141, 0.0230310
6: -0.0372989, 0.0150538, -0.0357131, 0.0138441, -0.0511430, 0.0507669
7: 0.9438763, 0.9809060, 0.9467847, 0.9807345, -0.0368582, 0.0341213
8: -0.0342033, 0.0229786, -0.0332632, 0.0206238, -0.0548271, 0.0562418
9: -0.0199509, 0.0194459, -0.0184410, 0.0180274, -0.0379784, 0.0378869

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0294948
time: 1.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0294948
time: 1.45 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0133744, 0.0059317, -0.0188169, 0.0183780
1: -0.0082955, 0.0025432, -0.0089539, 0.0029397, -0.0112352, 0.0114971
2: 0.0249697, 0.0590957, 0.0226662, 0.0605043, -0.0355346, 0.0364295
3: -0.0043361, 0.0115950, -0.0044011, 0.0125540, -0.0168901, 0.0159960
4: -0.0147213, 0.0107602, -0.0152713, 0.0118515, -0.0265728, 0.0260315
5: 0.0016670, 0.0239881, 0.0011289, 0.0246392, -0.0229723, 0.0228592
6: -0.0356655, 0.0138078, -0.0371998, 0.0149783, -0.0506438, 0.0510077
7: 0.9468721, 0.9807293, 0.9440578, 0.9808955, -0.0340234, 0.0366715
8: -0.0332350, 0.0205532, -0.0341446, 0.0228316, -0.0560666, 0.0546978
9: -0.0183957, 0.0179849, -0.0198567, 0.0193573, -0.0377530, 0.0378415

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
time: 1.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0133744, 0.0059317, -0.0193377, 0.0193661
1: -0.0089964, 0.0029653, -0.0089539, 0.0029397, -0.0119361, 0.0119192
2: 0.0225176, 0.0605952, 0.0226662, 0.0605043, -0.0379867, 0.0379290
3: -0.0044053, 0.0126159, -0.0044011, 0.0125540, -0.0169593, 0.0170170
4: -0.0153068, 0.0119219, -0.0152713, 0.0118515, -0.0271583, 0.0271932
5: 0.0010942, 0.0246813, 0.0011289, 0.0246392, -0.0235450, 0.0235523
6: -0.0372989, 0.0150538, -0.0371998, 0.0149783, -0.0522772, 0.0522537
7: 0.9438763, 0.9809060, 0.9440578, 0.9808955, -0.0370192, 0.0368482
8: -0.0342033, 0.0229786, -0.0341446, 0.0228316, -0.0570349, 0.0571233
9: -0.0199509, 0.0194459, -0.0198567, 0.0193573, -0.0393082, 0.0393025

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 61

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
time: 1.55 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
time: 1.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0128753, 0.0049850, -0.0190111, 0.0200431
1: -0.0098308, 0.0034677, -0.0082823, 0.0025353, -0.0123661, 0.0117501
2: 0.0195986, 0.0623801, 0.0250157, 0.0590676, -0.0394689, 0.0373644
3: -0.0044877, 0.0138312, -0.0043348, 0.0115758, -0.0160635, 0.0181660
4: -0.0160038, 0.0133048, -0.0147103, 0.0107384, -0.0267421, 0.0280152
5: 0.0004124, 0.0255063, 0.0016777, 0.0239751, -0.0235628, 0.0238286
6: -0.0392432, 0.0165370, -0.0356349, 0.0137845, -0.0530276, 0.0521719
7: 0.9403101, 0.9811166, 0.9469282, 0.9807261, -0.0404159, 0.0341884
8: -0.0353561, 0.0258659, -0.0332168, 0.0205077, -0.0558638, 0.0590827
9: -0.0218022, 0.0211850, -0.0183665, 0.0179574, -0.0397597, 0.0395515

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.31 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0145038, 0.0080739, -0.0128753, 0.0049850, -0.0194889, 0.0209493
1: -0.0104737, 0.0038549, -0.0082823, 0.0025353, -0.0130089, 0.0121372
2: 0.0173498, 0.0637554, 0.0250157, 0.0590676, -0.0417178, 0.0387397
3: -0.0045511, 0.0147675, -0.0043348, 0.0115758, -0.0161269, 0.0191023
4: -0.0165408, 0.0143703, -0.0147103, 0.0107384, -0.0272791, 0.0290806
5: -0.0001129, 0.0261420, 0.0016777, 0.0239751, -0.0240881, 0.0244643
6: -0.0407412, 0.0176798, -0.0356349, 0.0137845, -0.0545256, 0.0533146
7: 0.9375624, 0.9812785, 0.9469282, 0.9807261, -0.0431637, 0.0343503
8: -0.0362442, 0.0280903, -0.0332168, 0.0205077, -0.0567519, 0.0613071
9: -0.0232286, 0.0225250, -0.0183665, 0.0179574, -0.0411861, 0.0408915

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.16 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0133492, 0.0058838, -0.0199099, 0.0205169
1: -0.0098308, 0.0034677, -0.0089199, 0.0029192, -0.0127500, 0.0123877
2: 0.0195986, 0.0623801, 0.0227852, 0.0604315, -0.0408329, 0.0395949
3: -0.0044877, 0.0138312, -0.0043977, 0.0125045, -0.0169921, 0.0182289
4: -0.0160038, 0.0133048, -0.0152429, 0.0117951, -0.0277989, 0.0285477
5: 0.0004124, 0.0255063, 0.0011567, 0.0246056, -0.0241932, 0.0243496
6: -0.0392432, 0.0165370, -0.0371206, 0.0149178, -0.0541610, 0.0536576
7: 0.9403101, 0.9811166, 0.9442031, 0.9808869, -0.0405768, 0.0369135
8: -0.0353561, 0.0258659, -0.0340976, 0.0227140, -0.0580700, 0.0599635
9: -0.0218022, 0.0211850, -0.0197812, 0.0192864, -0.0410887, 0.0409662

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285016
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.44 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0145038, 0.0080739, -0.0133492, 0.0058838, -0.0203877, 0.0214231
1: -0.0104737, 0.0038549, -0.0089199, 0.0029192, -0.0133929, 0.0127748
2: 0.0173498, 0.0637554, 0.0227852, 0.0604315, -0.0430818, 0.0409702
3: -0.0045511, 0.0147675, -0.0043977, 0.0125045, -0.0170556, 0.0191653
4: -0.0165408, 0.0143703, -0.0152429, 0.0117951, -0.0283359, 0.0296132
5: -0.0001129, 0.0261420, 0.0011567, 0.0246056, -0.0247186, 0.0249853
6: -0.0407412, 0.0176798, -0.0371206, 0.0149178, -0.0556590, 0.0548004
7: 0.9375624, 0.9812785, 0.9442031, 0.9808869, -0.0433245, 0.0370754
8: -0.0362442, 0.0280903, -0.0340976, 0.0227140, -0.0589581, 0.0621879
9: -0.0232286, 0.0225250, -0.0197812, 0.0192864, -0.0425151, 0.0423062

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.36 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0128112, 0.0048635, -0.0134507, 0.0060765, -0.0188877, 0.0183142
1: -0.0081961, 0.0024833, -0.0090566, 0.0030015, -0.0111976, 0.0115399
2: 0.0253174, 0.0588830, 0.0223071, 0.0607240, -0.0354066, 0.0365760
3: -0.0043263, 0.0114502, -0.0044112, 0.0127035, -0.0170298, 0.0158614
4: -0.0146383, 0.0105954, -0.0153571, 0.0120217, -0.0266599, 0.0259525
5: 0.0017482, 0.0238898, 0.0010450, 0.0247408, -0.0229926, 0.0228448
6: -0.0354339, 0.0136311, -0.0374392, 0.0151608, -0.0505947, 0.0510703
7: 0.9472969, 0.9807042, 0.9436190, 0.9809213, -0.0336244, 0.0370852
8: -0.0330976, 0.0202093, -0.0342865, 0.0231869, -0.0562845, 0.0544958
9: -0.0181751, 0.0177777, -0.0200845, 0.0195713, -0.0377464, 0.0378621

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
time: 1.36 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
time: 1.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0139148, 0.0069566, -0.0134507, 0.0060765, -0.0199912, 0.0204073
1: -0.0096810, 0.0033775, -0.0090566, 0.0030015, -0.0126825, 0.0124341
2: 0.0201227, 0.0620596, 0.0223071, 0.0607240, -0.0406012, 0.0397526
3: -0.0044729, 0.0136130, -0.0044112, 0.0127035, -0.0171764, 0.0180242
4: -0.0158786, 0.0130565, -0.0153571, 0.0120217, -0.0279003, 0.0284136
5: 0.0005348, 0.0253582, 0.0010450, 0.0247408, -0.0242060, 0.0243132
6: -0.0388941, 0.0162707, -0.0374392, 0.0151608, -0.0540549, 0.0537099
7: 0.9409504, 0.9810788, 0.9436190, 0.9809213, -0.0399709, 0.0374598
8: -0.0351491, 0.0253475, -0.0342865, 0.0231869, -0.0583360, 0.0596340
9: -0.0214699, 0.0208728, -0.0200845, 0.0195713, -0.0410412, 0.0409572

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
time: 1.43 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128419, 0.0049216, -0.0145496, 0.0081607, -0.0210026, 0.0194712
1: -0.0082373, 0.0025082, -0.0105352, 0.0038919, -0.0121293, 0.0130434
2: 0.0251730, 0.0589713, 0.0171345, 0.0638870, -0.0387140, 0.0418368
3: -0.0043303, 0.0115103, -0.0045572, 0.0148572, -0.0191875, 0.0160675
4: -0.0146727, 0.0106638, -0.0165922, 0.0144723, -0.0291450, 0.0272560
5: 0.0017145, 0.0239306, -0.0001632, 0.0262029, -0.0244884, 0.0240939
6: -0.0355300, 0.0137045, -0.0408846, 0.0177892, -0.0533192, 0.0545891
7: 0.9471205, 0.9807147, 0.9372995, 0.9812942, -0.0341737, 0.0434152
8: -0.0331546, 0.0203520, -0.0363292, 0.0283033, -0.0614579, 0.0566813
9: -0.0182667, 0.0178637, -0.0233652, 0.0226532, -0.0409199, 0.0412289

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0148214, 0.0086762, -0.0145315, 0.0081264, -0.0229478, 0.0232077
1: -0.0109010, 0.0041122, -0.0105109, 0.0038773, -0.0147783, 0.0146231
2: 0.0158549, 0.0646695, 0.0172196, 0.0638350, -0.0479801, 0.0474499
3: -0.0045933, 0.0153899, -0.0045548, 0.0148217, -0.0194151, 0.0199448
4: -0.0168977, 0.0150786, -0.0165719, 0.0144320, -0.0313297, 0.0316504
5: -0.0004621, 0.0265646, -0.0001433, 0.0261788, -0.0266410, 0.0267079
6: -0.0417370, 0.0184394, -0.0408279, 0.0177459, -0.0594829, 0.0592673
7: 0.9357362, 0.9813864, 0.9374034, 0.9812880, -0.0455518, 0.0439829
8: -0.0368345, 0.0295690, -0.0362956, 0.0282190, -0.0650536, 0.0658646
9: -0.0241768, 0.0234156, -0.0233112, 0.0226025, -0.0467793, 0.0467268

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

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

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
time: 1.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0143816, 0.0078421, -0.0134682, 0.0061095, -0.0204911, 0.0213102
1: -0.0103092, 0.0037558, -0.0090800, 0.0030156, -0.0133248, 0.0128359
2: 0.0179252, 0.0634035, 0.0222251, 0.0607741, -0.0428489, 0.0411784
3: -0.0045349, 0.0145280, -0.0044135, 0.0127377, -0.0172726, 0.0189415
4: -0.0164034, 0.0140977, -0.0153767, 0.0120605, -0.0284639, 0.0294744
5: 0.0000215, 0.0259794, 0.0010259, 0.0247639, -0.0247425, 0.0249535
6: -0.0403579, 0.0173874, -0.0374937, 0.0152025, -0.0555604, 0.0548811
7: 0.9382654, 0.9812371, 0.9435188, 0.9809272, -0.0426618, 0.0377183
8: -0.0360169, 0.0275211, -0.0343189, 0.0232680, -0.0592850, 0.0618400
9: -0.0228637, 0.0221821, -0.0201365, 0.0196202, -0.0424838, 0.0423186

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
time: 1.37 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
time: 1.46 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0133155, 0.0058200, -0.0145669, 0.0081935, -0.0215090, 0.0203869
1: -0.0088747, 0.0028920, -0.0105585, 0.0039060, -0.0127806, 0.0134505
2: 0.0229435, 0.0603347, 0.0170529, 0.0639369, -0.0409933, 0.0432818
3: -0.0043933, 0.0124385, -0.0045595, 0.0148911, -0.0192844, 0.0169981
4: -0.0152051, 0.0117201, -0.0166117, 0.0145109, -0.0297160, 0.0283318
5: 0.0011937, 0.0245608, -0.0001823, 0.0262259, -0.0250322, 0.0247431
6: -0.0370151, 0.0148374, -0.0409389, 0.0178306, -0.0548457, 0.0557763
7: 0.9443967, 0.9808753, 0.9371998, 0.9813001, -0.0369034, 0.0436755
8: -0.0340351, 0.0225573, -0.0363614, 0.0283839, -0.0624190, 0.0589187
9: -0.0196808, 0.0191921, -0.0234168, 0.0227018, -0.0423826, 0.0426089

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 2.16 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0152935, 0.0095718, -0.0145487, 0.0081590, -0.0234525, 0.0241205
1: -0.0115363, 0.0044948, -0.0105340, 0.0038912, -0.0154275, 0.0150288
2: 0.0136324, 0.0660286, 0.0171386, 0.0638845, -0.0502521, 0.0488900
3: -0.0046561, 0.0163153, -0.0045571, 0.0148555, -0.0195115, 0.0208724
4: -0.0174284, 0.0161315, -0.0165912, 0.0144704, -0.0318988, 0.0327228
5: -0.0009813, 0.0271928, -0.0001623, 0.0262017, -0.0271830, 0.0273551
6: -0.0432174, 0.0195687, -0.0408819, 0.0177871, -0.0610045, 0.0604506
7: 0.9330209, 0.9815466, 0.9373045, 0.9812939, -0.0482730, 0.0442421
8: -0.0377122, 0.0317672, -0.0363276, 0.0282992, -0.0660115, 0.0680948
9: -0.0255864, 0.0247398, -0.0233626, 0.0226508, -0.0482372, 0.0481024

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.42 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289450
time: 2.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0139148, 0.0069566, -0.0133380, 0.0058626, -0.0197774, 0.0202946
1: -0.0096810, 0.0033775, -0.0089049, 0.0029102, -0.0125912, 0.0122824
2: 0.0201227, 0.0620596, 0.0228377, 0.0603994, -0.0402767, 0.0392219
3: -0.0044729, 0.0136130, -0.0043963, 0.0124826, -0.0169555, 0.0180092
4: -0.0158786, 0.0130565, -0.0152304, 0.0117702, -0.0276489, 0.0282869
5: 0.0005348, 0.0253582, 0.0011690, 0.0245908, -0.0240560, 0.0241892
6: -0.0388941, 0.0162707, -0.0370856, 0.0148911, -0.0537853, 0.0533563
7: 0.9409504, 0.9810788, 0.9442673, 0.9808830, -0.0399326, 0.0368115
8: -0.0351491, 0.0253475, -0.0340769, 0.0226620, -0.0578111, 0.0594244
9: -0.0214699, 0.0208728, -0.0197479, 0.0192551, -0.0407250, 0.0406207

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289450
time: 1.49 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
time: 1.30 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128419, 0.0049216, -0.0144326, 0.0079387, -0.0207806, 0.0193542
1: -0.0082373, 0.0025082, -0.0103778, 0.0037971, -0.0120344, 0.0128859
2: 0.0251730, 0.0589713, 0.0176853, 0.0635502, -0.0383772, 0.0412860
3: -0.0043303, 0.0115103, -0.0045417, 0.0146279, -0.0189582, 0.0160519
4: -0.0146727, 0.0106638, -0.0164607, 0.0142113, -0.0288841, 0.0271245
5: 0.0017145, 0.0239306, -0.0000346, 0.0260472, -0.0243327, 0.0239652
6: -0.0355300, 0.0137045, -0.0405177, 0.0175093, -0.0530393, 0.0542222
7: 0.9471205, 0.9807147, 0.9379725, 0.9812545, -0.0341339, 0.0427423
8: -0.0331546, 0.0203520, -0.0361117, 0.0277584, -0.0609131, 0.0564637
9: -0.0182667, 0.0178637, -0.0230158, 0.0223250, -0.0405917, 0.0408795

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.51 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0148214, 0.0086762, -0.0144070, 0.0078902, -0.0227116, 0.0230833
1: -0.0109010, 0.0041122, -0.0103433, 0.0037764, -0.0146774, 0.0144555
2: 0.0158549, 0.0646695, 0.0178056, 0.0634766, -0.0476217, 0.0468639
3: -0.0045933, 0.0153899, -0.0045383, 0.0145778, -0.0191711, 0.0199282
4: -0.0168977, 0.0150786, -0.0164319, 0.0141543, -0.0310521, 0.0315105
5: -0.0004621, 0.0265646, -0.0000065, 0.0260132, -0.0264753, 0.0265711
6: -0.0417370, 0.0184394, -0.0404375, 0.0174481, -0.0591851, 0.0588769
7: 0.9357362, 0.9813864, 0.9381194, 0.9812458, -0.0455096, 0.0432670
8: -0.0368345, 0.0295690, -0.0360642, 0.0276394, -0.0644740, 0.0656331
9: -0.0241768, 0.0234156, -0.0229395, 0.0222534, -0.0464301, 0.0463552

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0132872, 0.0057663, -0.0133575, 0.0058996, -0.0191868, 0.0191238
1: -0.0088365, 0.0028690, -0.0089311, 0.0029260, -0.0117625, 0.0118001
2: 0.0230768, 0.0602532, 0.0227461, 0.0604555, -0.0373787, 0.0375071
3: -0.0043895, 0.0123830, -0.0043988, 0.0125208, -0.0169103, 0.0167819
4: -0.0151733, 0.0116569, -0.0152523, 0.0118137, -0.0269869, 0.0269092
5: 0.0012248, 0.0245232, 0.0011476, 0.0246167, -0.0233918, 0.0233756
6: -0.0369263, 0.0147696, -0.0371466, 0.0149378, -0.0518641, 0.0519163
7: 0.9445596, 0.9808658, 0.9441553, 0.9808896, -0.0363300, 0.0367104
8: -0.0339825, 0.0224254, -0.0341131, 0.0227527, -0.0567351, 0.0565385
9: -0.0195962, 0.0191127, -0.0198060, 0.0193097, -0.0389059, 0.0389187

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
time: 2.36 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0143816, 0.0078421, -0.0133575, 0.0058996, -0.0202812, 0.0211995
1: -0.0103092, 0.0037558, -0.0089311, 0.0029260, -0.0132351, 0.0126869
2: 0.0179252, 0.0634035, 0.0227461, 0.0604555, -0.0425303, 0.0406574
3: -0.0045349, 0.0145280, -0.0043988, 0.0125208, -0.0170557, 0.0189268
4: -0.0164034, 0.0140977, -0.0152523, 0.0118137, -0.0282171, 0.0293500
5: 0.0000215, 0.0259794, 0.0011476, 0.0246167, -0.0245952, 0.0248318
6: -0.0403579, 0.0173874, -0.0371466, 0.0149378, -0.0552957, 0.0545340
7: 0.9382654, 0.9812371, 0.9441553, 0.9808896, -0.0426242, 0.0370818
8: -0.0360169, 0.0275211, -0.0341131, 0.0227527, -0.0587696, 0.0616343
9: -0.0228637, 0.0221821, -0.0198060, 0.0193097, -0.0421734, 0.0419881

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
time: 1.49 seconds

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

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.19 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.14 seconds

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

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.12 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.63 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0294948
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0294948
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0294948
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0294948
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0300605, upper bound: 0.0293418
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285016
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0294948
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0294948
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0294948
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0294948
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0301367, upper bound: 0.0293418
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0286323
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285016
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0285424
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0294948, upper bound: 0.0301367
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289450
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0289450
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0293418, upper bound: 0.0301667
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.63
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0128851, 0.0050036, -0.0178887, 0.0178887
1: -0.0082955, 0.0025432, -0.0082955, 0.0025432, -0.0108386, 0.0108386
2: 0.0249697, 0.0590957, 0.0249697, 0.0590957, -0.0341261, 0.0341261
3: -0.0043361, 0.0115950, -0.0043361, 0.0115950, -0.0159310, 0.0159310
4: -0.0147213, 0.0107602, -0.0147213, 0.0107602, -0.0254815, 0.0254815
5: 0.0016670, 0.0239881, 0.0016670, 0.0239881, -0.0223212, 0.0223212
6: -0.0356655, 0.0138078, -0.0356655, 0.0138078, -0.0494733, 0.0494733
7: 0.9468721, 0.9807293, 0.9468721, 0.9807293, -0.0338573, 0.0338573
8: -0.0332350, 0.0205532, -0.0332350, 0.0205532, -0.0537882, 0.0537882
9: -0.0183957, 0.0179849, -0.0183957, 0.0179849, -0.0363806, 0.0363806

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290794, upper bound: 0.0274471
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282359, upper bound: 0.0274471
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0140261, 0.0071677, -0.0200529, 0.0190297
1: -0.0082955, 0.0025432, -0.0098308, 0.0034677, -0.0117632, 0.0123740
2: 0.0249697, 0.0590957, 0.0195986, 0.0623801, -0.0374105, 0.0394971
3: -0.0043361, 0.0115950, -0.0044877, 0.0138312, -0.0181673, 0.0160826
4: -0.0147213, 0.0107602, -0.0160038, 0.0133048, -0.0280261, 0.0267639
5: 0.0016670, 0.0239881, 0.0004124, 0.0255063, -0.0238394, 0.0235758
6: -0.0356655, 0.0138078, -0.0392432, 0.0165370, -0.0522025, 0.0530510
7: 0.9468721, 0.9807293, 0.9403101, 0.9811166, -0.0342445, 0.0404192
8: -0.0332350, 0.0205532, -0.0353561, 0.0258659, -0.0591008, 0.0559093
9: -0.0183957, 0.0179849, -0.0218022, 0.0211850, -0.0395807, 0.0397871

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290794, upper bound: 0.0274471
time: 1.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282359, upper bound: 0.0274471
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0128851, 0.0050036, -0.0184096, 0.0188767
1: -0.0089964, 0.0029653, -0.0082955, 0.0025432, -0.0115396, 0.0112608
2: 0.0225176, 0.0605952, 0.0249697, 0.0590957, -0.0365781, 0.0356255
3: -0.0044053, 0.0126159, -0.0043361, 0.0115950, -0.0160002, 0.0169520
4: -0.0153068, 0.0119219, -0.0147213, 0.0107602, -0.0260670, 0.0266432
5: 0.0010942, 0.0246813, 0.0016670, 0.0239881, -0.0228939, 0.0230143
6: -0.0372989, 0.0150538, -0.0356655, 0.0138078, -0.0511067, 0.0507193
7: 0.9438763, 0.9809060, 0.9468721, 0.9807293, -0.0368531, 0.0340340
8: -0.0342033, 0.0229786, -0.0332350, 0.0205532, -0.0547565, 0.0562136
9: -0.0199509, 0.0194459, -0.0183957, 0.0179849, -0.0379358, 0.0378416

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274471
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274471
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0140261, 0.0071677, -0.0205738, 0.0200177
1: -0.0089964, 0.0029653, -0.0098308, 0.0034677, -0.0124642, 0.0127961
2: 0.0225176, 0.0605952, 0.0195986, 0.0623801, -0.0398625, 0.0409965
3: -0.0044053, 0.0126159, -0.0044877, 0.0138312, -0.0182365, 0.0171036
4: -0.0153068, 0.0119219, -0.0160038, 0.0133048, -0.0286116, 0.0279257
5: 0.0010942, 0.0246813, 0.0004124, 0.0255063, -0.0244121, 0.0242689
6: -0.0372989, 0.0150538, -0.0392432, 0.0165370, -0.0538359, 0.0542970
7: 0.9438763, 0.9809060, 0.9403101, 0.9811166, -0.0372403, 0.0405959
8: -0.0342033, 0.0229786, -0.0353561, 0.0258659, -0.0600692, 0.0583347
9: -0.0199509, 0.0194459, -0.0218022, 0.0211850, -0.0411360, 0.0412481

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274471
time: 1.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274471
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0134060, 0.0059916, -0.0188767, 0.0184096
1: -0.0082955, 0.0025432, -0.0089964, 0.0029653, -0.0112608, 0.0115396
2: 0.0249697, 0.0590957, 0.0225176, 0.0605952, -0.0356255, 0.0365781
3: -0.0043361, 0.0115950, -0.0044053, 0.0126159, -0.0169520, 0.0160002
4: -0.0147213, 0.0107602, -0.0153068, 0.0119219, -0.0266432, 0.0260670
5: 0.0016670, 0.0239881, 0.0010942, 0.0246813, -0.0230143, 0.0228939
6: -0.0356655, 0.0138078, -0.0372989, 0.0150538, -0.0507193, 0.0511067
7: 0.9468721, 0.9807293, 0.9438763, 0.9809060, -0.0340340, 0.0368531
8: -0.0332350, 0.0205532, -0.0342033, 0.0229786, -0.0562136, 0.0547565
9: -0.0183957, 0.0179849, -0.0199509, 0.0194459, -0.0378416, 0.0379358

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0145038, 0.0080739, -0.0209590, 0.0195074
1: -0.0082955, 0.0025432, -0.0104737, 0.0038549, -0.0121503, 0.0130168
2: 0.0249697, 0.0590957, 0.0173498, 0.0637554, -0.0387857, 0.0417459
3: -0.0043361, 0.0115950, -0.0045511, 0.0147675, -0.0191036, 0.0161461
4: -0.0147213, 0.0107602, -0.0165408, 0.0143703, -0.0290916, 0.0273010
5: 0.0016670, 0.0239881, -0.0001129, 0.0261420, -0.0244751, 0.0241011
6: -0.0356655, 0.0138078, -0.0407412, 0.0176798, -0.0533452, 0.0545490
7: 0.9468721, 0.9807293, 0.9375624, 0.9812785, -0.0344065, 0.0431669
8: -0.0332350, 0.0205532, -0.0362442, 0.0280903, -0.0613253, 0.0567974
9: -0.0183957, 0.0179849, -0.0232286, 0.0225250, -0.0409207, 0.0412135

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0134060, 0.0059916, -0.0193976, 0.0193976
1: -0.0089964, 0.0029653, -0.0089964, 0.0029653, -0.0119617, 0.0119617
2: 0.0225176, 0.0605952, 0.0225176, 0.0605952, -0.0380776, 0.0380776
3: -0.0044053, 0.0126159, -0.0044053, 0.0126159, -0.0170212, 0.0170212
4: -0.0153068, 0.0119219, -0.0153068, 0.0119219, -0.0272287, 0.0272287
5: 0.0010942, 0.0246813, 0.0010942, 0.0246813, -0.0235871, 0.0235871
6: -0.0372989, 0.0150538, -0.0372989, 0.0150538, -0.0523527, 0.0523527
7: 0.9438763, 0.9809060, 0.9438763, 0.9809060, -0.0370297, 0.0370297
8: -0.0342033, 0.0229786, -0.0342033, 0.0229786, -0.0571819, 0.0571819
9: -0.0199509, 0.0194459, -0.0199509, 0.0194459, -0.0393968, 0.0393968

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274104
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274104
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0145038, 0.0080739, -0.0214799, 0.0204955
1: -0.0089964, 0.0029653, -0.0104737, 0.0038549, -0.0128513, 0.0134389
2: 0.0225176, 0.0605952, 0.0173498, 0.0637554, -0.0412378, 0.0432454
3: -0.0044053, 0.0126159, -0.0045511, 0.0147675, -0.0191728, 0.0171670
4: -0.0153068, 0.0119219, -0.0165408, 0.0143703, -0.0296771, 0.0284627
5: 0.0010942, 0.0246813, -0.0001129, 0.0261420, -0.0250478, 0.0247942
6: -0.0372989, 0.0150538, -0.0407412, 0.0176798, -0.0549786, 0.0557950
7: 0.9438763, 0.9809060, 0.9375624, 0.9812785, -0.0374023, 0.0433436
8: -0.0342033, 0.0229786, -0.0362442, 0.0280903, -0.0622936, 0.0592228
9: -0.0199509, 0.0194459, -0.0232286, 0.0225250, -0.0424759, 0.0426745

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274104
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274104
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0128970, 0.0050262, -0.0190523, 0.0200648
1: -0.0098308, 0.0034677, -0.0083115, 0.0025528, -0.0123836, 0.0117792
2: 0.0195986, 0.0623801, 0.0249137, 0.0591299, -0.0395313, 0.0374665
3: -0.0044877, 0.0138312, -0.0043377, 0.0116183, -0.0161059, 0.0181689
4: -0.0160038, 0.0133048, -0.0147347, 0.0107867, -0.0267905, 0.0280395
5: 0.0004124, 0.0255063, 0.0016539, 0.0240039, -0.0235916, 0.0238524
6: -0.0392432, 0.0165370, -0.0357028, 0.0138363, -0.0530795, 0.0522398
7: 0.9403101, 0.9811166, 0.9468036, 0.9807334, -0.0404233, 0.0343130
8: -0.0353561, 0.0258659, -0.0332571, 0.0206086, -0.0559647, 0.0591229
9: -0.0218022, 0.0211850, -0.0184312, 0.0180182, -0.0398205, 0.0396162

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274471
time: 1.43 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274471
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0140097, 0.0071367, -0.0148829, 0.0087929, -0.0228026, 0.0220196
1: -0.0098088, 0.0034545, -0.0109837, 0.0041620, -0.0139708, 0.0144382
2: 0.0196757, 0.0623331, 0.0155655, 0.0648465, -0.0451709, 0.0467676
3: -0.0044855, 0.0137991, -0.0046015, 0.0155104, -0.0199959, 0.0184007
4: -0.0159854, 0.0132684, -0.0169668, 0.0152157, -0.0312011, 0.0302352
5: 0.0004303, 0.0254846, -0.0005297, 0.0266464, -0.0262160, 0.0260143
6: -0.0391919, 0.0164979, -0.0419297, 0.0185864, -0.0577784, 0.0584276
7: 0.9404041, 0.9811110, 0.9353825, 0.9814073, -0.0410033, 0.0457284
8: -0.0353257, 0.0257897, -0.0369488, 0.0298552, -0.0651809, 0.0627385
9: -0.0217535, 0.0211392, -0.0243603, 0.0235881, -0.0453415, 0.0454995

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0270152, upper bound: 0.0265075
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0265075
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0145038, 0.0080739, -0.0128970, 0.0050262, -0.0195300, 0.0209709
1: -0.0104737, 0.0038549, -0.0083115, 0.0025528, -0.0130265, 0.0121664
2: 0.0173498, 0.0637554, 0.0249137, 0.0591299, -0.0417802, 0.0388417
3: -0.0045511, 0.0147675, -0.0043377, 0.0116183, -0.0161694, 0.0191052
4: -0.0165408, 0.0143703, -0.0147347, 0.0107867, -0.0273275, 0.0291050
5: -0.0001129, 0.0261420, 0.0016539, 0.0240039, -0.0241169, 0.0244882
6: -0.0407412, 0.0176798, -0.0357028, 0.0138363, -0.0545775, 0.0533826
7: 0.9375624, 0.9812785, 0.9468036, 0.9807334, -0.0431710, 0.0344749
8: -0.0362442, 0.0280903, -0.0332571, 0.0206086, -0.0568528, 0.0613474
9: -0.0232286, 0.0225250, -0.0184312, 0.0180182, -0.0412469, 0.0409561

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0144867, 0.0080415, -0.0148829, 0.0087929, -0.0232796, 0.0229244
1: -0.0104507, 0.0038410, -0.0109837, 0.0041620, -0.0146127, 0.0148247
2: 0.0174302, 0.0637062, 0.0155655, 0.0648465, -0.0474163, 0.0481407
3: -0.0045489, 0.0147340, -0.0046015, 0.0155104, -0.0200593, 0.0193356
4: -0.0165216, 0.0143322, -0.0169668, 0.0152157, -0.0317373, 0.0312990
5: -0.0000941, 0.0261193, -0.0005297, 0.0266464, -0.0267405, 0.0266490
6: -0.0406876, 0.0176389, -0.0419297, 0.0185864, -0.0592740, 0.0595686
7: 0.9376608, 0.9812729, 0.9353825, 0.9814073, -0.0437465, 0.0458904
8: -0.0362124, 0.0280107, -0.0369488, 0.0298552, -0.0660676, 0.0649595
9: -0.0231776, 0.0224770, -0.0243603, 0.0235881, -0.0467656, 0.0468373

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269805, upper bound: 0.0265075
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264988, upper bound: 0.0265075
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0134153, 0.0060092, -0.0200353, 0.0205831
1: -0.0098308, 0.0034677, -0.0090089, 0.0029728, -0.0128036, 0.0124766
2: 0.0195986, 0.0623801, 0.0224739, 0.0606219, -0.0410233, 0.0399062
3: -0.0044877, 0.0138312, -0.0044065, 0.0126341, -0.0171218, 0.0182377
4: -0.0160038, 0.0133048, -0.0153173, 0.0119426, -0.0279464, 0.0286221
5: 0.0004124, 0.0255063, 0.0010840, 0.0246936, -0.0242812, 0.0244223
6: -0.0392432, 0.0165370, -0.0373280, 0.0150760, -0.0543192, 0.0538650
7: 0.9403101, 0.9811166, 0.9438229, 0.9809093, -0.0405992, 0.0372937
8: -0.0353561, 0.0258659, -0.0342206, 0.0230219, -0.0583779, 0.0600865
9: -0.0218022, 0.0211850, -0.0199786, 0.0194719, -0.0412741, 0.0411637

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0140097, 0.0071367, -0.0153611, 0.0096999, -0.0237097, 0.0224978
1: -0.0098088, 0.0034545, -0.0116272, 0.0045495, -0.0143583, 0.0150817
2: 0.0196757, 0.0623331, 0.0133143, 0.0662231, -0.0465475, 0.0490187
3: -0.0044855, 0.0137991, -0.0046650, 0.0164477, -0.0209332, 0.0184642
4: -0.0159854, 0.0132684, -0.0175044, 0.0162822, -0.0322676, 0.0307727
5: 0.0004303, 0.0254846, -0.0010556, 0.0272827, -0.0268524, 0.0265402
6: -0.0391919, 0.0164979, -0.0434292, 0.0197303, -0.0589223, 0.0599271
7: 0.9404041, 0.9811110, 0.9326323, 0.9815695, -0.0411654, 0.0484787
8: -0.0353257, 0.0257897, -0.0378378, 0.0320819, -0.0674075, 0.0636275
9: -0.0217535, 0.0211392, -0.0257881, 0.0249294, -0.0466828, 0.0469273

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0270152, upper bound: 0.0264988
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0264988
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0145038, 0.0080739, -0.0134153, 0.0060092, -0.0205131, 0.0214892
1: -0.0104737, 0.0038549, -0.0090089, 0.0029728, -0.0134465, 0.0128638
2: 0.0173498, 0.0637554, 0.0224739, 0.0606219, -0.0432721, 0.0412815
3: -0.0045511, 0.0147675, -0.0044065, 0.0126341, -0.0171852, 0.0191741
4: -0.0165408, 0.0143703, -0.0153173, 0.0119426, -0.0284834, 0.0296876
5: -0.0001129, 0.0261420, 0.0010840, 0.0246936, -0.0248065, 0.0250581
6: -0.0407412, 0.0176798, -0.0373280, 0.0150760, -0.0558172, 0.0550077
7: 0.9375624, 0.9812785, 0.9438229, 0.9809093, -0.0433469, 0.0374557
8: -0.0362442, 0.0280903, -0.0342206, 0.0230219, -0.0592660, 0.0623109
9: -0.0232286, 0.0225250, -0.0199786, 0.0194719, -0.0427005, 0.0425036

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0144867, 0.0080415, -0.0153611, 0.0096999, -0.0241867, 0.0234026
1: -0.0104507, 0.0038410, -0.0116272, 0.0045495, -0.0150002, 0.0154682
2: 0.0174302, 0.0637062, 0.0133143, 0.0662231, -0.0487929, 0.0503918
3: -0.0045489, 0.0147340, -0.0046650, 0.0164477, -0.0209966, 0.0193991
4: -0.0165216, 0.0143322, -0.0175044, 0.0162822, -0.0328038, 0.0318366
5: -0.0000941, 0.0261193, -0.0010556, 0.0272827, -0.0273769, 0.0271748
6: -0.0406876, 0.0176389, -0.0434292, 0.0197303, -0.0604179, 0.0610681
7: 0.9376608, 0.9812729, 0.9326323, 0.9815695, -0.0439087, 0.0486406
8: -0.0362124, 0.0280107, -0.0378378, 0.0320819, -0.0682943, 0.0658485
9: -0.0231776, 0.0224770, -0.0257881, 0.0249294, -0.0481069, 0.0482651

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269805, upper bound: 0.0264988
time: 1.30 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264988, upper bound: 0.0264988
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0128112, 0.0048635, -0.0177486, 0.0178148
1: -0.0082955, 0.0025432, -0.0081961, 0.0024833, -0.0107788, 0.0107393
2: 0.0249697, 0.0590957, 0.0253174, 0.0588830, -0.0339134, 0.0337783
3: -0.0043361, 0.0115950, -0.0043263, 0.0114502, -0.0157862, 0.0159212
4: -0.0147213, 0.0107602, -0.0146383, 0.0105954, -0.0253167, 0.0253984
5: 0.0016670, 0.0239881, 0.0017482, 0.0238898, -0.0222229, 0.0222399
6: -0.0356655, 0.0138078, -0.0354339, 0.0136311, -0.0492966, 0.0492417
7: 0.9468721, 0.9807293, 0.9472969, 0.9807042, -0.0338322, 0.0334324
8: -0.0332350, 0.0205532, -0.0330976, 0.0202093, -0.0534442, 0.0536508
9: -0.0183957, 0.0179849, -0.0181751, 0.0177777, -0.0361734, 0.0361600

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0291232, upper bound: 0.0274471
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282613, upper bound: 0.0274471
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0139148, 0.0069566, -0.0198417, 0.0189184
1: -0.0082955, 0.0025432, -0.0096810, 0.0033775, -0.0116730, 0.0122242
2: 0.0249697, 0.0590957, 0.0201227, 0.0620596, -0.0370900, 0.0389730
3: -0.0043361, 0.0115950, -0.0044729, 0.0136130, -0.0179491, 0.0160678
4: -0.0147213, 0.0107602, -0.0158786, 0.0130565, -0.0277778, 0.0266388
5: 0.0016670, 0.0239881, 0.0005348, 0.0253582, -0.0236912, 0.0234533
6: -0.0356655, 0.0138078, -0.0388941, 0.0162707, -0.0519362, 0.0527019
7: 0.9468721, 0.9807293, 0.9409504, 0.9810788, -0.0342067, 0.0397789
8: -0.0332350, 0.0205532, -0.0351491, 0.0253475, -0.0585824, 0.0557023
9: -0.0183957, 0.0179849, -0.0214699, 0.0208728, -0.0392685, 0.0394547

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0291232, upper bound: 0.0274471
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282613, upper bound: 0.0274471
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0128112, 0.0048635, -0.0182695, 0.0188029
1: -0.0089964, 0.0029653, -0.0081961, 0.0024833, -0.0114797, 0.0111614
2: 0.0225176, 0.0605952, 0.0253174, 0.0588830, -0.0363654, 0.0352778
3: -0.0044053, 0.0126159, -0.0043263, 0.0114502, -0.0158555, 0.0169422
4: -0.0153068, 0.0119219, -0.0146383, 0.0105954, -0.0259022, 0.0265602
5: 0.0010942, 0.0246813, 0.0017482, 0.0238898, -0.0227956, 0.0229331
6: -0.0372989, 0.0150538, -0.0354339, 0.0136311, -0.0509300, 0.0504877
7: 0.9438763, 0.9809060, 0.9472969, 0.9807042, -0.0368280, 0.0336091
8: -0.0342033, 0.0229786, -0.0330976, 0.0202093, -0.0544126, 0.0560763
9: -0.0199509, 0.0194459, -0.0181751, 0.0177777, -0.0377286, 0.0376210

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289450, upper bound: 0.0274471
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282008, upper bound: 0.0274471
time: 1.37 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0139148, 0.0069566, -0.0203626, 0.0199064
1: -0.0089964, 0.0029653, -0.0096810, 0.0033775, -0.0123739, 0.0126463
2: 0.0225176, 0.0605952, 0.0201227, 0.0620596, -0.0395421, 0.0404725
3: -0.0044053, 0.0126159, -0.0044729, 0.0136130, -0.0180183, 0.0170888
4: -0.0153068, 0.0119219, -0.0158786, 0.0130565, -0.0283633, 0.0278006
5: 0.0010942, 0.0246813, 0.0005348, 0.0253582, -0.0242640, 0.0241465
6: -0.0372989, 0.0150538, -0.0388941, 0.0162707, -0.0535696, 0.0539479
7: 0.9438763, 0.9809060, 0.9409504, 0.9810788, -0.0372025, 0.0399556
8: -0.0342033, 0.0229786, -0.0351491, 0.0253475, -0.0595508, 0.0581277
9: -0.0199509, 0.0194459, -0.0214699, 0.0208728, -0.0408237, 0.0409157

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289450, upper bound: 0.0274471
time: 1.13 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282008, upper bound: 0.0274471
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0132872, 0.0057663, -0.0186514, 0.0182908
1: -0.0082955, 0.0025432, -0.0088365, 0.0028690, -0.0111645, 0.0113797
2: 0.0249697, 0.0590957, 0.0230768, 0.0602532, -0.0352835, 0.0360189
3: -0.0043361, 0.0115950, -0.0043895, 0.0123830, -0.0167191, 0.0159845
4: -0.0147213, 0.0107602, -0.0151733, 0.0116569, -0.0263782, 0.0259334
5: 0.0016670, 0.0239881, 0.0012248, 0.0245232, -0.0228562, 0.0227633
6: -0.0356655, 0.0138078, -0.0369263, 0.0147696, -0.0504351, 0.0507342
7: 0.9468721, 0.9807293, 0.9445596, 0.9808658, -0.0339937, 0.0361698
8: -0.0332350, 0.0205532, -0.0339825, 0.0224254, -0.0556604, 0.0545357
9: -0.0183957, 0.0179849, -0.0195962, 0.0191127, -0.0375083, 0.0375811

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0291114, upper bound: 0.0274104
time: 1.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282607, upper bound: 0.0274104
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128851, 0.0050036, -0.0143816, 0.0078421, -0.0207272, 0.0193852
1: -0.0082955, 0.0025432, -0.0103092, 0.0037558, -0.0120513, 0.0128523
2: 0.0249697, 0.0590957, 0.0179252, 0.0634035, -0.0384338, 0.0411705
3: -0.0043361, 0.0115950, -0.0045349, 0.0145280, -0.0188640, 0.0161299
4: -0.0147213, 0.0107602, -0.0164034, 0.0140977, -0.0288190, 0.0271636
5: 0.0016670, 0.0239881, 0.0000215, 0.0259794, -0.0243124, 0.0239667
6: -0.0356655, 0.0138078, -0.0403579, 0.0173874, -0.0530529, 0.0541658
7: 0.9468721, 0.9807293, 0.9382654, 0.9812371, -0.0343651, 0.0424639
8: -0.0332350, 0.0205532, -0.0360169, 0.0275211, -0.0607561, 0.0565701
9: -0.0183957, 0.0179849, -0.0228637, 0.0221821, -0.0405778, 0.0408485

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0291114, upper bound: 0.0274104
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282607, upper bound: 0.0274104
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0132872, 0.0057663, -0.0191723, 0.0192788
1: -0.0089964, 0.0029653, -0.0088365, 0.0028690, -0.0118654, 0.0118018
2: 0.0225176, 0.0605952, 0.0230768, 0.0602532, -0.0377356, 0.0375183
3: -0.0044053, 0.0126159, -0.0043895, 0.0123830, -0.0167883, 0.0170054
4: -0.0153068, 0.0119219, -0.0151733, 0.0116569, -0.0269637, 0.0270952
5: 0.0010942, 0.0246813, 0.0012248, 0.0245232, -0.0234290, 0.0234564
6: -0.0372989, 0.0150538, -0.0369263, 0.0147696, -0.0520685, 0.0519801
7: 0.9438763, 0.9809060, 0.9445596, 0.9808658, -0.0369895, 0.0363464
8: -0.0342033, 0.0229786, -0.0339825, 0.0224254, -0.0566287, 0.0569611
9: -0.0199509, 0.0194459, -0.0195962, 0.0191127, -0.0390636, 0.0390421

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289450, upper bound: 0.0274104
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282008, upper bound: 0.0274104
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0134060, 0.0059916, -0.0143816, 0.0078421, -0.0212481, 0.0203732
1: -0.0089964, 0.0029653, -0.0103092, 0.0037558, -0.0127522, 0.0132745
2: 0.0225176, 0.0605952, 0.0179252, 0.0634035, -0.0408859, 0.0426700
3: -0.0044053, 0.0126159, -0.0045349, 0.0145280, -0.0189333, 0.0171508
4: -0.0153068, 0.0119219, -0.0164034, 0.0140977, -0.0294045, 0.0283253
5: 0.0010942, 0.0246813, 0.0000215, 0.0259794, -0.0248852, 0.0246598
6: -0.0372989, 0.0150538, -0.0403579, 0.0173874, -0.0546862, 0.0554117
7: 0.9438763, 0.9809060, 0.9382654, 0.9812371, -0.0373608, 0.0426406
8: -0.0342033, 0.0229786, -0.0360169, 0.0275211, -0.0617245, 0.0589956
9: -0.0199509, 0.0194459, -0.0228637, 0.0221821, -0.0421331, 0.0423095

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0289450, upper bound: 0.0274104
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282008, upper bound: 0.0274104
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0128419, 0.0049216, -0.0189477, 0.0200096
1: -0.0098308, 0.0034677, -0.0082373, 0.0025082, -0.0123390, 0.0117051
2: 0.0195986, 0.0623801, 0.0251730, 0.0589713, -0.0393727, 0.0372071
3: -0.0044877, 0.0138312, -0.0043303, 0.0115103, -0.0159979, 0.0181615
4: -0.0160038, 0.0133048, -0.0146727, 0.0106638, -0.0266676, 0.0279776
5: 0.0004124, 0.0255063, 0.0017145, 0.0239306, -0.0235183, 0.0237919
6: -0.0392432, 0.0165370, -0.0355300, 0.0137045, -0.0529477, 0.0520671
7: 0.9403101, 0.9811166, 0.9471205, 0.9807147, -0.0404046, 0.0339960
8: -0.0353561, 0.0258659, -0.0331546, 0.0203520, -0.0557081, 0.0590205
9: -0.0218022, 0.0211850, -0.0182667, 0.0178637, -0.0396659, 0.0394517

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274471
time: 1.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274471
time: 1.15 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0140097, 0.0071367, -0.0148214, 0.0086762, -0.0226860, 0.0219581
1: -0.0098088, 0.0034545, -0.0109010, 0.0041122, -0.0139210, 0.0143555
2: 0.0196757, 0.0623331, 0.0158549, 0.0646695, -0.0449939, 0.0464782
3: -0.0044855, 0.0137991, -0.0045933, 0.0153899, -0.0198755, 0.0183925
4: -0.0159854, 0.0132684, -0.0168977, 0.0150786, -0.0310640, 0.0301661
5: 0.0004303, 0.0254846, -0.0004621, 0.0265646, -0.0261342, 0.0259467
6: -0.0391919, 0.0164979, -0.0417370, 0.0184394, -0.0576313, 0.0582349
7: 0.9404041, 0.9811110, 0.9357362, 0.9813864, -0.0409823, 0.0453748
8: -0.0353257, 0.0257897, -0.0368345, 0.0295690, -0.0648946, 0.0626243
9: -0.0217535, 0.0211392, -0.0241768, 0.0234156, -0.0451691, 0.0453159

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0270328, upper bound: 0.0265131
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0265131
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0145038, 0.0080739, -0.0128419, 0.0049216, -0.0194255, 0.0209158
1: -0.0104737, 0.0038549, -0.0082373, 0.0025082, -0.0129818, 0.0120922
2: 0.0173498, 0.0637554, 0.0251730, 0.0589713, -0.0416216, 0.0385823
3: -0.0045511, 0.0147675, -0.0043303, 0.0115103, -0.0160614, 0.0190979
4: -0.0165408, 0.0143703, -0.0146727, 0.0106638, -0.0272046, 0.0290430
5: -0.0001129, 0.0261420, 0.0017145, 0.0239306, -0.0240436, 0.0244276
6: -0.0407412, 0.0176798, -0.0355300, 0.0137045, -0.0544457, 0.0532098
7: 0.9375624, 0.9812785, 0.9471205, 0.9807147, -0.0431523, 0.0341580
8: -0.0362442, 0.0280903, -0.0331546, 0.0203520, -0.0565962, 0.0612449
9: -0.0232286, 0.0225250, -0.0182667, 0.0178637, -0.0410923, 0.0407917

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
time: 1.29 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0144867, 0.0080415, -0.0148214, 0.0086762, -0.0231630, 0.0228629
1: -0.0104507, 0.0038410, -0.0109010, 0.0041122, -0.0145628, 0.0147420
2: 0.0174302, 0.0637062, 0.0158549, 0.0646695, -0.0472393, 0.0478513
3: -0.0045489, 0.0147340, -0.0045933, 0.0153899, -0.0199388, 0.0193274
4: -0.0165216, 0.0143322, -0.0168977, 0.0150786, -0.0316001, 0.0312299
5: -0.0000941, 0.0261193, -0.0004621, 0.0265646, -0.0266587, 0.0265814
6: -0.0406876, 0.0176389, -0.0417370, 0.0184394, -0.0591269, 0.0593758
7: 0.9376608, 0.9812729, 0.9357362, 0.9813864, -0.0437256, 0.0455368
8: -0.0362124, 0.0280107, -0.0368345, 0.0295690, -0.0657814, 0.0648453
9: -0.0231776, 0.0224770, -0.0241768, 0.0234156, -0.0465932, 0.0466538

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269973, upper bound: 0.0265131
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264988, upper bound: 0.0265131
time: 1.25 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0140261, 0.0071677, -0.0133155, 0.0058200, -0.0198461, 0.0204833
1: -0.0098308, 0.0034677, -0.0088747, 0.0028920, -0.0127228, 0.0123424
2: 0.0195986, 0.0623801, 0.0229435, 0.0603347, -0.0407360, 0.0394366
3: -0.0044877, 0.0138312, -0.0043933, 0.0124385, -0.0169262, 0.0182245
4: -0.0160038, 0.0133048, -0.0152051, 0.0117201, -0.0277239, 0.0285099
5: 0.0004124, 0.0255063, 0.0011937, 0.0245608, -0.0241485, 0.0243126
6: -0.0392432, 0.0165370, -0.0370151, 0.0148374, -0.0540806, 0.0535521
7: 0.9403101, 0.9811166, 0.9443967, 0.9808753, -0.0405652, 0.0367199
8: -0.0353561, 0.0258659, -0.0340351, 0.0225573, -0.0579134, 0.0599010
9: -0.0218022, 0.0211850, -0.0196808, 0.0191921, -0.0409943, 0.0408658

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.16 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0140097, 0.0071367, -0.0152935, 0.0095718, -0.0235815, 0.0224303
1: -0.0098088, 0.0034545, -0.0115363, 0.0044948, -0.0143036, 0.0149908
2: 0.0196757, 0.0623331, 0.0136324, 0.0660286, -0.0463530, 0.0487007
3: -0.0044855, 0.0137991, -0.0046561, 0.0163153, -0.0208008, 0.0184552
4: -0.0159854, 0.0132684, -0.0174284, 0.0161315, -0.0321169, 0.0306968
5: 0.0004303, 0.0254846, -0.0009813, 0.0271928, -0.0267625, 0.0264659
6: -0.0391919, 0.0164979, -0.0432174, 0.0195687, -0.0587607, 0.0597153
7: 0.9404041, 0.9811110, 0.9330209, 0.9815466, -0.0411425, 0.0480901
8: -0.0353257, 0.0257897, -0.0377122, 0.0317672, -0.0670929, 0.0635020
9: -0.0217535, 0.0211392, -0.0255864, 0.0247398, -0.0464933, 0.0467256

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0270328, upper bound: 0.0265043
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0265043
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0145038, 0.0080739, -0.0133155, 0.0058200, -0.0203238, 0.0213894
1: -0.0104737, 0.0038549, -0.0088747, 0.0028920, -0.0133656, 0.0127295
2: 0.0173498, 0.0637554, 0.0229435, 0.0603347, -0.0429849, 0.0408119
3: -0.0045511, 0.0147675, -0.0043933, 0.0124385, -0.0169897, 0.0191608
4: -0.0165408, 0.0143703, -0.0152051, 0.0117201, -0.0282609, 0.0295754
5: -0.0001129, 0.0261420, 0.0011937, 0.0245608, -0.0246738, 0.0249483
6: -0.0407412, 0.0176798, -0.0370151, 0.0148374, -0.0555786, 0.0546949
7: 0.9375624, 0.9812785, 0.9443967, 0.9808753, -0.0433129, 0.0368819
8: -0.0362442, 0.0280903, -0.0340351, 0.0225573, -0.0588015, 0.0621254
9: -0.0232286, 0.0225250, -0.0196808, 0.0191921, -0.0424207, 0.0422057

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.22 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0144867, 0.0080415, -0.0152935, 0.0095718, -0.0240585, 0.0233350
1: -0.0104507, 0.0038410, -0.0115363, 0.0044948, -0.0149454, 0.0153773
2: 0.0174302, 0.0637062, 0.0136324, 0.0660286, -0.0485984, 0.0500738
3: -0.0045489, 0.0147340, -0.0046561, 0.0163153, -0.0208642, 0.0193901
4: -0.0165216, 0.0143322, -0.0174284, 0.0161315, -0.0326531, 0.0317606
5: -0.0000941, 0.0261193, -0.0009813, 0.0271928, -0.0272870, 0.0271006
6: -0.0406876, 0.0176389, -0.0432174, 0.0195687, -0.0602563, 0.0608563
7: 0.9376608, 0.9812729, 0.9330209, 0.9815466, -0.0438858, 0.0482520
8: -0.0362124, 0.0280107, -0.0377122, 0.0317672, -0.0679796, 0.0657230
9: -0.0231776, 0.0224770, -0.0255864, 0.0247398, -0.0479174, 0.0480634

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0269973, upper bound: 0.0265043
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0264988, upper bound: 0.0265043
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0128112, 0.0048635, -0.0128851, 0.0050036, -0.0178148, 0.0177486
1: -0.0081961, 0.0024833, -0.0082955, 0.0025432, -0.0107393, 0.0107788
2: 0.0253174, 0.0588830, 0.0249697, 0.0590957, -0.0337783, 0.0339134
3: -0.0043263, 0.0114502, -0.0043361, 0.0115950, -0.0159212, 0.0157862
4: -0.0146383, 0.0105954, -0.0147213, 0.0107602, -0.0253984, 0.0253167
5: 0.0017482, 0.0238898, 0.0016670, 0.0239881, -0.0222399, 0.0222229
6: -0.0354339, 0.0136311, -0.0356655, 0.0138078, -0.0492417, 0.0492966
7: 0.9472969, 0.9807042, 0.9468721, 0.9807293, -0.0334324, 0.0338322
8: -0.0330976, 0.0202093, -0.0332350, 0.0205532, -0.0536508, 0.0534442
9: -0.0181751, 0.0177777, -0.0183957, 0.0179849, -0.0361600, 0.0361734

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0296077, upper bound: 0.0283447
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128112, 0.0048635, -0.0134060, 0.0059916, -0.0188029, 0.0182695
1: -0.0081961, 0.0024833, -0.0089964, 0.0029653, -0.0111614, 0.0114797
2: 0.0253174, 0.0588830, 0.0225176, 0.0605952, -0.0352778, 0.0363654
3: -0.0043263, 0.0114502, -0.0044053, 0.0126159, -0.0169422, 0.0158555
4: -0.0146383, 0.0105954, -0.0153068, 0.0119219, -0.0265602, 0.0259022
5: 0.0017482, 0.0238898, 0.0010942, 0.0246813, -0.0229331, 0.0227956
6: -0.0354339, 0.0136311, -0.0372989, 0.0150538, -0.0504877, 0.0509300
7: 0.9472969, 0.9807042, 0.9438763, 0.9809060, -0.0336091, 0.0368280
8: -0.0330976, 0.0202093, -0.0342033, 0.0229786, -0.0560763, 0.0544126
9: -0.0181751, 0.0177777, -0.0199509, 0.0194459, -0.0376210, 0.0377286

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0296077, upper bound: 0.0283447
time: 1.26 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0139148, 0.0069566, -0.0128851, 0.0050036, -0.0189184, 0.0198417
1: -0.0096810, 0.0033775, -0.0082955, 0.0025432, -0.0122242, 0.0116730
2: 0.0201227, 0.0620596, 0.0249697, 0.0590957, -0.0389730, 0.0370900
3: -0.0044729, 0.0136130, -0.0043361, 0.0115950, -0.0160678, 0.0179491
4: -0.0158786, 0.0130565, -0.0147213, 0.0107602, -0.0266388, 0.0277778
5: 0.0005348, 0.0253582, 0.0016670, 0.0239881, -0.0234533, 0.0236912
6: -0.0388941, 0.0162707, -0.0356655, 0.0138078, -0.0527019, 0.0519362
7: 0.9409504, 0.9810788, 0.9468721, 0.9807293, -0.0397789, 0.0342067
8: -0.0351491, 0.0253475, -0.0332350, 0.0205532, -0.0557023, 0.0585824
9: -0.0214699, 0.0208728, -0.0183957, 0.0179849, -0.0394547, 0.0392685

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286579, upper bound: 0.0282716
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0139148, 0.0069566, -0.0134060, 0.0059916, -0.0199064, 0.0203626
1: -0.0096810, 0.0033775, -0.0089964, 0.0029653, -0.0126463, 0.0123739
2: 0.0201227, 0.0620596, 0.0225176, 0.0605952, -0.0404725, 0.0395421
3: -0.0044729, 0.0136130, -0.0044053, 0.0126159, -0.0170888, 0.0180183
4: -0.0158786, 0.0130565, -0.0153068, 0.0119219, -0.0278006, 0.0283633
5: 0.0005348, 0.0253582, 0.0010942, 0.0246813, -0.0241465, 0.0242640
6: -0.0388941, 0.0162707, -0.0372989, 0.0150538, -0.0539479, 0.0535696
7: 0.9409504, 0.9810788, 0.9438763, 0.9809060, -0.0399556, 0.0372025
8: -0.0351491, 0.0253475, -0.0342033, 0.0229786, -0.0581277, 0.0595508
9: -0.0214699, 0.0208728, -0.0199509, 0.0194459, -0.0409157, 0.0408237

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286579, upper bound: 0.0282716
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0128419, 0.0049216, -0.0140261, 0.0071677, -0.0200096, 0.0189477
1: -0.0082373, 0.0025082, -0.0098308, 0.0034677, -0.0117051, 0.0123390
2: 0.0251730, 0.0589713, 0.0195986, 0.0623801, -0.0372071, 0.0393727
3: -0.0043303, 0.0115103, -0.0044877, 0.0138312, -0.0181615, 0.0159979
4: -0.0146727, 0.0106638, -0.0160038, 0.0133048, -0.0279776, 0.0266676
5: 0.0017145, 0.0239306, 0.0004124, 0.0255063, -0.0237919, 0.0235183
6: -0.0355300, 0.0137045, -0.0392432, 0.0165370, -0.0520671, 0.0529477
7: 0.9471205, 0.9807147, 0.9403101, 0.9811166, -0.0339960, 0.0404046
8: -0.0331546, 0.0203520, -0.0353561, 0.0258659, -0.0590205, 0.0557081
9: -0.0182667, 0.0178637, -0.0218022, 0.0211850, -0.0394517, 0.0396659

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.30 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128419, 0.0049216, -0.0145038, 0.0080739, -0.0209158, 0.0194255
1: -0.0082373, 0.0025082, -0.0104737, 0.0038549, -0.0120922, 0.0129818
2: 0.0251730, 0.0589713, 0.0173498, 0.0637554, -0.0385823, 0.0416216
3: -0.0043303, 0.0115103, -0.0045511, 0.0147675, -0.0190979, 0.0160614
4: -0.0146727, 0.0106638, -0.0165408, 0.0143703, -0.0290430, 0.0272046
5: 0.0017145, 0.0239306, -0.0001129, 0.0261420, -0.0244276, 0.0240436
6: -0.0355300, 0.0137045, -0.0407412, 0.0176798, -0.0532098, 0.0544457
7: 0.9471205, 0.9807147, 0.9375624, 0.9812785, -0.0341580, 0.0431523
8: -0.0331546, 0.0203520, -0.0362442, 0.0280903, -0.0612449, 0.0565962
9: -0.0182667, 0.0178637, -0.0232286, 0.0225250, -0.0407917, 0.0410923

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 2.18 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0148214, 0.0086762, -0.0140097, 0.0071367, -0.0219581, 0.0226860
1: -0.0109010, 0.0041122, -0.0098088, 0.0034545, -0.0143555, 0.0139210
2: 0.0158549, 0.0646695, 0.0196757, 0.0623331, -0.0464782, 0.0449939
3: -0.0045933, 0.0153899, -0.0044855, 0.0137991, -0.0183925, 0.0198755
4: -0.0168977, 0.0150786, -0.0159854, 0.0132684, -0.0301661, 0.0310640
5: -0.0004621, 0.0265646, 0.0004303, 0.0254846, -0.0259467, 0.0261342
6: -0.0417370, 0.0184394, -0.0391919, 0.0164979, -0.0582349, 0.0576313
7: 0.9357362, 0.9813864, 0.9404041, 0.9811110, -0.0453748, 0.0409823
8: -0.0368345, 0.0295690, -0.0353257, 0.0257897, -0.0626243, 0.0648946
9: -0.0241768, 0.0234156, -0.0217535, 0.0211392, -0.0453159, 0.0451691

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 2.17 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0148214, 0.0086762, -0.0144867, 0.0080415, -0.0228629, 0.0231630
1: -0.0109010, 0.0041122, -0.0104507, 0.0038410, -0.0147420, 0.0145628
2: 0.0158549, 0.0646695, 0.0174302, 0.0637062, -0.0478513, 0.0472393
3: -0.0045933, 0.0153899, -0.0045489, 0.0147340, -0.0193274, 0.0199388
4: -0.0168977, 0.0150786, -0.0165216, 0.0143322, -0.0312299, 0.0316001
5: -0.0004621, 0.0265646, -0.0000941, 0.0261193, -0.0265814, 0.0266587
6: -0.0417370, 0.0184394, -0.0406876, 0.0176389, -0.0593758, 0.0591269
7: 0.9357362, 0.9813864, 0.9376608, 0.9812729, -0.0455368, 0.0437256
8: -0.0368345, 0.0295690, -0.0362124, 0.0280107, -0.0648453, 0.0657814
9: -0.0241768, 0.0234156, -0.0231776, 0.0224770, -0.0466538, 0.0465932

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0132872, 0.0057663, -0.0128851, 0.0050036, -0.0182908, 0.0186514
1: -0.0088365, 0.0028690, -0.0082955, 0.0025432, -0.0113797, 0.0111645
2: 0.0230768, 0.0602532, 0.0249697, 0.0590957, -0.0360189, 0.0352835
3: -0.0043895, 0.0123830, -0.0043361, 0.0115950, -0.0159845, 0.0167191
4: -0.0151733, 0.0116569, -0.0147213, 0.0107602, -0.0259334, 0.0263782
5: 0.0012248, 0.0245232, 0.0016670, 0.0239881, -0.0227633, 0.0228562
6: -0.0369263, 0.0147696, -0.0356655, 0.0138078, -0.0507342, 0.0504351
7: 0.9445596, 0.9808658, 0.9468721, 0.9807293, -0.0361698, 0.0339937
8: -0.0339825, 0.0224254, -0.0332350, 0.0205532, -0.0545357, 0.0556604
9: -0.0195962, 0.0191127, -0.0183957, 0.0179849, -0.0375811, 0.0375083

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.50 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0132872, 0.0057663, -0.0134060, 0.0059916, -0.0192788, 0.0191723
1: -0.0088365, 0.0028690, -0.0089964, 0.0029653, -0.0118018, 0.0118654
2: 0.0230768, 0.0602532, 0.0225176, 0.0605952, -0.0375183, 0.0377356
3: -0.0043895, 0.0123830, -0.0044053, 0.0126159, -0.0170054, 0.0167883
4: -0.0151733, 0.0116569, -0.0153068, 0.0119219, -0.0270952, 0.0269637
5: 0.0012248, 0.0245232, 0.0010942, 0.0246813, -0.0234564, 0.0234290
6: -0.0369263, 0.0147696, -0.0372989, 0.0150538, -0.0519801, 0.0520685
7: 0.9445596, 0.9808658, 0.9438763, 0.9809060, -0.0363464, 0.0369895
8: -0.0339825, 0.0224254, -0.0342033, 0.0229786, -0.0569611, 0.0566287
9: -0.0195962, 0.0191127, -0.0199509, 0.0194459, -0.0390421, 0.0390636

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
time: 1.74 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0143816, 0.0078421, -0.0128851, 0.0050036, -0.0193852, 0.0207272
1: -0.0103092, 0.0037558, -0.0082955, 0.0025432, -0.0128523, 0.0120513
2: 0.0179252, 0.0634035, 0.0249697, 0.0590957, -0.0411705, 0.0384338
3: -0.0045349, 0.0145280, -0.0043361, 0.0115950, -0.0161299, 0.0188640
4: -0.0164034, 0.0140977, -0.0147213, 0.0107602, -0.0271636, 0.0288190
5: 0.0000215, 0.0259794, 0.0016670, 0.0239881, -0.0239667, 0.0243124
6: -0.0403579, 0.0173874, -0.0356655, 0.0138078, -0.0541658, 0.0530529
7: 0.9382654, 0.9812371, 0.9468721, 0.9807293, -0.0424639, 0.0343651
8: -0.0360169, 0.0275211, -0.0332350, 0.0205532, -0.0565701, 0.0607561
9: -0.0228637, 0.0221821, -0.0183957, 0.0179849, -0.0408485, 0.0405778

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285404, upper bound: 0.0282746
time: 1.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0143816, 0.0078421, -0.0134060, 0.0059916, -0.0203732, 0.0212481
1: -0.0103092, 0.0037558, -0.0089964, 0.0029653, -0.0132745, 0.0127522
2: 0.0179252, 0.0634035, 0.0225176, 0.0605952, -0.0426700, 0.0408859
3: -0.0045349, 0.0145280, -0.0044053, 0.0126159, -0.0171508, 0.0189333
4: -0.0164034, 0.0140977, -0.0153068, 0.0119219, -0.0283253, 0.0294045
5: 0.0000215, 0.0259794, 0.0010942, 0.0246813, -0.0246598, 0.0248852
6: -0.0403579, 0.0173874, -0.0372989, 0.0150538, -0.0554117, 0.0546862
7: 0.9382654, 0.9812371, 0.9438763, 0.9809060, -0.0426406, 0.0373608
8: -0.0360169, 0.0275211, -0.0342033, 0.0229786, -0.0589956, 0.0617245
9: -0.0228637, 0.0221821, -0.0199509, 0.0194459, -0.0423095, 0.0421331

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285404, upper bound: 0.0282746
time: 1.40 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
time: 1.36 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0133155, 0.0058200, -0.0140261, 0.0071677, -0.0204833, 0.0198461
1: -0.0088747, 0.0028920, -0.0098308, 0.0034677, -0.0123424, 0.0127228
2: 0.0229435, 0.0603347, 0.0195986, 0.0623801, -0.0394366, 0.0407360
3: -0.0043933, 0.0124385, -0.0044877, 0.0138312, -0.0182245, 0.0169262
4: -0.0152051, 0.0117201, -0.0160038, 0.0133048, -0.0285099, 0.0277239
5: 0.0011937, 0.0245608, 0.0004124, 0.0255063, -0.0243126, 0.0241485
6: -0.0370151, 0.0148374, -0.0392432, 0.0165370, -0.0535521, 0.0540806
7: 0.9443967, 0.9808753, 0.9403101, 0.9811166, -0.0367199, 0.0405652
8: -0.0340351, 0.0225573, -0.0353561, 0.0258659, -0.0599010, 0.0579134
9: -0.0196808, 0.0191921, -0.0218022, 0.0211850, -0.0408658, 0.0409943

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0133155, 0.0058200, -0.0145038, 0.0080739, -0.0213894, 0.0203238
1: -0.0088747, 0.0028920, -0.0104737, 0.0038549, -0.0127295, 0.0133656
2: 0.0229435, 0.0603347, 0.0173498, 0.0637554, -0.0408119, 0.0429849
3: -0.0043933, 0.0124385, -0.0045511, 0.0147675, -0.0191608, 0.0169897
4: -0.0152051, 0.0117201, -0.0165408, 0.0143703, -0.0295754, 0.0282609
5: 0.0011937, 0.0245608, -0.0001129, 0.0261420, -0.0249483, 0.0246738
6: -0.0370151, 0.0148374, -0.0407412, 0.0176798, -0.0546949, 0.0555786
7: 0.9443967, 0.9808753, 0.9375624, 0.9812785, -0.0368819, 0.0433129
8: -0.0340351, 0.0225573, -0.0362442, 0.0280903, -0.0621254, 0.0588015
9: -0.0196808, 0.0191921, -0.0232286, 0.0225250, -0.0422057, 0.0424207

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.27 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 2.11 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0152935, 0.0095718, -0.0140097, 0.0071367, -0.0224303, 0.0235815
1: -0.0115363, 0.0044948, -0.0098088, 0.0034545, -0.0149908, 0.0143036
2: 0.0136324, 0.0660286, 0.0196757, 0.0623331, -0.0487007, 0.0463530
3: -0.0046561, 0.0163153, -0.0044855, 0.0137991, -0.0184552, 0.0208008
4: -0.0174284, 0.0161315, -0.0159854, 0.0132684, -0.0306968, 0.0321169
5: -0.0009813, 0.0271928, 0.0004303, 0.0254846, -0.0264659, 0.0267625
6: -0.0432174, 0.0195687, -0.0391919, 0.0164979, -0.0597153, 0.0587607
7: 0.9330209, 0.9815466, 0.9404041, 0.9811110, -0.0480901, 0.0411425
8: -0.0377122, 0.0317672, -0.0353257, 0.0257897, -0.0635020, 0.0670929
9: -0.0255864, 0.0247398, -0.0217535, 0.0211392, -0.0467256, 0.0464933

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.28 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0152935, 0.0095718, -0.0144867, 0.0080415, -0.0233350, 0.0240585
1: -0.0115363, 0.0044948, -0.0104507, 0.0038410, -0.0153773, 0.0149454
2: 0.0136324, 0.0660286, 0.0174302, 0.0637062, -0.0500738, 0.0485984
3: -0.0046561, 0.0163153, -0.0045489, 0.0147340, -0.0193901, 0.0208642
4: -0.0174284, 0.0161315, -0.0165216, 0.0143322, -0.0317606, 0.0326531
5: -0.0009813, 0.0271928, -0.0000941, 0.0261193, -0.0271006, 0.0272870
6: -0.0432174, 0.0195687, -0.0406876, 0.0176389, -0.0608563, 0.0602563
7: 0.9330209, 0.9815466, 0.9376608, 0.9812729, -0.0482520, 0.0438858
8: -0.0377122, 0.0317672, -0.0362124, 0.0280107, -0.0657230, 0.0679796
9: -0.0255864, 0.0247398, -0.0231776, 0.0224770, -0.0480634, 0.0479174

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

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

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

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

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.29 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0139148, 0.0069566, -0.0133033, 0.0057968, -0.0197116, 0.0202599
1: -0.0096810, 0.0033775, -0.0088582, 0.0028821, -0.0125631, 0.0122357
2: 0.0201227, 0.0620596, 0.0230010, 0.0602995, -0.0401768, 0.0390586
3: -0.0044729, 0.0136130, -0.0043916, 0.0124146, -0.0168875, 0.0180046
4: -0.0158786, 0.0130565, -0.0151914, 0.0116929, -0.0275715, 0.0282479
5: 0.0005348, 0.0253582, 0.0012071, 0.0245446, -0.0240098, 0.0241511
6: -0.0388941, 0.0162707, -0.0369768, 0.0148082, -0.0537023, 0.0532475
7: 0.9409504, 0.9810788, 0.9444669, 0.9808712, -0.0399208, 0.0366119
8: -0.0351491, 0.0253475, -0.0340124, 0.0225004, -0.0576495, 0.0593599
9: -0.0214699, 0.0208728, -0.0196443, 0.0191578, -0.0406277, 0.0405170

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0138900, 0.0069096, -0.0152825, 0.0095509, -0.0234409, 0.0221922
1: -0.0096477, 0.0033575, -0.0115215, 0.0044859, -0.0141335, 0.0148790
2: 0.0202393, 0.0619884, 0.0136841, 0.0659970, -0.0457577, 0.0483043
3: -0.0044696, 0.0135645, -0.0046546, 0.0162938, -0.0207634, 0.0182191
4: -0.0158508, 0.0130013, -0.0174161, 0.0161070, -0.0319578, 0.0304174
5: 0.0005620, 0.0253253, -0.0009692, 0.0271782, -0.0266162, 0.0262945
6: -0.0388165, 0.0162115, -0.0431829, 0.0195424, -0.0583589, 0.0593944
7: 0.9410927, 0.9810704, 0.9330840, 0.9815429, -0.0404502, 0.0479864
8: -0.0351030, 0.0252322, -0.0376918, 0.0317161, -0.0668192, 0.0629240
9: -0.0213959, 0.0208033, -0.0255536, 0.0247090, -0.0461050, 0.0463569

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
time: 1.34 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0128419, 0.0049216, -0.0143982, 0.0078736, -0.0207155, 0.0193199
1: -0.0082373, 0.0025082, -0.0103316, 0.0037693, -0.0120066, 0.0128397
2: 0.0251730, 0.0589713, 0.0178469, 0.0634514, -0.0382784, 0.0411244
3: -0.0043303, 0.0115103, -0.0045371, 0.0145606, -0.0188909, 0.0160474
4: -0.0146727, 0.0106638, -0.0164221, 0.0141348, -0.0288075, 0.0270859
5: 0.0017145, 0.0239306, 0.0000032, 0.0260015, -0.0242871, 0.0239275
6: -0.0355300, 0.0137045, -0.0404101, 0.0174272, -0.0529572, 0.0541146
7: 0.9471205, 0.9807147, 0.9381698, 0.9812428, -0.0341223, 0.0425450
8: -0.0331546, 0.0203520, -0.0360479, 0.0275986, -0.0607533, 0.0563999
9: -0.0182667, 0.0178637, -0.0229134, 0.0222288, -0.0404955, 0.0407770

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 2.19 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0128419, 0.0049216, -0.0163739, 0.0116209, -0.0244628, 0.0212955
1: -0.0082373, 0.0025082, -0.0129900, 0.0053702, -0.0136075, 0.0154982
2: 0.0251730, 0.0589713, 0.0085469, 0.0691385, -0.0439654, 0.0504245
3: -0.0043303, 0.0115103, -0.0047996, 0.0184327, -0.0227630, 0.0163099
4: -0.0146727, 0.0106638, -0.0186427, 0.0185410, -0.0332137, 0.0293066
5: 0.0017145, 0.0239306, -0.0021692, 0.0286303, -0.0269159, 0.0260998
6: -0.0355300, 0.0137045, -0.0466049, 0.0221528, -0.0576829, 0.0603094
7: 0.9471205, 0.9807147, 0.9268076, 0.9819133, -0.0347927, 0.0539072
8: -0.0331546, 0.0203520, -0.0397206, 0.0367976, -0.0699522, 0.0600726
9: -0.0182667, 0.0178637, -0.0288119, 0.0277699, -0.0460366, 0.0466756

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0148214, 0.0086762, -0.0138900, 0.0069096, -0.0217310, 0.0225662
1: -0.0109010, 0.0041122, -0.0096477, 0.0033575, -0.0142584, 0.0137599
2: 0.0158549, 0.0646695, 0.0202393, 0.0619884, -0.0461335, 0.0444302
3: -0.0045933, 0.0153899, -0.0044696, 0.0135645, -0.0181578, 0.0198595
4: -0.0168977, 0.0150786, -0.0158508, 0.0130013, -0.0298990, 0.0309294
5: -0.0004621, 0.0265646, 0.0005620, 0.0253253, -0.0257874, 0.0260026
6: -0.0417370, 0.0184394, -0.0388165, 0.0162115, -0.0579484, 0.0572558
7: 0.9357362, 0.9813864, 0.9410927, 0.9810704, -0.0453342, 0.0402936
8: -0.0368345, 0.0295690, -0.0351030, 0.0252322, -0.0620667, 0.0646720
9: -0.0241768, 0.0234156, -0.0213959, 0.0208033, -0.0449801, 0.0448116

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.21 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0148214, 0.0086762, -0.0143569, 0.0077952, -0.0226166, 0.0230331
1: -0.0109010, 0.0041122, -0.0102759, 0.0037358, -0.0146368, 0.0143881
2: 0.0158549, 0.0646695, 0.0180414, 0.0633324, -0.0474775, 0.0466281
3: -0.0045933, 0.0153899, -0.0045316, 0.0144796, -0.0190729, 0.0199216
4: -0.0168977, 0.0150786, -0.0163756, 0.0140426, -0.0309403, 0.0314542
5: -0.0004621, 0.0265646, 0.0000486, 0.0259465, -0.0264086, 0.0265160
6: -0.0417370, 0.0184394, -0.0402805, 0.0173283, -0.0590653, 0.0587198
7: 0.9357362, 0.9813864, 0.9384076, 0.9812287, -0.0454925, 0.0429788
8: -0.0368345, 0.0295690, -0.0359710, 0.0274061, -0.0642407, 0.0655400
9: -0.0241768, 0.0234156, -0.0227899, 0.0221129, -0.0462896, 0.0462056

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0132872, 0.0057663, -0.0128112, 0.0048635, -0.0181507, 0.0185775
1: -0.0088365, 0.0028690, -0.0081961, 0.0024833, -0.0113199, 0.0110651
2: 0.0230768, 0.0602532, 0.0253174, 0.0588830, -0.0358062, 0.0349358
3: -0.0043895, 0.0123830, -0.0043263, 0.0114502, -0.0158397, 0.0167093
4: -0.0151733, 0.0116569, -0.0146383, 0.0105954, -0.0257687, 0.0262952
5: 0.0012248, 0.0245232, 0.0017482, 0.0238898, -0.0226650, 0.0227750
6: -0.0369263, 0.0147696, -0.0354339, 0.0136311, -0.0505575, 0.0502035
7: 0.9445596, 0.9808658, 0.9472969, 0.9807042, -0.0361447, 0.0335689
8: -0.0339825, 0.0224254, -0.0330976, 0.0202093, -0.0541917, 0.0555231
9: -0.0195962, 0.0191127, -0.0181751, 0.0177777, -0.0373739, 0.0372878

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.11 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0132872, 0.0057663, -0.0132872, 0.0057663, -0.0190535, 0.0190535
1: -0.0088365, 0.0028690, -0.0088365, 0.0028690, -0.0117055, 0.0117055
2: 0.0230768, 0.0602532, 0.0230768, 0.0602532, -0.0371763, 0.0371763
3: -0.0043895, 0.0123830, -0.0043895, 0.0123830, -0.0167725, 0.0167725
4: -0.0151733, 0.0116569, -0.0151733, 0.0116569, -0.0268302, 0.0268302
5: 0.0012248, 0.0245232, 0.0012248, 0.0245232, -0.0232983, 0.0232983
6: -0.0369263, 0.0147696, -0.0369263, 0.0147696, -0.0516960, 0.0516960
7: 0.9445596, 0.9808658, 0.9445596, 0.9808658, -0.0363062, 0.0363062
8: -0.0339825, 0.0224254, -0.0339825, 0.0224254, -0.0564079, 0.0564079
9: -0.0195962, 0.0191127, -0.0195962, 0.0191127, -0.0387089, 0.0387089

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.18 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0143816, 0.0078421, -0.0128112, 0.0048635, -0.0192451, 0.0206533
1: -0.0103092, 0.0037558, -0.0081961, 0.0024833, -0.0127925, 0.0119519
2: 0.0179252, 0.0634035, 0.0253174, 0.0588830, -0.0409579, 0.0380861
3: -0.0045349, 0.0145280, -0.0043263, 0.0114502, -0.0159851, 0.0188542
4: -0.0164034, 0.0140977, -0.0146383, 0.0105954, -0.0269988, 0.0287360
5: 0.0000215, 0.0259794, 0.0017482, 0.0238898, -0.0238684, 0.0242312
6: -0.0403579, 0.0173874, -0.0354339, 0.0136311, -0.0539891, 0.0528213
7: 0.9382654, 0.9812371, 0.9472969, 0.9807042, -0.0424388, 0.0339402
8: -0.0360169, 0.0275211, -0.0330976, 0.0202093, -0.0562262, 0.0606188
9: -0.0228637, 0.0221821, -0.0181751, 0.0177777, -0.0406413, 0.0403573

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285404, upper bound: 0.0282746
time: 2.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0143816, 0.0078421, -0.0132872, 0.0057663, -0.0201479, 0.0211292
1: -0.0103092, 0.0037558, -0.0088365, 0.0028690, -0.0131782, 0.0125924
2: 0.0179252, 0.0634035, 0.0230768, 0.0602532, -0.0423280, 0.0403267
3: -0.0045349, 0.0145280, -0.0043895, 0.0123830, -0.0169180, 0.0189175
4: -0.0164034, 0.0140977, -0.0151733, 0.0116569, -0.0280603, 0.0292710
5: 0.0000215, 0.0259794, 0.0012248, 0.0245232, -0.0245017, 0.0247546
6: -0.0403579, 0.0173874, -0.0369263, 0.0147696, -0.0551276, 0.0543137
7: 0.9382654, 0.9812371, 0.9445596, 0.9808658, -0.0426003, 0.0366775
8: -0.0360169, 0.0275211, -0.0339825, 0.0224254, -0.0584424, 0.0615036
9: -0.0228637, 0.0221821, -0.0195962, 0.0191127, -0.0419763, 0.0417783

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285404, upper bound: 0.0282746
time: 1.49 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
time: 1.35 seconds

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

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.15 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.07 seconds

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

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 2.05 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
time: 1.22 seconds

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

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 61

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.36 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
time: 1.20 seconds

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

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

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
time: 1.14 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.33 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0290794, upper bound: 0.0274471
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282359, upper bound: 0.0274471
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0290794, upper bound: 0.0274471
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282359, upper bound: 0.0274471
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274471
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274471
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274471
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274471
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0290693, upper bound: 0.0274104
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274104
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274104
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0289034, upper bound: 0.0274104
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274104
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274471
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274471
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0270152, upper bound: 0.0265075
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0265075
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0269805, upper bound: 0.0265075
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0264988, upper bound: 0.0265075
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0270152, upper bound: 0.0264988
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0264988
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0269805, upper bound: 0.0264988
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0264988, upper bound: 0.0264988
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0291232, upper bound: 0.0274471
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282613, upper bound: 0.0274471
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0291232, upper bound: 0.0274471
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282613, upper bound: 0.0274471
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0289450, upper bound: 0.0274471
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282008, upper bound: 0.0274471
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0289450, upper bound: 0.0274471
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282008, upper bound: 0.0274471
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0291114, upper bound: 0.0274104
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282607, upper bound: 0.0274104
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0291114, upper bound: 0.0274104
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282607, upper bound: 0.0274104
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0289450, upper bound: 0.0274104
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282008, upper bound: 0.0274104
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0289450, upper bound: 0.0274104
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0282008, upper bound: 0.0274104
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274471
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274471
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0270328, upper bound: 0.0265131
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0265131
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274471
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0269973, upper bound: 0.0265131
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0264988, upper bound: 0.0265131
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0270328, upper bound: 0.0265043
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0265075, upper bound: 0.0265043
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0269973, upper bound: 0.0265043
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0264988, upper bound: 0.0265043
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0296077, upper bound: 0.0283447
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0296077, upper bound: 0.0283447
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0286579, upper bound: 0.0282716
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0286579, upper bound: 0.0282716
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285404, upper bound: 0.0282746
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285404, upper bound: 0.0282746
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0282008
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0286323, upper bound: 0.0274104
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274471, upper bound: 0.0274104
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0294413, upper bound: 0.0283447
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285404, upper bound: 0.0282746
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285404, upper bound: 0.0282746
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0282117
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0285016, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.33
Output dim: 7, lower bound: -0.0274104, upper bound: 0.0274104

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0128634, 0.0049623, -0.0128851, 0.0050036, -0.0178670, 0.0178474
1: -0.0082662, 0.0025256, -0.0082955, 0.0025432, -0.0108094, 0.0108210
2: 0.0250721, 0.0590331, 0.0249697, 0.0590957, -0.0340236, 0.0340634
3: -0.0043332, 0.0115523, -0.0043361, 0.0115950, -0.0159281, 0.0158884
4: -0.0146968, 0.0107116, -0.0147213, 0.0107602, -0.0254570, 0.0254329
5: 0.0016909, 0.0239592, 0.0016670, 0.0239881, -0.0222972, 0.0222922
6: -0.0355973, 0.0137558, -0.0356655, 0.0138078, -0.0494051, 0.0494213
7: 0.9469972, 0.9807220, 0.9468721, 0.9807293, -0.0337322, 0.0338499
8: -0.0331945, 0.0204519, -0.0332350, 0.0205532, -0.0537477, 0.0536869
9: -0.0183307, 0.0179238, -0.0183957, 0.0179849, -0.0363156, 0.0363195

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283858
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283858
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0148500, 0.0087305, -0.0128688, 0.0049727, -0.0198227, 0.0215993
1: -0.0109395, 0.0041354, -0.0082735, 0.0025300, -0.0134694, 0.0124089
2: 0.0157203, 0.0647518, 0.0250464, 0.0590487, -0.0433285, 0.0397054
3: -0.0045971, 0.0154460, -0.0043339, 0.0115630, -0.0161601, 0.0197799
4: -0.0169299, 0.0151423, -0.0147030, 0.0107238, -0.0276537, 0.0298453
5: -0.0004936, 0.0266026, 0.0016849, 0.0239664, -0.0244600, 0.0249177
6: -0.0418266, 0.0185078, -0.0356143, 0.0137688, -0.0555954, 0.0541221
7: 0.9355717, 0.9813961, 0.9469659, 0.9807239, -0.0451522, 0.0344302
8: -0.0368877, 0.0297021, -0.0332047, 0.0204773, -0.0573650, 0.0629067
9: -0.0242621, 0.0234958, -0.0183470, 0.0179391, -0.0422013, 0.0418428

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273865, upper bound: 0.0279552
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273865, upper bound: 0.0273865
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128634, 0.0049623, -0.0140261, 0.0071677, -0.0200311, 0.0189884
1: -0.0082662, 0.0025256, -0.0098308, 0.0034677, -0.0117339, 0.0123564
2: 0.0250721, 0.0590331, 0.0195986, 0.0623801, -0.0373080, 0.0394344
3: -0.0043332, 0.0115523, -0.0044877, 0.0138312, -0.0181644, 0.0160400
4: -0.0146968, 0.0107116, -0.0160038, 0.0133048, -0.0280017, 0.0267154
5: 0.0016909, 0.0239592, 0.0004124, 0.0255063, -0.0238154, 0.0235468
6: -0.0355973, 0.0137558, -0.0392432, 0.0165370, -0.0521343, 0.0529990
7: 0.9469972, 0.9807220, 0.9403101, 0.9811166, -0.0341194, 0.0404118
8: -0.0331945, 0.0204519, -0.0353561, 0.0258659, -0.0590604, 0.0558080
9: -0.0183307, 0.0179238, -0.0218022, 0.0211850, -0.0395157, 0.0397261

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282359, upper bound: 0.0274471
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282359, upper bound: 0.0274471
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0148500, 0.0087305, -0.0140097, 0.0071367, -0.0219867, 0.0227402
1: -0.0109395, 0.0041354, -0.0098088, 0.0034545, -0.0143940, 0.0139442
2: 0.0157203, 0.0647518, 0.0196757, 0.0623331, -0.0466128, 0.0450762
3: -0.0045971, 0.0154460, -0.0044855, 0.0137991, -0.0183963, 0.0199315
4: -0.0169299, 0.0151423, -0.0159854, 0.0132684, -0.0301982, 0.0311277
5: -0.0004936, 0.0266026, 0.0004303, 0.0254846, -0.0259782, 0.0261723
6: -0.0418266, 0.0185078, -0.0391919, 0.0164979, -0.0583245, 0.0576997
7: 0.9355717, 0.9813961, 0.9404041, 0.9811110, -0.0455393, 0.0409921
8: -0.0368877, 0.0297021, -0.0353257, 0.0257897, -0.0626774, 0.0650277
9: -0.0242621, 0.0234958, -0.0217535, 0.0211392, -0.0454013, 0.0452493

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0272891, upper bound: 0.0270320
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0272161, upper bound: 0.0265075
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0133834, 0.0059487, -0.0128851, 0.0050036, -0.0183870, 0.0188338
1: -0.0089659, 0.0029469, -0.0082955, 0.0025432, -0.0115091, 0.0112424
2: 0.0226242, 0.0605300, 0.0249697, 0.0590957, -0.0364715, 0.0355603
3: -0.0044023, 0.0125715, -0.0043361, 0.0115950, -0.0159972, 0.0169076
4: -0.0152814, 0.0118714, -0.0147213, 0.0107602, -0.0260415, 0.0265927
5: 0.0011191, 0.0246511, 0.0016670, 0.0239881, -0.0228690, 0.0229842
6: -0.0372279, 0.0149997, -0.0356655, 0.0138078, -0.0510357, 0.0506652
7: 0.9440065, 0.9808984, 0.9468721, 0.9807293, -0.0367228, 0.0340264
8: -0.0341612, 0.0228733, -0.0332350, 0.0205532, -0.0547144, 0.0561082
9: -0.0198833, 0.0193824, -0.0183957, 0.0179849, -0.0378682, 0.0377781

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283858
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283858
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0153304, 0.0096417, -0.0128688, 0.0049727, -0.0203031, 0.0225105
1: -0.0115859, 0.0045246, -0.0082735, 0.0025300, -0.0141159, 0.0127982
2: 0.0134588, 0.0661348, 0.0250464, 0.0590487, -0.0455900, 0.0410883
3: -0.0046610, 0.0163876, -0.0043339, 0.0115630, -0.0162240, 0.0207215
4: -0.0174699, 0.0162138, -0.0147030, 0.0107238, -0.0281937, 0.0309167
5: -0.0010218, 0.0272419, 0.0016849, 0.0239664, -0.0249883, 0.0255570
6: -0.0433330, 0.0196569, -0.0356143, 0.0137688, -0.0571019, 0.0552713
7: 0.9328088, 0.9815592, 0.9469659, 0.9807239, -0.0479151, 0.0345933
8: -0.0377808, 0.0319390, -0.0332047, 0.0204773, -0.0582581, 0.0651437
9: -0.0256965, 0.0248433, -0.0183470, 0.0179391, -0.0436356, 0.0431903

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273688, upper bound: 0.0279550
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273688, upper bound: 0.0273865
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0133834, 0.0059487, -0.0140261, 0.0071677, -0.0205511, 0.0199748
1: -0.0089659, 0.0029469, -0.0098308, 0.0034677, -0.0124337, 0.0127777
2: 0.0226242, 0.0605300, 0.0195986, 0.0623801, -0.0397559, 0.0409314
3: -0.0044023, 0.0125715, -0.0044877, 0.0138312, -0.0182335, 0.0170592
4: -0.0152814, 0.0118714, -0.0160038, 0.0133048, -0.0285862, 0.0278752
5: 0.0011191, 0.0246511, 0.0004124, 0.0255063, -0.0243872, 0.0242387
6: -0.0372279, 0.0149997, -0.0392432, 0.0165370, -0.0537649, 0.0542428
7: 0.9440065, 0.9808984, 0.9403101, 0.9811166, -0.0371101, 0.0405883
8: -0.0341612, 0.0228733, -0.0353561, 0.0258659, -0.0600271, 0.0582293
9: -0.0198833, 0.0193824, -0.0218022, 0.0211850, -0.0410683, 0.0411846

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274471
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274471
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0153304, 0.0096417, -0.0140097, 0.0071367, -0.0224671, 0.0236515
1: -0.0115859, 0.0045246, -0.0098088, 0.0034545, -0.0150404, 0.0143334
2: 0.0134588, 0.0661348, 0.0196757, 0.0623331, -0.0488743, 0.0464591
3: -0.0046610, 0.0163876, -0.0044855, 0.0137991, -0.0184601, 0.0208731
4: -0.0174699, 0.0162138, -0.0159854, 0.0132684, -0.0307382, 0.0321992
5: -0.0010218, 0.0272419, 0.0004303, 0.0254846, -0.0265064, 0.0268116
6: -0.0433330, 0.0196569, -0.0391919, 0.0164979, -0.0598310, 0.0588489
7: 0.9328088, 0.9815592, 0.9404041, 0.9811110, -0.0483022, 0.0411552
8: -0.0377808, 0.0319390, -0.0353257, 0.0257897, -0.0635705, 0.0672647
9: -0.0256965, 0.0248433, -0.0217535, 0.0211392, -0.0468357, 0.0465968

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0272692, upper bound: 0.0270318
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0271855, upper bound: 0.0265075
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0128634, 0.0049623, -0.0134060, 0.0059916, -0.0188550, 0.0183683
1: -0.0082662, 0.0025256, -0.0089964, 0.0029653, -0.0112315, 0.0115220
2: 0.0250721, 0.0590331, 0.0225176, 0.0605952, -0.0355231, 0.0365155
3: -0.0043332, 0.0115523, -0.0044053, 0.0126159, -0.0169491, 0.0159576
4: -0.0146968, 0.0107116, -0.0153068, 0.0119219, -0.0266188, 0.0260184
5: 0.0016909, 0.0239592, 0.0010942, 0.0246813, -0.0229904, 0.0228650
6: -0.0355973, 0.0137558, -0.0372989, 0.0150538, -0.0506511, 0.0510547
7: 0.9469972, 0.9807220, 0.9438763, 0.9809060, -0.0339088, 0.0368457
8: -0.0331945, 0.0204519, -0.0342033, 0.0229786, -0.0561732, 0.0546552
9: -0.0183307, 0.0179238, -0.0199509, 0.0194459, -0.0377766, 0.0378748

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283858, upper bound: 0.0283447
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0148500, 0.0087305, -0.0133891, 0.0059596, -0.0208095, 0.0221196
1: -0.0109395, 0.0041354, -0.0089737, 0.0029516, -0.0138910, 0.0131090
2: 0.0157203, 0.0647518, 0.0225972, 0.0605465, -0.0448262, 0.0421547
3: -0.0045971, 0.0154460, -0.0044030, 0.0125828, -0.0171799, 0.0198490
4: -0.0169299, 0.0151423, -0.0152878, 0.0118842, -0.0288141, 0.0304301
5: -0.0004936, 0.0266026, 0.0011128, 0.0246588, -0.0251523, 0.0254899
6: -0.0418266, 0.0185078, -0.0372459, 0.0150134, -0.0568400, 0.0557536
7: 0.9355717, 0.9813961, 0.9439735, 0.9809004, -0.0453287, 0.0374227
8: -0.0368877, 0.0297021, -0.0341719, 0.0228999, -0.0597876, 0.0638740
9: -0.0242621, 0.0234958, -0.0199005, 0.0193985, -0.0436606, 0.0433963

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273865, upper bound: 0.0279192
time: 1.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273865, upper bound: 0.0273688
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0128634, 0.0049623, -0.0145038, 0.0080739, -0.0209373, 0.0194662
1: -0.0082662, 0.0025256, -0.0104737, 0.0038549, -0.0121211, 0.0129992
2: 0.0250721, 0.0590331, 0.0173498, 0.0637554, -0.0386833, 0.0416833
3: -0.0043332, 0.0115523, -0.0045511, 0.0147675, -0.0191007, 0.0161034
4: -0.0146968, 0.0107116, -0.0165408, 0.0143703, -0.0290672, 0.0272524
5: 0.0016909, 0.0239592, -0.0001129, 0.0261420, -0.0244512, 0.0240721
6: -0.0355973, 0.0137558, -0.0407412, 0.0176798, -0.0532770, 0.0544970
7: 0.9469972, 0.9807220, 0.9375624, 0.9812785, -0.0342814, 0.0431595
8: -0.0331945, 0.0204519, -0.0362442, 0.0280903, -0.0612848, 0.0566961
9: -0.0183307, 0.0179238, -0.0232286, 0.0225250, -0.0408557, 0.0411525

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0282347, upper bound: 0.0274104
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0148500, 0.0087305, -0.0144867, 0.0080415, -0.0228915, 0.0232172
1: -0.0109395, 0.0041354, -0.0104507, 0.0038410, -0.0147805, 0.0145860
2: 0.0157203, 0.0647518, 0.0174302, 0.0637062, -0.0479859, 0.0473216
3: -0.0045971, 0.0154460, -0.0045489, 0.0147340, -0.0193312, 0.0199949
4: -0.0169299, 0.0151423, -0.0165216, 0.0143322, -0.0312621, 0.0316639
5: -0.0004936, 0.0266026, -0.0000941, 0.0261193, -0.0266129, 0.0266968
6: -0.0418266, 0.0185078, -0.0406876, 0.0176389, -0.0594655, 0.0591953
7: 0.9355717, 0.9813961, 0.9376608, 0.9812729, -0.0457013, 0.0437353
8: -0.0368877, 0.0297021, -0.0362124, 0.0280107, -0.0648984, 0.0659145
9: -0.0242621, 0.0234958, -0.0231776, 0.0224770, -0.0467391, 0.0466734

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0272940, upper bound: 0.0270011
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0272261, upper bound: 0.0264988
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0133834, 0.0059487, -0.0134060, 0.0059916, -0.0193750, 0.0193547
1: -0.0089659, 0.0029469, -0.0089964, 0.0029653, -0.0119312, 0.0119433
2: 0.0226242, 0.0605300, 0.0225176, 0.0605952, -0.0379710, 0.0380124
3: -0.0044023, 0.0125715, -0.0044053, 0.0126159, -0.0170182, 0.0169768
4: -0.0152814, 0.0118714, -0.0153068, 0.0119219, -0.0272033, 0.0271782
5: 0.0011191, 0.0246511, 0.0010942, 0.0246813, -0.0235622, 0.0235569
6: -0.0372279, 0.0149997, -0.0372989, 0.0150538, -0.0522817, 0.0522985
7: 0.9440065, 0.9808984, 0.9438763, 0.9809060, -0.0368995, 0.0370222
8: -0.0341612, 0.0228733, -0.0342033, 0.0229786, -0.0571399, 0.0570766
9: -0.0198833, 0.0193824, -0.0199509, 0.0194459, -0.0393292, 0.0393333

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.33 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0283447, upper bound: 0.0283447
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0153304, 0.0096417, -0.0133891, 0.0059596, -0.0212900, 0.0230308
1: -0.0115859, 0.0045246, -0.0089737, 0.0029516, -0.0145375, 0.0134983
2: 0.0134588, 0.0661348, 0.0225972, 0.0605465, -0.0470877, 0.0435376
3: -0.0046610, 0.0163876, -0.0044030, 0.0125828, -0.0172437, 0.0207906
4: -0.0174699, 0.0162138, -0.0152878, 0.0118842, -0.0293541, 0.0315016
5: -0.0010218, 0.0272419, 0.0011128, 0.0246588, -0.0256806, 0.0261291
6: -0.0433330, 0.0196569, -0.0372459, 0.0150134, -0.0583465, 0.0569028
7: 0.9328088, 0.9815592, 0.9439735, 0.9809004, -0.0480917, 0.0375857
8: -0.0377808, 0.0319390, -0.0341719, 0.0228999, -0.0606807, 0.0661109
9: -0.0256965, 0.0248433, -0.0199005, 0.0193985, -0.0450950, 0.0447438

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273688, upper bound: 0.0279253
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0273688, upper bound: 0.0273688
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0133834, 0.0059487, -0.0145038, 0.0080739, -0.0214573, 0.0204525
1: -0.0089659, 0.0029469, -0.0104737, 0.0038549, -0.0128208, 0.0134206
2: 0.0226242, 0.0605300, 0.0173498, 0.0637554, -0.0411312, 0.0431802
3: -0.0044023, 0.0125715, -0.0045511, 0.0147675, -0.0191698, 0.0171227
4: -0.0152814, 0.0118714, -0.0165408, 0.0143703, -0.0296517, 0.0284122
5: 0.0011191, 0.0246511, -0.0001129, 0.0261420, -0.0250230, 0.0247641
6: -0.0372279, 0.0149997, -0.0407412, 0.0176798, -0.0549076, 0.0557408
7: 0.9440065, 0.9808984, 0.9375624, 0.9812785, -0.0372720, 0.0433360
8: -0.0341612, 0.0228733, -0.0362442, 0.0280903, -0.0622515, 0.0591174
9: -0.0198833, 0.0193824, -0.0232286, 0.0225250, -0.0424083, 0.0426110

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274104
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0281711, upper bound: 0.0274104
time: 1.32 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.64 + 598.19 = 601.84 seconds
