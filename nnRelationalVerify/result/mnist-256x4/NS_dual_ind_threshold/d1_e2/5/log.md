## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0010952


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0017451, 0.0017451)
1: (-0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004348, 0.0004348)
2: (0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0023044, 0.0023044)
3: (-0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0010489, 0.0010489)
4: (0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004460, 0.0004460)
5: (0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0028984, 0.0028984)
6: (-0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0007356, 0.0007356)
7: (-0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0019033, 0.0019033)
8: (-0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0010009, 0.0010009)
9: (-0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0011606, 0.0011606)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 1.94 = 3.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0013690, upper bound: 0.0013690

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012844, upper bound: 0.0012364
time: 1.45 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012844, upper bound: 0.0012844
time: 1.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.67 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.67
Output dim: 0, lower bound: -0.0012844, upper bound: 0.0012364
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.67
Output dim: 0, lower bound: -0.0012844, upper bound: 0.0012844

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.9906998, 0.9930281, 0.9906954, 0.9931488, -0.0017338, 0.0015790
1: -0.0035813, -0.0030012, -0.0035824, -0.0029711, -0.0004320, 0.0003934
2: 0.0058506, 0.0089251, 0.0056912, 0.0089309, -0.0020850, 0.0022894
3: -0.0053354, -0.0039361, -0.0053381, -0.0038635, -0.0010421, 0.0009490
4: 0.0016603, 0.0022553, 0.0016294, 0.0022564, -0.0004036, 0.0004431
5: 0.0063180, 0.0101849, 0.0061174, 0.0101921, -0.0026224, 0.0028795
6: -0.0010442, -0.0000627, -0.0010460, -0.0000118, -0.0007308, 0.0006656
7: -0.0058393, -0.0033000, -0.0058441, -0.0031683, -0.0018909, 0.0017221
8: -0.0026350, -0.0012996, -0.0026375, -0.0012303, -0.0009944, 0.0009056
9: -0.0003569, 0.0011915, -0.0004373, 0.0011944, -0.0010501, 0.0011531

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012364, upper bound: 0.0012364
time: 1.11 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012364, upper bound: 0.0012364
time: 1.69 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.9905502, 0.9930212, 0.9906967, 0.9931209, -0.0019875, 0.0016482
1: -0.0036186, -0.0030029, -0.0035821, -0.0029780, -0.0004952, 0.0004107
2: 0.0058597, 0.0091227, 0.0057281, 0.0089291, -0.0021765, 0.0026244
3: -0.0054254, -0.0039402, -0.0053373, -0.0038803, -0.0011945, 0.0009906
4: 0.0016620, 0.0022936, 0.0016365, 0.0022561, -0.0004213, 0.0005080
5: 0.0063294, 0.0104335, 0.0061639, 0.0101900, -0.0027374, 0.0033009
6: -0.0011073, -0.0000656, -0.0010455, -0.0000236, -0.0008378, 0.0006948
7: -0.0060025, -0.0033074, -0.0058426, -0.0031987, -0.0021676, 0.0017976
8: -0.0027208, -0.0013035, -0.0026367, -0.0012463, -0.0011399, 0.0009454
9: -0.0003524, 0.0012911, -0.0004187, 0.0011936, -0.0010962, 0.0013218

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012364, upper bound: 0.0012844
time: 1.09 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012364, upper bound: 0.0012844
time: 1.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.58 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 0, lower bound: -0.0012364, upper bound: 0.0012364
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 0, lower bound: -0.0012364, upper bound: 0.0012364
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 0, lower bound: -0.0012364, upper bound: 0.0012844
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 0, lower bound: -0.0012364, upper bound: 0.0012844

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.9906998, 0.9930281, 0.9906998, 0.9930281, -0.0015676, 0.0015676
1: -0.0035813, -0.0030012, -0.0035813, -0.0030012, -0.0003906, 0.0003906
2: 0.0058506, 0.0089251, 0.0058506, 0.0089251, -0.0020700, 0.0020700
3: -0.0053354, -0.0039361, -0.0053354, -0.0039361, -0.0009422, 0.0009422
4: 0.0016603, 0.0022553, 0.0016603, 0.0022553, -0.0004007, 0.0004007
5: 0.0063180, 0.0101849, 0.0063180, 0.0101849, -0.0026036, 0.0026036
6: -0.0010442, -0.0000627, -0.0010442, -0.0000627, -0.0006608, 0.0006608
7: -0.0058393, -0.0033000, -0.0058393, -0.0033000, -0.0017097, 0.0017097
8: -0.0026350, -0.0012996, -0.0026350, -0.0012996, -0.0008991, 0.0008991
9: -0.0003569, 0.0011915, -0.0003569, 0.0011915, -0.0010426, 0.0010426

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011714, upper bound: 0.0011496
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011868, upper bound: 0.0011834
time: 1.21 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.9906998, 0.9930281, 0.9905502, 0.9930212, -0.0016754, 0.0018346
1: -0.0035813, -0.0030012, -0.0036186, -0.0030029, -0.0004175, 0.0004571
2: 0.0058506, 0.0089251, 0.0058597, 0.0091227, -0.0024225, 0.0022123
3: -0.0053354, -0.0039361, -0.0054254, -0.0039402, -0.0010070, 0.0011026
4: 0.0016603, 0.0022553, 0.0016620, 0.0022936, -0.0004689, 0.0004282
5: 0.0063180, 0.0101849, 0.0063294, 0.0104335, -0.0030469, 0.0027825
6: -0.0010442, -0.0000627, -0.0011073, -0.0000656, -0.0007062, 0.0007733
7: -0.0058393, -0.0033000, -0.0060025, -0.0033074, -0.0018272, 0.0020009
8: -0.0026350, -0.0012996, -0.0027208, -0.0013035, -0.0009609, 0.0010522
9: -0.0003569, 0.0011915, -0.0003524, 0.0012911, -0.0012201, 0.0011142

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011714, upper bound: 0.0011496
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011868, upper bound: 0.0011834
time: 1.36 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.9905502, 0.9930212, 0.9906998, 0.9930281, -0.0018346, 0.0016754
1: -0.0036186, -0.0030029, -0.0035813, -0.0030012, -0.0004571, 0.0004175
2: 0.0058597, 0.0091227, 0.0058506, 0.0089251, -0.0022123, 0.0024225
3: -0.0054254, -0.0039402, -0.0053354, -0.0039361, -0.0011026, 0.0010070
4: 0.0016620, 0.0022936, 0.0016603, 0.0022553, -0.0004282, 0.0004689
5: 0.0063294, 0.0104335, 0.0063180, 0.0101849, -0.0027825, 0.0030469
6: -0.0011073, -0.0000656, -0.0010442, -0.0000627, -0.0007733, 0.0007062
7: -0.0060025, -0.0033074, -0.0058393, -0.0033000, -0.0020009, 0.0018272
8: -0.0027208, -0.0013035, -0.0026350, -0.0012996, -0.0010522, 0.0009609
9: -0.0003524, 0.0012911, -0.0003569, 0.0011915, -0.0011142, 0.0012201

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011679, upper bound: 0.0011938
time: 1.45 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0012315
time: 1.21 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.9905502, 0.9930212, 0.9905502, 0.9930212, -0.0016379, 0.0016379
1: -0.0036186, -0.0030029, -0.0036186, -0.0030029, -0.0004081, 0.0004081
2: 0.0058597, 0.0091227, 0.0058597, 0.0091227, -0.0021629, 0.0021629
3: -0.0054254, -0.0039402, -0.0054254, -0.0039402, -0.0009844, 0.0009844
4: 0.0016620, 0.0022936, 0.0016620, 0.0022936, -0.0004186, 0.0004186
5: 0.0063294, 0.0104335, 0.0063294, 0.0104335, -0.0027203, 0.0027203
6: -0.0011073, -0.0000656, -0.0011073, -0.0000656, -0.0006904, 0.0006904
7: -0.0060025, -0.0033074, -0.0060025, -0.0033074, -0.0017864, 0.0017864
8: -0.0027208, -0.0013035, -0.0027208, -0.0013035, -0.0009394, 0.0009394
9: -0.0003524, 0.0012911, -0.0003524, 0.0012911, -0.0010893, 0.0010893

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 70

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011679, upper bound: 0.0011938
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0012315
time: 1.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.56 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -0.0011714, upper bound: 0.0011496
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -0.0011868, upper bound: 0.0011834
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -0.0011714, upper bound: 0.0011496
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -0.0011868, upper bound: 0.0011834
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -0.0011679, upper bound: 0.0011938
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0012315
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -0.0011679, upper bound: 0.0011938
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 0, lower bound: -0.0011834, upper bound: 0.0012315

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9906541, 0.9928788, 0.9907041, 0.9929988, -0.0015045, 0.0013703
1: -0.0035927, -0.0030384, -0.0035803, -0.0030085, -0.0003749, 0.0003414
2: 0.0060478, 0.0089854, 0.0058894, 0.0089195, -0.0018095, 0.0019867
3: -0.0053629, -0.0040258, -0.0053329, -0.0039537, -0.0009043, 0.0008236
4: 0.0016984, 0.0022670, 0.0016678, 0.0022542, -0.0003502, 0.0003845
5: 0.0065659, 0.0102607, 0.0063667, 0.0101778, -0.0022758, 0.0024988
6: -0.0010635, -0.0001257, -0.0010424, -0.0000751, -0.0006342, 0.0005776
7: -0.0058891, -0.0034628, -0.0058346, -0.0033320, -0.0016409, 0.0014945
8: -0.0026612, -0.0013852, -0.0026325, -0.0013164, -0.0008629, 0.0007859
9: -0.0002577, 0.0012219, -0.0003374, 0.0011887, -0.0009113, 0.0010006

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011552, upper bound: 0.0011552
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011552, upper bound: 0.0011552
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9907055, 0.9929835, 0.9906998, 0.9930281, -0.0015581, 0.0013884
1: -0.0035799, -0.0030123, -0.0035813, -0.0030012, -0.0003882, 0.0003459
2: 0.0059096, 0.0089176, 0.0058506, 0.0089251, -0.0018333, 0.0020575
3: -0.0053320, -0.0039629, -0.0053354, -0.0039361, -0.0009365, 0.0008344
4: 0.0016717, 0.0022539, 0.0016603, 0.0022553, -0.0003548, 0.0003982
5: 0.0063921, 0.0101754, 0.0063180, 0.0101849, -0.0023058, 0.0025878
6: -0.0010418, -0.0000816, -0.0010442, -0.0000627, -0.0006568, 0.0005852
7: -0.0058331, -0.0033486, -0.0058393, -0.0033000, -0.0016994, 0.0015142
8: -0.0026317, -0.0013252, -0.0026350, -0.0012996, -0.0008937, 0.0007963
9: -0.0003273, 0.0011877, -0.0003569, 0.0011915, -0.0009234, 0.0010363

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011552, upper bound: 0.0011737
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011552, upper bound: 0.0011886
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9906541, 0.9928788, 0.9905533, 0.9929916, -0.0016189, 0.0016377
1: -0.0035927, -0.0030384, -0.0036178, -0.0030103, -0.0004034, 0.0004081
2: 0.0060478, 0.0089854, 0.0058988, 0.0091186, -0.0021626, 0.0021377
3: -0.0053629, -0.0040258, -0.0054235, -0.0039580, -0.0009730, 0.0009843
4: 0.0016984, 0.0022670, 0.0016696, 0.0022928, -0.0004186, 0.0004137
5: 0.0065659, 0.0102607, 0.0063786, 0.0104283, -0.0027199, 0.0026887
6: -0.0010635, -0.0001257, -0.0011060, -0.0000781, -0.0006824, 0.0006903
7: -0.0058891, -0.0034628, -0.0059991, -0.0033397, -0.0017656, 0.0017861
8: -0.0026612, -0.0013852, -0.0027190, -0.0013205, -0.0009285, 0.0009393
9: -0.0002577, 0.0012219, -0.0003327, 0.0012890, -0.0010892, 0.0010767

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011578, upper bound: 0.0011060
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011730, upper bound: 0.0011060
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9907055, 0.9929835, 0.9905502, 0.9930212, -0.0016659, 0.0016878
1: -0.0035799, -0.0030123, -0.0036186, -0.0030029, -0.0004151, 0.0004205
2: 0.0059096, 0.0089176, 0.0058597, 0.0091227, -0.0022287, 0.0021998
3: -0.0053320, -0.0039629, -0.0054254, -0.0039402, -0.0010012, 0.0010144
4: 0.0016717, 0.0022539, 0.0016620, 0.0022936, -0.0004314, 0.0004258
5: 0.0063921, 0.0101754, 0.0063294, 0.0104335, -0.0028031, 0.0027667
6: -0.0010418, -0.0000816, -0.0011073, -0.0000656, -0.0007022, 0.0007115
7: -0.0058331, -0.0033486, -0.0060025, -0.0033074, -0.0018169, 0.0018408
8: -0.0026317, -0.0013252, -0.0027208, -0.0013035, -0.0009555, 0.0009680
9: -0.0003273, 0.0011877, -0.0003524, 0.0012911, -0.0011225, 0.0011079

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011938, upper bound: 0.0011679
time: 1.26 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011938, upper bound: 0.0011834
time: 1.37 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9905003, 0.9928735, 0.9907041, 0.9929988, -0.0018054, 0.0014967
1: -0.0036310, -0.0030397, -0.0035803, -0.0030085, -0.0004498, 0.0003729
2: 0.0060547, 0.0091884, 0.0058894, 0.0089195, -0.0019763, 0.0023840
3: -0.0054553, -0.0040290, -0.0053329, -0.0039537, -0.0010851, 0.0008995
4: 0.0016998, 0.0023063, 0.0016678, 0.0022542, -0.0003825, 0.0004614
5: 0.0065747, 0.0105161, 0.0063667, 0.0101778, -0.0024857, 0.0029984
6: -0.0011283, -0.0001279, -0.0010424, -0.0000751, -0.0007610, 0.0006309
7: -0.0060568, -0.0034686, -0.0058346, -0.0033320, -0.0019690, 0.0016323
8: -0.0027494, -0.0013882, -0.0026325, -0.0013164, -0.0010355, 0.0008584
9: -0.0002541, 0.0013242, -0.0003374, 0.0011887, -0.0009954, 0.0012007

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0011938
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0011938
time: 1.47 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9905549, 0.9929780, 0.9906998, 0.9930281, -0.0018262, 0.0015153
1: -0.0036174, -0.0030137, -0.0035813, -0.0030012, -0.0004550, 0.0003776
2: 0.0059168, 0.0091166, 0.0058506, 0.0089251, -0.0020009, 0.0024115
3: -0.0054226, -0.0039662, -0.0053354, -0.0039361, -0.0010976, 0.0009107
4: 0.0016731, 0.0022924, 0.0016603, 0.0022553, -0.0003873, 0.0004667
5: 0.0064012, 0.0104257, 0.0063180, 0.0101849, -0.0025166, 0.0030330
6: -0.0011053, -0.0000839, -0.0010442, -0.0000627, -0.0007698, 0.0006388
7: -0.0059974, -0.0033546, -0.0058393, -0.0033000, -0.0019917, 0.0016526
8: -0.0027181, -0.0013283, -0.0026350, -0.0012996, -0.0010474, 0.0008691
9: -0.0003236, 0.0012880, -0.0003569, 0.0011915, -0.0010078, 0.0012145

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0012157
time: 1.26 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0012315
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9905003, 0.9928735, 0.9905533, 0.9929916, -0.0015767, 0.0014434
1: -0.0036310, -0.0030397, -0.0036178, -0.0030103, -0.0003929, 0.0003597
2: 0.0060547, 0.0091884, 0.0058988, 0.0091186, -0.0019060, 0.0020820
3: -0.0054553, -0.0040290, -0.0054235, -0.0039580, -0.0009476, 0.0008675
4: 0.0016998, 0.0023063, 0.0016696, 0.0022928, -0.0003689, 0.0004030
5: 0.0065747, 0.0105161, 0.0063786, 0.0104283, -0.0023973, 0.0026186
6: -0.0011283, -0.0001279, -0.0011060, -0.0000781, -0.0006646, 0.0006085
7: -0.0060568, -0.0034686, -0.0059991, -0.0033397, -0.0017196, 0.0015743
8: -0.0027494, -0.0013882, -0.0027190, -0.0013205, -0.0009043, 0.0008279
9: -0.0002541, 0.0013242, -0.0003327, 0.0012890, -0.0009600, 0.0010486

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011122, upper bound: 0.0011477
time: 1.12 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011255, upper bound: 0.0011477
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9905549, 0.9929780, 0.9905502, 0.9930212, -0.0016292, 0.0014606
1: -0.0036174, -0.0030137, -0.0036186, -0.0030029, -0.0004060, 0.0003639
2: 0.0059168, 0.0091166, 0.0058597, 0.0091227, -0.0019287, 0.0021514
3: -0.0054226, -0.0039662, -0.0054254, -0.0039402, -0.0009792, 0.0008779
4: 0.0016731, 0.0022924, 0.0016620, 0.0022936, -0.0003733, 0.0004164
5: 0.0064012, 0.0104257, 0.0063294, 0.0104335, -0.0024258, 0.0027059
6: -0.0011053, -0.0000839, -0.0011073, -0.0000656, -0.0006868, 0.0006157
7: -0.0059974, -0.0033546, -0.0060025, -0.0033074, -0.0017769, 0.0015930
8: -0.0027181, -0.0013283, -0.0027208, -0.0013035, -0.0009345, 0.0008377
9: -0.0003236, 0.0012880, -0.0003524, 0.0012911, -0.0009714, 0.0010836

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 70

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0012156
time: 1.15 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0012315
time: 1.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.33 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011552, upper bound: 0.0011552
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011552, upper bound: 0.0011552
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011552, upper bound: 0.0011737
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011552, upper bound: 0.0011886
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011578, upper bound: 0.0011060
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011730, upper bound: 0.0011060
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011938, upper bound: 0.0011679
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011938, upper bound: 0.0011834
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0011938
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0011938
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0012157
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0012315
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011122, upper bound: 0.0011477
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011255, upper bound: 0.0011477
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0012156
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.33
Output dim: 0, lower bound: -0.0011496, upper bound: 0.0012315

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9906541, 0.9928788, 0.9906541, 0.9928788, -0.0013537, 0.0013537
1: -0.0035927, -0.0030384, -0.0035927, -0.0030384, -0.0003373, 0.0003373
2: 0.0060478, 0.0089854, 0.0060478, 0.0089854, -0.0017875, 0.0017875
3: -0.0053629, -0.0040258, -0.0053629, -0.0040258, -0.0008136, 0.0008136
4: 0.0016984, 0.0022670, 0.0016984, 0.0022670, -0.0003460, 0.0003460
5: 0.0065659, 0.0102607, 0.0065659, 0.0102607, -0.0022482, 0.0022482
6: -0.0010635, -0.0001257, -0.0010635, -0.0001257, -0.0005706, 0.0005706
7: -0.0058891, -0.0034628, -0.0058891, -0.0034628, -0.0014764, 0.0014764
8: -0.0026612, -0.0013852, -0.0026612, -0.0013852, -0.0007764, 0.0007764
9: -0.0002577, 0.0012219, -0.0002577, 0.0012219, -0.0009003, 0.0009003

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0010954
time: 1.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011114
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9906541, 0.9928788, 0.9907055, 0.9929835, -0.0015327, 0.0013651
1: -0.0035927, -0.0030384, -0.0035799, -0.0030123, -0.0003819, 0.0003402
2: 0.0060478, 0.0089854, 0.0059096, 0.0089176, -0.0018026, 0.0020240
3: -0.0053629, -0.0040258, -0.0053320, -0.0039629, -0.0009212, 0.0008205
4: 0.0016984, 0.0022670, 0.0016717, 0.0022539, -0.0003489, 0.0003917
5: 0.0065659, 0.0102607, 0.0063921, 0.0101754, -0.0022672, 0.0025456
6: -0.0010635, -0.0001257, -0.0010418, -0.0000816, -0.0006461, 0.0005754
7: -0.0058891, -0.0034628, -0.0058331, -0.0033486, -0.0016717, 0.0014889
8: -0.0026612, -0.0013852, -0.0026317, -0.0013252, -0.0008791, 0.0007830
9: -0.0002577, 0.0012219, -0.0003273, 0.0011877, -0.0009079, 0.0010194

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0010954
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011114
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9907055, 0.9929835, 0.9906541, 0.9928788, -0.0013651, 0.0015327
1: -0.0035799, -0.0030123, -0.0035927, -0.0030384, -0.0003402, 0.0003819
2: 0.0059096, 0.0089176, 0.0060478, 0.0089854, -0.0020240, 0.0018026
3: -0.0053320, -0.0039629, -0.0053629, -0.0040258, -0.0008205, 0.0009212
4: 0.0016717, 0.0022539, 0.0016984, 0.0022670, -0.0003917, 0.0003489
5: 0.0063921, 0.0101754, 0.0065659, 0.0102607, -0.0025456, 0.0022672
6: -0.0010418, -0.0000816, -0.0010635, -0.0001257, -0.0005754, 0.0006461
7: -0.0058331, -0.0033486, -0.0058891, -0.0034628, -0.0014889, 0.0016717
8: -0.0026317, -0.0013252, -0.0026612, -0.0013852, -0.0007830, 0.0008791
9: -0.0003273, 0.0011877, -0.0002577, 0.0012219, -0.0010194, 0.0009079

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011169
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011306
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9907055, 0.9929835, 0.9907055, 0.9929835, -0.0013799, 0.0013799
1: -0.0035799, -0.0030123, -0.0035799, -0.0030123, -0.0003438, 0.0003438
2: 0.0059096, 0.0089176, 0.0059096, 0.0089176, -0.0018221, 0.0018221
3: -0.0053320, -0.0039629, -0.0053320, -0.0039629, -0.0008293, 0.0008293
4: 0.0016717, 0.0022539, 0.0016717, 0.0022539, -0.0003527, 0.0003527
5: 0.0063921, 0.0101754, 0.0063921, 0.0101754, -0.0022917, 0.0022917
6: -0.0010418, -0.0000816, -0.0010418, -0.0000816, -0.0005817, 0.0005817
7: -0.0058331, -0.0033486, -0.0058331, -0.0033486, -0.0015049, 0.0015049
8: -0.0026317, -0.0013252, -0.0026317, -0.0013252, -0.0007914, 0.0007914
9: -0.0003273, 0.0011877, -0.0003273, 0.0011877, -0.0009177, 0.0009177

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011322
time: 1.51 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011463
time: 1.37 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9906544, 0.9928747, 0.9905586, 0.9929273, -0.0015439, 0.0016222
1: -0.0035926, -0.0030394, -0.0036165, -0.0030263, -0.0003847, 0.0004042
2: 0.0060531, 0.0089850, 0.0059837, 0.0091116, -0.0021421, 0.0020387
3: -0.0053627, -0.0040282, -0.0054203, -0.0039966, -0.0009279, 0.0009750
4: 0.0016995, 0.0022669, 0.0016860, 0.0022914, -0.0004146, 0.0003946
5: 0.0065727, 0.0102602, 0.0064853, 0.0104195, -0.0026942, 0.0025642
6: -0.0010633, -0.0001274, -0.0011037, -0.0001052, -0.0006508, 0.0006838
7: -0.0058888, -0.0034672, -0.0059933, -0.0034099, -0.0016839, 0.0017692
8: -0.0026610, -0.0013875, -0.0027160, -0.0013574, -0.0008855, 0.0009304
9: -0.0002550, 0.0012217, -0.0002899, 0.0012855, -0.0010789, 0.0010268

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010812, upper bound: 0.0009935
time: 1.20 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010454, upper bound: 0.0009813
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9906548, 0.9928698, 0.9905150, 0.9929349, -0.0015624, 0.0016845
1: -0.0035925, -0.0030406, -0.0036274, -0.0030244, -0.0003893, 0.0004197
2: 0.0060596, 0.0089846, 0.0059737, 0.0091691, -0.0022244, 0.0020632
3: -0.0053625, -0.0040312, -0.0054465, -0.0039921, -0.0009391, 0.0010124
4: 0.0017007, 0.0022668, 0.0016841, 0.0023026, -0.0004305, 0.0003993
5: 0.0065809, 0.0102597, 0.0064728, 0.0104918, -0.0027977, 0.0025949
6: -0.0010632, -0.0001295, -0.0011221, -0.0001020, -0.0006586, 0.0007101
7: -0.0058884, -0.0034726, -0.0060408, -0.0034016, -0.0017040, 0.0018372
8: -0.0026608, -0.0013904, -0.0027410, -0.0013530, -0.0008961, 0.0009662
9: -0.0002517, 0.0012215, -0.0002950, 0.0013144, -0.0011203, 0.0010391

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010932, upper bound: 0.0009933
time: 1.13 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010545, upper bound: 0.0009800
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9907055, 0.9929835, 0.9905003, 0.9928735, -0.0014915, 0.0018336
1: -0.0035799, -0.0030123, -0.0036310, -0.0030397, -0.0003716, 0.0004569
2: 0.0059096, 0.0089176, 0.0060547, 0.0091884, -0.0024212, 0.0019695
3: -0.0053320, -0.0039629, -0.0054553, -0.0040290, -0.0008964, 0.0011020
4: 0.0016717, 0.0022539, 0.0016998, 0.0023063, -0.0004686, 0.0003812
5: 0.0063921, 0.0101754, 0.0065747, 0.0105161, -0.0030452, 0.0024771
6: -0.0010418, -0.0000816, -0.0011283, -0.0001279, -0.0006287, 0.0007729
7: -0.0058331, -0.0033486, -0.0060568, -0.0034686, -0.0016267, 0.0019997
8: -0.0026317, -0.0013252, -0.0027494, -0.0013882, -0.0008555, 0.0010516
9: -0.0003273, 0.0011877, -0.0002541, 0.0013242, -0.0012194, 0.0009919

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011476, upper bound: 0.0011122
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011476, upper bound: 0.0011255
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9907055, 0.9929835, 0.9905549, 0.9929780, -0.0015068, 0.0016799
1: -0.0035799, -0.0030123, -0.0036174, -0.0030137, -0.0003755, 0.0004186
2: 0.0059096, 0.0089176, 0.0059168, 0.0091166, -0.0022183, 0.0019897
3: -0.0053320, -0.0039629, -0.0054226, -0.0039662, -0.0009056, 0.0010097
4: 0.0016717, 0.0022539, 0.0016731, 0.0022924, -0.0004294, 0.0003851
5: 0.0063921, 0.0101754, 0.0064012, 0.0104257, -0.0027901, 0.0025025
6: -0.0010418, -0.0000816, -0.0011053, -0.0000839, -0.0006352, 0.0007082
7: -0.0058331, -0.0033486, -0.0059974, -0.0033546, -0.0016434, 0.0018322
8: -0.0026317, -0.0013252, -0.0027181, -0.0013283, -0.0008642, 0.0009635
9: -0.0003273, 0.0011877, -0.0003236, 0.0012880, -0.0011173, 0.0010021

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011476, upper bound: 0.0011286
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011476, upper bound: 0.0011417
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905003, 0.9928735, 0.9906541, 0.9928788, -0.0016545, 0.0014800
1: -0.0036310, -0.0030397, -0.0035927, -0.0030384, -0.0004123, 0.0003688
2: 0.0060547, 0.0091884, 0.0060478, 0.0089854, -0.0019544, 0.0021848
3: -0.0054553, -0.0040290, -0.0053629, -0.0040258, -0.0009944, 0.0008896
4: 0.0016998, 0.0023063, 0.0016984, 0.0022670, -0.0003783, 0.0004229
5: 0.0065747, 0.0105161, 0.0065659, 0.0102607, -0.0024581, 0.0027478
6: -0.0011283, -0.0001279, -0.0010635, -0.0001257, -0.0006974, 0.0006239
7: -0.0060568, -0.0034686, -0.0058891, -0.0034628, -0.0018045, 0.0016142
8: -0.0027494, -0.0013882, -0.0026612, -0.0013852, -0.0009490, 0.0008489
9: -0.0002541, 0.0013242, -0.0002577, 0.0012219, -0.0009843, 0.0011004

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011306
time: 1.28 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011476
time: 1.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905003, 0.9928735, 0.9907055, 0.9929835, -0.0018336, 0.0014915
1: -0.0036310, -0.0030397, -0.0035799, -0.0030123, -0.0004569, 0.0003716
2: 0.0060547, 0.0091884, 0.0059096, 0.0089176, -0.0019695, 0.0024212
3: -0.0054553, -0.0040290, -0.0053320, -0.0039629, -0.0011020, 0.0008964
4: 0.0016998, 0.0023063, 0.0016717, 0.0022539, -0.0003812, 0.0004686
5: 0.0065747, 0.0105161, 0.0063921, 0.0101754, -0.0024771, 0.0030452
6: -0.0011283, -0.0001279, -0.0010418, -0.0000816, -0.0007729, 0.0006287
7: -0.0060568, -0.0034686, -0.0058331, -0.0033486, -0.0019997, 0.0016267
8: -0.0027494, -0.0013882, -0.0026317, -0.0013252, -0.0010516, 0.0008555
9: -0.0002541, 0.0013242, -0.0003273, 0.0011877, -0.0009919, 0.0012194

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011306
time: 1.41 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011477
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9905549, 0.9929780, 0.9906541, 0.9928788, -0.0016332, 0.0016245
1: -0.0036174, -0.0030137, -0.0035927, -0.0030384, -0.0004069, 0.0004048
2: 0.0059168, 0.0091166, 0.0060478, 0.0089854, -0.0021451, 0.0021566
3: -0.0054226, -0.0039662, -0.0053629, -0.0040258, -0.0009816, 0.0009764
4: 0.0016731, 0.0022924, 0.0016984, 0.0022670, -0.0004152, 0.0004174
5: 0.0064012, 0.0104257, 0.0065659, 0.0102607, -0.0026980, 0.0027125
6: -0.0011053, -0.0000839, -0.0010635, -0.0001257, -0.0006884, 0.0006848
7: -0.0059974, -0.0033546, -0.0058891, -0.0034628, -0.0017812, 0.0017717
8: -0.0027181, -0.0013283, -0.0026612, -0.0013852, -0.0009367, 0.0009317
9: -0.0003236, 0.0012880, -0.0002577, 0.0012219, -0.0010804, 0.0010862

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011578
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011730
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9905549, 0.9929780, 0.9907055, 0.9929835, -0.0016799, 0.0015068
1: -0.0036174, -0.0030137, -0.0035799, -0.0030123, -0.0004186, 0.0003755
2: 0.0059168, 0.0091166, 0.0059096, 0.0089176, -0.0019897, 0.0022183
3: -0.0054226, -0.0039662, -0.0053320, -0.0039629, -0.0010097, 0.0009056
4: 0.0016731, 0.0022924, 0.0016717, 0.0022539, -0.0003851, 0.0004294
5: 0.0064012, 0.0104257, 0.0063921, 0.0101754, -0.0025025, 0.0027901
6: -0.0011053, -0.0000839, -0.0010418, -0.0000816, -0.0007082, 0.0006352
7: -0.0059974, -0.0033546, -0.0058331, -0.0033486, -0.0018322, 0.0016434
8: -0.0027181, -0.0013283, -0.0026317, -0.0013252, -0.0009635, 0.0008642
9: -0.0003236, 0.0012880, -0.0003273, 0.0011877, -0.0010021, 0.0011173

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011742
time: 1.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011912
time: 1.60 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905007, 0.9928694, 0.9905586, 0.9929273, -0.0014967, 0.0014277
1: -0.0036309, -0.0030407, -0.0036165, -0.0030263, -0.0003729, 0.0003557
2: 0.0060602, 0.0091880, 0.0059837, 0.0091116, -0.0018852, 0.0019764
3: -0.0054551, -0.0040315, -0.0054203, -0.0039966, -0.0008996, 0.0008581
4: 0.0017008, 0.0023062, 0.0016860, 0.0022914, -0.0003649, 0.0003825
5: 0.0065816, 0.0105155, 0.0064853, 0.0104195, -0.0023711, 0.0024858
6: -0.0011281, -0.0001296, -0.0011037, -0.0001052, -0.0006309, 0.0006018
7: -0.0060564, -0.0034731, -0.0059933, -0.0034099, -0.0016324, 0.0015571
8: -0.0027492, -0.0013906, -0.0027160, -0.0013574, -0.0008585, 0.0008188
9: -0.0002514, 0.0013239, -0.0002899, 0.0012855, -0.0009495, 0.0009954

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010373, upper bound: 0.0010104
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010144, upper bound: 0.0010031
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905010, 0.9928648, 0.9905150, 0.9929349, -0.0015179, 0.0015058
1: -0.0036308, -0.0030419, -0.0036274, -0.0030244, -0.0003782, 0.0003752
2: 0.0060663, 0.0091876, 0.0059737, 0.0091691, -0.0019884, 0.0020043
3: -0.0054549, -0.0040342, -0.0054465, -0.0039921, -0.0009123, 0.0009050
4: 0.0017020, 0.0023061, 0.0016841, 0.0023026, -0.0003849, 0.0003879
5: 0.0065892, 0.0105150, 0.0064728, 0.0104918, -0.0025009, 0.0025209
6: -0.0011280, -0.0001316, -0.0011221, -0.0001020, -0.0006398, 0.0006348
7: -0.0060561, -0.0034781, -0.0060408, -0.0034016, -0.0016554, 0.0016423
8: -0.0027490, -0.0013932, -0.0027410, -0.0013530, -0.0008706, 0.0008637
9: -0.0002483, 0.0013237, -0.0002950, 0.0013144, -0.0010015, 0.0010095

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010460, upper bound: 0.0010099
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010206, upper bound: 0.0010011
time: 1.14 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9905549, 0.9929780, 0.9905003, 0.9928735, -0.0014381, 0.0016050
1: -0.0036174, -0.0030137, -0.0036310, -0.0030397, -0.0003583, 0.0003999
2: 0.0059168, 0.0091166, 0.0060547, 0.0091884, -0.0021194, 0.0018990
3: -0.0054226, -0.0039662, -0.0054553, -0.0040290, -0.0008643, 0.0009647
4: 0.0016731, 0.0022924, 0.0016998, 0.0023063, -0.0004102, 0.0003675
5: 0.0064012, 0.0104257, 0.0065747, 0.0105161, -0.0026657, 0.0023884
6: -0.0011053, -0.0000839, -0.0011283, -0.0001279, -0.0006062, 0.0006766
7: -0.0059974, -0.0033546, -0.0060568, -0.0034686, -0.0015684, 0.0017505
8: -0.0027181, -0.0013283, -0.0027494, -0.0013882, -0.0008248, 0.0009206
9: -0.0003236, 0.0012880, -0.0002541, 0.0013242, -0.0010675, 0.0009564

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011578
time: 1.19 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011730
time: 1.19 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9905549, 0.9929780, 0.9905549, 0.9929780, -0.0014534, 0.0014534
1: -0.0036174, -0.0030137, -0.0036174, -0.0030137, -0.0003622, 0.0003622
2: 0.0059168, 0.0091166, 0.0059168, 0.0091166, -0.0019192, 0.0019192
3: -0.0054226, -0.0039662, -0.0054226, -0.0039662, -0.0008735, 0.0008735
4: 0.0016731, 0.0022924, 0.0016731, 0.0022924, -0.0003715, 0.0003715
5: 0.0064012, 0.0104257, 0.0064012, 0.0104257, -0.0024139, 0.0024139
6: -0.0011053, -0.0000839, -0.0011053, -0.0000839, -0.0006127, 0.0006127
7: -0.0059974, -0.0033546, -0.0059974, -0.0033546, -0.0015851, 0.0015851
8: -0.0027181, -0.0013283, -0.0027181, -0.0013283, -0.0008336, 0.0008336
9: -0.0003236, 0.0012880, -0.0003236, 0.0012880, -0.0009666, 0.0009666

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011742
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011912
time: 1.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.05 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0010954
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011114
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0010954
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011114
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011169
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011306
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011322
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011114, upper bound: 0.0011463
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0010812, upper bound: 0.0009935
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0010454, upper bound: 0.0009813
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0010932, upper bound: 0.0009933
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0010545, upper bound: 0.0009800
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011476, upper bound: 0.0011122
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011476, upper bound: 0.0011255
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011476, upper bound: 0.0011286
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011476, upper bound: 0.0011417
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011306
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011476
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011306
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011477
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011578
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011730
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011742
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011912
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0010373, upper bound: 0.0010104
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0010144, upper bound: 0.0010031
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0010460, upper bound: 0.0010099
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0010206, upper bound: 0.0010011
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011578
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011730
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011742
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.05
Output dim: 0, lower bound: -0.0011060, upper bound: 0.0011912

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9906593, 0.9928152, 0.9906544, 0.9928747, -0.0013356, 0.0012751
1: -0.0035914, -0.0030542, -0.0035926, -0.0030394, -0.0003328, 0.0003177
2: 0.0061317, 0.0089786, 0.0060531, 0.0089850, -0.0016838, 0.0017636
3: -0.0053598, -0.0040640, -0.0053627, -0.0040282, -0.0008027, 0.0007664
4: 0.0017147, 0.0022657, 0.0016995, 0.0022669, -0.0003259, 0.0003413
5: 0.0066716, 0.0102521, 0.0065727, 0.0102602, -0.0021177, 0.0022182
6: -0.0010613, -0.0001525, -0.0010633, -0.0001274, -0.0005630, 0.0005375
7: -0.0058835, -0.0035321, -0.0058888, -0.0034672, -0.0014567, 0.0013907
8: -0.0026582, -0.0014217, -0.0026610, -0.0013875, -0.0007660, 0.0007313
9: -0.0002154, 0.0012185, -0.0002550, 0.0012217, -0.0008480, 0.0008883

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009992, upper bound: 0.0010086
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009889, upper bound: 0.0009776
time: 1.11 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9906198, 0.9928198, 0.9906548, 0.9928698, -0.0014211, 0.0012958
1: -0.0036013, -0.0030531, -0.0035925, -0.0030406, -0.0003541, 0.0003229
2: 0.0061256, 0.0090308, 0.0060596, 0.0089846, -0.0017111, 0.0018765
3: -0.0053835, -0.0040612, -0.0053625, -0.0040312, -0.0008541, 0.0007788
4: 0.0017135, 0.0022758, 0.0017007, 0.0022668, -0.0003312, 0.0003632
5: 0.0066639, 0.0103178, 0.0065809, 0.0102597, -0.0021522, 0.0023602
6: -0.0010779, -0.0001505, -0.0010632, -0.0001295, -0.0005990, 0.0005462
7: -0.0059266, -0.0035271, -0.0058884, -0.0034726, -0.0015499, 0.0014133
8: -0.0026809, -0.0014190, -0.0026608, -0.0013904, -0.0008151, 0.0007432
9: -0.0002184, 0.0012447, -0.0002517, 0.0012215, -0.0008618, 0.0009451

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009987, upper bound: 0.0010199
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009862, upper bound: 0.0009862
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9906593, 0.9928152, 0.9907058, 0.9929794, -0.0015150, 0.0012867
1: -0.0035914, -0.0030542, -0.0035798, -0.0030133, -0.0003775, 0.0003206
2: 0.0061317, 0.0089786, 0.0059148, 0.0089172, -0.0016990, 0.0020005
3: -0.0053598, -0.0040640, -0.0053318, -0.0039653, -0.0009106, 0.0007733
4: 0.0017147, 0.0022657, 0.0016727, 0.0022538, -0.0003288, 0.0003872
5: 0.0066716, 0.0102521, 0.0063988, 0.0101749, -0.0021370, 0.0025161
6: -0.0010613, -0.0001525, -0.0010417, -0.0000832, -0.0006386, 0.0005424
7: -0.0058835, -0.0035321, -0.0058327, -0.0033530, -0.0016523, 0.0014033
8: -0.0026582, -0.0014217, -0.0026315, -0.0013275, -0.0008689, 0.0007380
9: -0.0002154, 0.0012185, -0.0003246, 0.0011875, -0.0008557, 0.0010076

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010311, upper bound: 0.0010086
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010291, upper bound: 0.0009782
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9906198, 0.9928198, 0.9907061, 0.9929743, -0.0015988, 0.0013071
1: -0.0036013, -0.0030531, -0.0035797, -0.0030146, -0.0003984, 0.0003257
2: 0.0061256, 0.0090308, 0.0059216, 0.0089168, -0.0017260, 0.0021111
3: -0.0053835, -0.0040612, -0.0053316, -0.0039684, -0.0009609, 0.0007856
4: 0.0017135, 0.0022758, 0.0016740, 0.0022537, -0.0003341, 0.0004086
5: 0.0066639, 0.0103178, 0.0064073, 0.0101744, -0.0021708, 0.0026553
6: -0.0010779, -0.0001505, -0.0010415, -0.0000854, -0.0006739, 0.0005510
7: -0.0059266, -0.0035271, -0.0058324, -0.0033586, -0.0017437, 0.0014256
8: -0.0026809, -0.0014190, -0.0026314, -0.0013304, -0.0009170, 0.0007497
9: -0.0002184, 0.0012447, -0.0003212, 0.0011873, -0.0008693, 0.0010633

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010311, upper bound: 0.0010199
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010283, upper bound: 0.0009873
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9907106, 0.9929209, 0.9906544, 0.9928747, -0.0013489, 0.0014518
1: -0.0035786, -0.0030279, -0.0035926, -0.0030394, -0.0003361, 0.0003618
2: 0.0059921, 0.0089109, 0.0060531, 0.0089850, -0.0019172, 0.0017812
3: -0.0053290, -0.0040005, -0.0053627, -0.0040282, -0.0008107, 0.0008726
4: 0.0016877, 0.0022526, 0.0016995, 0.0022669, -0.0003711, 0.0003447
5: 0.0064960, 0.0101670, 0.0065727, 0.0102602, -0.0024113, 0.0022403
6: -0.0010397, -0.0001079, -0.0010633, -0.0001274, -0.0005686, 0.0006120
7: -0.0058276, -0.0034168, -0.0058888, -0.0034672, -0.0014712, 0.0015834
8: -0.0026288, -0.0013610, -0.0026610, -0.0013875, -0.0007737, 0.0008327
9: -0.0002857, 0.0011844, -0.0002550, 0.0012217, -0.0009656, 0.0008971

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009995, upper bound: 0.0010449
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009893, upper bound: 0.0010219
time: 1.36 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9906732, 0.9929241, 0.9906548, 0.9928698, -0.0014276, 0.0014699
1: -0.0035879, -0.0030271, -0.0035925, -0.0030406, -0.0003557, 0.0003663
2: 0.0059879, 0.0089602, 0.0060596, 0.0089846, -0.0019410, 0.0018851
3: -0.0053514, -0.0039986, -0.0053625, -0.0040312, -0.0008580, 0.0008834
4: 0.0016868, 0.0022621, 0.0017007, 0.0022668, -0.0003757, 0.0003649
5: 0.0064907, 0.0102290, 0.0065809, 0.0102597, -0.0024412, 0.0023709
6: -0.0010554, -0.0001066, -0.0010632, -0.0001295, -0.0006018, 0.0006196
7: -0.0058683, -0.0034133, -0.0058884, -0.0034726, -0.0015570, 0.0016031
8: -0.0026502, -0.0013592, -0.0026608, -0.0013904, -0.0008188, 0.0008431
9: -0.0002878, 0.0012092, -0.0002517, 0.0012215, -0.0009776, 0.0009494

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009991, upper bound: 0.0010539
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009873, upper bound: 0.0010283
time: 1.33 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9907106, 0.9929209, 0.9907058, 0.9929794, -0.0013613, 0.0013007
1: -0.0035786, -0.0030279, -0.0035798, -0.0030133, -0.0003392, 0.0003241
2: 0.0059921, 0.0089109, 0.0059148, 0.0089172, -0.0017176, 0.0017976
3: -0.0053290, -0.0040005, -0.0053318, -0.0039653, -0.0008182, 0.0007818
4: 0.0016877, 0.0022526, 0.0016727, 0.0022538, -0.0003324, 0.0003479
5: 0.0064960, 0.0101670, 0.0063988, 0.0101749, -0.0021603, 0.0022609
6: -0.0010397, -0.0001079, -0.0010417, -0.0000832, -0.0005739, 0.0005483
7: -0.0058276, -0.0034168, -0.0058327, -0.0033530, -0.0014847, 0.0014187
8: -0.0026288, -0.0013610, -0.0026315, -0.0013275, -0.0007808, 0.0007461
9: -0.0002857, 0.0011844, -0.0003246, 0.0011875, -0.0008651, 0.0009054

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010578, upper bound: 0.0010722
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010659, upper bound: 0.0010686
time: 1.36 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9906732, 0.9929241, 0.9907061, 0.9929743, -0.0014452, 0.0013207
1: -0.0035879, -0.0030271, -0.0035797, -0.0030146, -0.0003601, 0.0003291
2: 0.0059879, 0.0089602, 0.0059216, 0.0089168, -0.0017440, 0.0019084
3: -0.0053514, -0.0039986, -0.0053316, -0.0039684, -0.0008686, 0.0007938
4: 0.0016868, 0.0022621, 0.0016740, 0.0022537, -0.0003375, 0.0003694
5: 0.0064907, 0.0102290, 0.0064073, 0.0101744, -0.0021935, 0.0024002
6: -0.0010554, -0.0001066, -0.0010415, -0.0000854, -0.0006092, 0.0005567
7: -0.0058683, -0.0034133, -0.0058324, -0.0033586, -0.0015762, 0.0014404
8: -0.0026502, -0.0013592, -0.0026314, -0.0013304, -0.0008289, 0.0007575
9: -0.0002878, 0.0012092, -0.0003212, 0.0011873, -0.0008784, 0.0009612

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010578, upper bound: 0.0010827
time: 1.17 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010659, upper bound: 0.0010780
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9907106, 0.9929209, 0.9905007, 0.9928694, -0.0014760, 0.0017528
1: -0.0035786, -0.0030279, -0.0036309, -0.0030407, -0.0003678, 0.0004368
2: 0.0059921, 0.0089109, 0.0060602, 0.0091880, -0.0023146, 0.0019490
3: -0.0053290, -0.0040005, -0.0054551, -0.0040315, -0.0008871, 0.0010535
4: 0.0016877, 0.0022526, 0.0017008, 0.0023062, -0.0004480, 0.0003772
5: 0.0064960, 0.0101670, 0.0065816, 0.0105155, -0.0029111, 0.0024514
6: -0.0010397, -0.0001079, -0.0011281, -0.0001296, -0.0006222, 0.0007389
7: -0.0058276, -0.0034168, -0.0060564, -0.0034731, -0.0016098, 0.0019117
8: -0.0026288, -0.0013610, -0.0027492, -0.0013906, -0.0008466, 0.0010053
9: -0.0002857, 0.0011844, -0.0002514, 0.0013239, -0.0011657, 0.0009816

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010104, upper bound: 0.0010373
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010031, upper bound: 0.0010144
time: 1.28 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9906732, 0.9929241, 0.9905010, 0.9928648, -0.0015518, 0.0017708
1: -0.0035879, -0.0030271, -0.0036308, -0.0030419, -0.0003867, 0.0004412
2: 0.0059879, 0.0089602, 0.0060663, 0.0091876, -0.0023383, 0.0020492
3: -0.0053514, -0.0039986, -0.0054549, -0.0040342, -0.0009327, 0.0010643
4: 0.0016868, 0.0022621, 0.0017020, 0.0023061, -0.0004526, 0.0003966
5: 0.0064907, 0.0102290, 0.0065892, 0.0105150, -0.0029409, 0.0025773
6: -0.0010554, -0.0001066, -0.0011280, -0.0001316, -0.0006541, 0.0007464
7: -0.0058683, -0.0034133, -0.0060561, -0.0034781, -0.0016925, 0.0019313
8: -0.0026502, -0.0013592, -0.0027490, -0.0013932, -0.0008901, 0.0010156
9: -0.0002878, 0.0012092, -0.0002483, 0.0013237, -0.0011777, 0.0010321

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010099, upper bound: 0.0010460
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010011, upper bound: 0.0010206
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9907106, 0.9929209, 0.9905552, 0.9929739, -0.0014890, 0.0016010
1: -0.0035786, -0.0030279, -0.0036174, -0.0030147, -0.0003710, 0.0003989
2: 0.0059921, 0.0089109, 0.0059223, 0.0091161, -0.0021141, 0.0019662
3: -0.0053290, -0.0040005, -0.0054224, -0.0039687, -0.0008949, 0.0009622
4: 0.0016877, 0.0022526, 0.0016741, 0.0022923, -0.0004092, 0.0003806
5: 0.0064960, 0.0101670, 0.0064081, 0.0104251, -0.0026590, 0.0024730
6: -0.0010397, -0.0001079, -0.0011052, -0.0000856, -0.0006277, 0.0006749
7: -0.0058276, -0.0034168, -0.0059971, -0.0033591, -0.0016240, 0.0017461
8: -0.0026288, -0.0013610, -0.0027179, -0.0013307, -0.0008540, 0.0009183
9: -0.0002857, 0.0011844, -0.0003209, 0.0012877, -0.0010648, 0.0009903

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010682
time: 1.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010651
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9906732, 0.9929241, 0.9905554, 0.9929693, -0.0015701, 0.0016209
1: -0.0035879, -0.0030271, -0.0036173, -0.0030158, -0.0003912, 0.0004039
2: 0.0059879, 0.0089602, 0.0059282, 0.0091157, -0.0021404, 0.0020734
3: -0.0053514, -0.0039986, -0.0054222, -0.0039714, -0.0009437, 0.0009742
4: 0.0016868, 0.0022621, 0.0016753, 0.0022922, -0.0004143, 0.0004013
5: 0.0064907, 0.0102290, 0.0064156, 0.0104247, -0.0026920, 0.0026077
6: -0.0010554, -0.0001066, -0.0011051, -0.0000875, -0.0006619, 0.0006833
7: -0.0058683, -0.0034133, -0.0059967, -0.0033641, -0.0017125, 0.0017678
8: -0.0026502, -0.0013592, -0.0027178, -0.0013333, -0.0009006, 0.0009297
9: -0.0002878, 0.0012092, -0.0003178, 0.0012875, -0.0010780, 0.0010442

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010781
time: 1.37 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010743
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9905059, 0.9928079, 0.9906544, 0.9928747, -0.0016384, 0.0014063
1: -0.0036296, -0.0030560, -0.0035926, -0.0030394, -0.0004083, 0.0003504
2: 0.0061413, 0.0091812, 0.0060531, 0.0089850, -0.0018570, 0.0021635
3: -0.0054520, -0.0040684, -0.0053627, -0.0040282, -0.0009848, 0.0008452
4: 0.0017165, 0.0023049, 0.0016995, 0.0022669, -0.0003594, 0.0004187
5: 0.0066836, 0.0105069, 0.0065727, 0.0102602, -0.0023356, 0.0027212
6: -0.0011259, -0.0001555, -0.0010633, -0.0001274, -0.0006907, 0.0005928
7: -0.0060508, -0.0035400, -0.0058888, -0.0034672, -0.0017870, 0.0015338
8: -0.0027462, -0.0014258, -0.0026610, -0.0013875, -0.0009397, 0.0008066
9: -0.0002105, 0.0013205, -0.0002550, 0.0012217, -0.0009353, 0.0010897

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009927, upper bound: 0.0010368
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009798, upper bound: 0.0009922
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9904637, 0.9928154, 0.9906548, 0.9928698, -0.0016983, 0.0014254
1: -0.0036402, -0.0030542, -0.0035925, -0.0030406, -0.0004232, 0.0003552
2: 0.0061314, 0.0092370, 0.0060596, 0.0089846, -0.0018822, 0.0022426
3: -0.0054774, -0.0040639, -0.0053625, -0.0040312, -0.0010207, 0.0008567
4: 0.0017146, 0.0023157, 0.0017007, 0.0022668, -0.0003643, 0.0004340
5: 0.0066712, 0.0105771, 0.0065809, 0.0102597, -0.0023673, 0.0028206
6: -0.0011438, -0.0001524, -0.0010632, -0.0001295, -0.0007159, 0.0006008
7: -0.0060969, -0.0035319, -0.0058884, -0.0034726, -0.0018522, 0.0015546
8: -0.0027704, -0.0014215, -0.0026608, -0.0013904, -0.0009741, 0.0008175
9: -0.0002155, 0.0013486, -0.0002517, 0.0012215, -0.0009480, 0.0011295

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009923, upper bound: 0.0010440
time: 1.16 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009772, upper bound: 0.0009980
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9905059, 0.9928079, 0.9907058, 0.9929794, -0.0018178, 0.0014179
1: -0.0036296, -0.0030560, -0.0035798, -0.0030133, -0.0004530, 0.0003533
2: 0.0061413, 0.0091812, 0.0059148, 0.0089172, -0.0018723, 0.0024004
3: -0.0054520, -0.0040684, -0.0053318, -0.0039653, -0.0010926, 0.0008522
4: 0.0017165, 0.0023049, 0.0016727, 0.0022538, -0.0003624, 0.0004646
5: 0.0066836, 0.0105069, 0.0063988, 0.0101749, -0.0023548, 0.0030191
6: -0.0011259, -0.0001555, -0.0010417, -0.0000832, -0.0007663, 0.0005977
7: -0.0060508, -0.0035400, -0.0058327, -0.0033530, -0.0019826, 0.0015464
8: -0.0027462, -0.0014258, -0.0026315, -0.0013275, -0.0010426, 0.0008132
9: -0.0002105, 0.0013205, -0.0003246, 0.0011875, -0.0009430, 0.0012090

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010257, upper bound: 0.0010368
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010211, upper bound: 0.0009948
time: 1.23 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9904637, 0.9928154, 0.9907061, 0.9929743, -0.0018760, 0.0014366
1: -0.0036402, -0.0030542, -0.0035797, -0.0030146, -0.0004674, 0.0003580
2: 0.0061314, 0.0092370, 0.0059216, 0.0089168, -0.0018970, 0.0024772
3: -0.0054774, -0.0040639, -0.0053316, -0.0039684, -0.0011275, 0.0008634
4: 0.0017146, 0.0023157, 0.0016740, 0.0022537, -0.0003672, 0.0004795
5: 0.0066712, 0.0105771, 0.0064073, 0.0101744, -0.0023860, 0.0031157
6: -0.0011438, -0.0001524, -0.0010415, -0.0000854, -0.0007908, 0.0006056
7: -0.0060969, -0.0035319, -0.0058324, -0.0033586, -0.0020460, 0.0015668
8: -0.0027704, -0.0014215, -0.0026314, -0.0013304, -0.0010760, 0.0008240
9: -0.0002155, 0.0013486, -0.0003212, 0.0011873, -0.0009554, 0.0012476

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010257, upper bound: 0.0010442
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010206, upper bound: 0.0010011
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9905601, 0.9929131, 0.9906544, 0.9928747, -0.0016179, 0.0015487
1: -0.0036161, -0.0030298, -0.0035926, -0.0030394, -0.0004031, 0.0003859
2: 0.0060026, 0.0091096, 0.0060531, 0.0089850, -0.0020451, 0.0021364
3: -0.0054194, -0.0040052, -0.0053627, -0.0040282, -0.0009724, 0.0009308
4: 0.0016897, 0.0022910, 0.0016995, 0.0022669, -0.0003958, 0.0004135
5: 0.0065091, 0.0104169, 0.0065727, 0.0102602, -0.0025722, 0.0026870
6: -0.0011031, -0.0001113, -0.0010633, -0.0001274, -0.0006820, 0.0006528
7: -0.0059917, -0.0034255, -0.0058888, -0.0034672, -0.0017645, 0.0016891
8: -0.0027151, -0.0013656, -0.0026610, -0.0013875, -0.0009279, 0.0008883
9: -0.0002804, 0.0012844, -0.0002550, 0.0012217, -0.0010300, 0.0010760

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009935, upper bound: 0.0010813
time: 1.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009813, upper bound: 0.0010454
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9905165, 0.9929214, 0.9906548, 0.9928698, -0.0016801, 0.0015666
1: -0.0036270, -0.0030277, -0.0035925, -0.0030406, -0.0004186, 0.0003903
2: 0.0059915, 0.0091671, 0.0060596, 0.0089846, -0.0020686, 0.0022185
3: -0.0054456, -0.0040002, -0.0053625, -0.0040312, -0.0010098, 0.0009416
4: 0.0016875, 0.0023022, 0.0017007, 0.0022668, -0.0004004, 0.0004294
5: 0.0064951, 0.0104892, 0.0065809, 0.0102597, -0.0026018, 0.0027903
6: -0.0011214, -0.0001077, -0.0010632, -0.0001295, -0.0007082, 0.0006604
7: -0.0060391, -0.0034163, -0.0058884, -0.0034726, -0.0018324, 0.0017086
8: -0.0027401, -0.0013607, -0.0026608, -0.0013904, -0.0009636, 0.0008985
9: -0.0002860, 0.0013134, -0.0002517, 0.0012215, -0.0010419, 0.0011174

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009933, upper bound: 0.0010932
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009800, upper bound: 0.0010545
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9905601, 0.9929131, 0.9907058, 0.9929794, -0.0016638, 0.0014328
1: -0.0036161, -0.0030298, -0.0035798, -0.0030133, -0.0004146, 0.0003570
2: 0.0060026, 0.0091096, 0.0059148, 0.0089172, -0.0018920, 0.0021971
3: -0.0054194, -0.0040052, -0.0053318, -0.0039653, -0.0010000, 0.0008611
4: 0.0016897, 0.0022910, 0.0016727, 0.0022538, -0.0003662, 0.0004252
5: 0.0065091, 0.0104169, 0.0063988, 0.0101749, -0.0023796, 0.0027634
6: -0.0011031, -0.0001113, -0.0010417, -0.0000832, -0.0007014, 0.0006040
7: -0.0059917, -0.0034255, -0.0058327, -0.0033530, -0.0018147, 0.0015627
8: -0.0027151, -0.0013656, -0.0026315, -0.0013275, -0.0009543, 0.0008218
9: -0.0002804, 0.0012844, -0.0003246, 0.0011875, -0.0009529, 0.0011066

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010569, upper bound: 0.0011181
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011075
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9905165, 0.9929214, 0.9907061, 0.9929743, -0.0017229, 0.0014511
1: -0.0036270, -0.0030277, -0.0035797, -0.0030146, -0.0004293, 0.0003616
2: 0.0059915, 0.0091671, 0.0059216, 0.0089168, -0.0019161, 0.0022751
3: -0.0054456, -0.0040002, -0.0053316, -0.0039684, -0.0010355, 0.0008721
4: 0.0016875, 0.0023022, 0.0016740, 0.0022537, -0.0003709, 0.0004403
5: 0.0064951, 0.0104892, 0.0064073, 0.0101744, -0.0024100, 0.0028615
6: -0.0011214, -0.0001077, -0.0010415, -0.0000854, -0.0007263, 0.0006117
7: -0.0060391, -0.0034163, -0.0058324, -0.0033586, -0.0018791, 0.0015826
8: -0.0027401, -0.0013607, -0.0026314, -0.0013304, -0.0009882, 0.0008323
9: -0.0002860, 0.0013134, -0.0003212, 0.0011873, -0.0009651, 0.0011459

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010569, upper bound: 0.0011326
time: 1.57 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011202
time: 1.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9905601, 0.9929131, 0.9905007, 0.9928694, -0.0014226, 0.0015251
1: -0.0036161, -0.0030298, -0.0036309, -0.0030407, -0.0003545, 0.0003800
2: 0.0060026, 0.0091096, 0.0060602, 0.0091880, -0.0020139, 0.0018785
3: -0.0054194, -0.0040052, -0.0054551, -0.0040315, -0.0008550, 0.0009167
4: 0.0016897, 0.0022910, 0.0017008, 0.0023062, -0.0003898, 0.0003636
5: 0.0065091, 0.0104169, 0.0065816, 0.0105155, -0.0025330, 0.0023626
6: -0.0011031, -0.0001113, -0.0011281, -0.0001296, -0.0005997, 0.0006429
7: -0.0059917, -0.0034255, -0.0060564, -0.0034731, -0.0015515, 0.0016634
8: -0.0027151, -0.0013656, -0.0027492, -0.0013906, -0.0008159, 0.0008748
9: -0.0002804, 0.0012844, -0.0002514, 0.0013239, -0.0010143, 0.0009461

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009923, upper bound: 0.0010812
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009810, upper bound: 0.0010454
time: 1.43 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9905165, 0.9929214, 0.9905010, 0.9928648, -0.0015009, 0.0015456
1: -0.0036270, -0.0030277, -0.0036308, -0.0030419, -0.0003740, 0.0003851
2: 0.0059915, 0.0091671, 0.0060663, 0.0091876, -0.0020410, 0.0019819
3: -0.0054456, -0.0040002, -0.0054549, -0.0040342, -0.0009021, 0.0009290
4: 0.0016875, 0.0023022, 0.0017020, 0.0023061, -0.0003950, 0.0003836
5: 0.0064951, 0.0104892, 0.0065892, 0.0105150, -0.0025670, 0.0024927
6: -0.0011214, -0.0001077, -0.0011280, -0.0001316, -0.0006327, 0.0006515
7: -0.0060391, -0.0034163, -0.0060561, -0.0034781, -0.0016369, 0.0016857
8: -0.0027401, -0.0013607, -0.0027490, -0.0013932, -0.0008609, 0.0008865
9: -0.0002860, 0.0013134, -0.0002483, 0.0013237, -0.0010279, 0.0009982

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009921, upper bound: 0.0010932
time: 1.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009795, upper bound: 0.0010545
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9905601, 0.9929131, 0.9905552, 0.9929739, -0.0014358, 0.0013779
1: -0.0036161, -0.0030298, -0.0036174, -0.0030147, -0.0003578, 0.0003433
2: 0.0060026, 0.0091096, 0.0059223, 0.0091161, -0.0018195, 0.0018959
3: -0.0054194, -0.0040052, -0.0054224, -0.0039687, -0.0008629, 0.0008281
4: 0.0016897, 0.0022910, 0.0016741, 0.0022923, -0.0003522, 0.0003670
5: 0.0065091, 0.0104169, 0.0064081, 0.0104251, -0.0022884, 0.0023846
6: -0.0011031, -0.0001113, -0.0011052, -0.0000856, -0.0006052, 0.0005808
7: -0.0059917, -0.0034255, -0.0059971, -0.0033591, -0.0015659, 0.0015028
8: -0.0027151, -0.0013656, -0.0027179, -0.0013307, -0.0008235, 0.0007903
9: -0.0002804, 0.0012844, -0.0003209, 0.0012877, -0.0009164, 0.0009549

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010569, upper bound: 0.0011181
time: 1.57 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011075
time: 1.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9905165, 0.9929214, 0.9905554, 0.9929693, -0.0015183, 0.0013992
1: -0.0036270, -0.0030277, -0.0036173, -0.0030158, -0.0003783, 0.0003486
2: 0.0059915, 0.0091671, 0.0059282, 0.0091157, -0.0018476, 0.0020049
3: -0.0054456, -0.0040002, -0.0054222, -0.0039714, -0.0009126, 0.0008409
4: 0.0016875, 0.0023022, 0.0016753, 0.0022922, -0.0003576, 0.0003880
5: 0.0064951, 0.0104892, 0.0064156, 0.0104247, -0.0023238, 0.0025217
6: -0.0011214, -0.0001077, -0.0011051, -0.0000875, -0.0006400, 0.0005898
7: -0.0060391, -0.0034163, -0.0059967, -0.0033641, -0.0016559, 0.0015260
8: -0.0027401, -0.0013607, -0.0027178, -0.0013333, -0.0008708, 0.0008025
9: -0.0002860, 0.0013134, -0.0003178, 0.0012875, -0.0009305, 0.0010098

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010569, upper bound: 0.0011327
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011202
time: 1.45 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.96 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009992, upper bound: 0.0010086
NS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009889, upper bound: 0.0009776
NS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009987, upper bound: 0.0010199
NS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009862, upper bound: 0.0009862
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010311, upper bound: 0.0010086
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010291, upper bound: 0.0009782
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010311, upper bound: 0.0010199
NS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010283, upper bound: 0.0009873
NS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009995, upper bound: 0.0010449
NS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009893, upper bound: 0.0010219
NS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009991, upper bound: 0.0010539
NS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009873, upper bound: 0.0010283
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010578, upper bound: 0.0010722
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010659, upper bound: 0.0010686
NS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010578, upper bound: 0.0010827
NS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010659, upper bound: 0.0010780
NS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010104, upper bound: 0.0010373
NS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010031, upper bound: 0.0010144
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010099, upper bound: 0.0010460
NS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010011, upper bound: 0.0010206
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010682
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010651
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010781
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010743
NS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009927, upper bound: 0.0010368
NS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009798, upper bound: 0.0009922
NS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009923, upper bound: 0.0010440
NS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009772, upper bound: 0.0009980
NS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010257, upper bound: 0.0010368
NS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010211, upper bound: 0.0009948
NS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010257, upper bound: 0.0010442
NS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010206, upper bound: 0.0010011
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009935, upper bound: 0.0010813
NS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009813, upper bound: 0.0010454
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009933, upper bound: 0.0010932
NS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009800, upper bound: 0.0010545
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010569, upper bound: 0.0011181
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011075
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010569, upper bound: 0.0011326
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011202
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009923, upper bound: 0.0010812
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009810, upper bound: 0.0010454
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009921, upper bound: 0.0010932
NS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0009795, upper bound: 0.0010545
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010569, upper bound: 0.0011181
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011075
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010569, upper bound: 0.0011327
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011202

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9907106, 0.9929209, 0.9905588, 0.9929256, -0.0014385, 0.0015976
1: -0.0035786, -0.0030279, -0.0036164, -0.0030267, -0.0003584, 0.0003981
2: 0.0059921, 0.0089109, 0.0059858, 0.0091112, -0.0021096, 0.0018995
3: -0.0053290, -0.0040005, -0.0054201, -0.0039976, -0.0008646, 0.0009602
4: 0.0016877, 0.0022526, 0.0016864, 0.0022913, -0.0004083, 0.0003677
5: 0.0064960, 0.0101670, 0.0064881, 0.0104189, -0.0026533, 0.0023891
6: -0.0010397, -0.0001079, -0.0011036, -0.0001059, -0.0006064, 0.0006734
7: -0.0058276, -0.0034168, -0.0059930, -0.0034116, -0.0015689, 0.0017424
8: -0.0026288, -0.0013610, -0.0027158, -0.0013583, -0.0008251, 0.0009163
9: -0.0002857, 0.0011844, -0.0002888, 0.0012852, -0.0010625, 0.0009567

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010521
time: 1.47 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010651
time: 1.23 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9907128, 0.9928993, 0.9903901, 0.9928768, -0.0014716, 0.0017824
1: -0.0035781, -0.0030333, -0.0036585, -0.0030389, -0.0003667, 0.0004441
2: 0.0060207, 0.0089079, 0.0060504, 0.0093341, -0.0023536, 0.0019433
3: -0.0053276, -0.0040135, -0.0055216, -0.0040270, -0.0008845, 0.0010713
4: 0.0016932, 0.0022520, 0.0016989, 0.0023345, -0.0004555, 0.0003761
5: 0.0065319, 0.0101632, 0.0065692, 0.0106993, -0.0029602, 0.0024441
6: -0.0010387, -0.0001170, -0.0011748, -0.0001265, -0.0006203, 0.0007513
7: -0.0058251, -0.0034405, -0.0061771, -0.0034649, -0.0016050, 0.0019439
8: -0.0026275, -0.0013735, -0.0028126, -0.0013863, -0.0008441, 0.0010223
9: -0.0002713, 0.0011829, -0.0002563, 0.0013975, -0.0011854, 0.0009787

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010521
time: 1.23 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010651
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9906732, 0.9929241, 0.9905592, 0.9929214, -0.0015198, 0.0016175
1: -0.0035879, -0.0030271, -0.0036164, -0.0030278, -0.0003787, 0.0004030
2: 0.0059879, 0.0089602, 0.0059916, 0.0091108, -0.0021359, 0.0020069
3: -0.0053514, -0.0039986, -0.0054199, -0.0040003, -0.0009135, 0.0009722
4: 0.0016868, 0.0022621, 0.0016876, 0.0022913, -0.0004134, 0.0003884
5: 0.0064907, 0.0102290, 0.0064954, 0.0104184, -0.0026864, 0.0025242
6: -0.0010554, -0.0001066, -0.0011035, -0.0001078, -0.0006407, 0.0006818
7: -0.0058683, -0.0034133, -0.0059926, -0.0034164, -0.0016576, 0.0017641
8: -0.0026502, -0.0013592, -0.0027156, -0.0013608, -0.0008717, 0.0009277
9: -0.0002878, 0.0012092, -0.0002859, 0.0012850, -0.0010757, 0.0010108

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010670
time: 1.42 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010743
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9906756, 0.9929016, 0.9903904, 0.9928719, -0.0015530, 0.0018023
1: -0.0035873, -0.0030327, -0.0036584, -0.0030401, -0.0003870, 0.0004491
2: 0.0060177, 0.0089571, 0.0060568, 0.0093337, -0.0023799, 0.0020507
3: -0.0053500, -0.0040121, -0.0055214, -0.0040299, -0.0009334, 0.0010832
4: 0.0016926, 0.0022615, 0.0017002, 0.0023344, -0.0004606, 0.0003969
5: 0.0065282, 0.0102251, 0.0065773, 0.0106988, -0.0029933, 0.0025793
6: -0.0010544, -0.0001161, -0.0011746, -0.0001285, -0.0006547, 0.0007597
7: -0.0058657, -0.0034380, -0.0061768, -0.0034702, -0.0016938, 0.0019657
8: -0.0026489, -0.0013721, -0.0028125, -0.0013891, -0.0008907, 0.0010337
9: -0.0002728, 0.0012076, -0.0002531, 0.0013973, -0.0011987, 0.0010329

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010670
time: 1.43 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010743
time: 1.40 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905601, 0.9929131, 0.9907096, 0.9929338, -0.0016199, 0.0014288
1: -0.0036161, -0.0030298, -0.0035789, -0.0030247, -0.0004036, 0.0003560
2: 0.0060026, 0.0091096, 0.0059751, 0.0089122, -0.0018867, 0.0021390
3: -0.0054194, -0.0040052, -0.0053295, -0.0039927, -0.0009736, 0.0008588
4: 0.0016897, 0.0022910, 0.0016844, 0.0022528, -0.0003652, 0.0004140
5: 0.0065091, 0.0104169, 0.0064746, 0.0101686, -0.0023730, 0.0026904
6: -0.0011031, -0.0001113, -0.0010401, -0.0001025, -0.0006828, 0.0006023
7: -0.0059917, -0.0034255, -0.0058286, -0.0034028, -0.0017667, 0.0015583
8: -0.0027151, -0.0013656, -0.0026293, -0.0013536, -0.0009291, 0.0008195
9: -0.0002804, 0.0012844, -0.0002942, 0.0011850, -0.0009503, 0.0010773

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0010917
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011075
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905623, 0.9928926, 0.9905592, 0.9928713, -0.0016562, 0.0016257
1: -0.0036156, -0.0030349, -0.0036164, -0.0030403, -0.0004127, 0.0004051
2: 0.0060296, 0.0091067, 0.0060578, 0.0091109, -0.0021467, 0.0021870
3: -0.0054181, -0.0040175, -0.0054200, -0.0040304, -0.0009954, 0.0009771
4: 0.0016949, 0.0022905, 0.0017004, 0.0022913, -0.0004155, 0.0004233
5: 0.0065430, 0.0104132, 0.0065785, 0.0104185, -0.0027000, 0.0027507
6: -0.0011022, -0.0001199, -0.0011035, -0.0001289, -0.0006982, 0.0006853
7: -0.0059893, -0.0034478, -0.0059927, -0.0034710, -0.0018064, 0.0017731
8: -0.0027138, -0.0013773, -0.0027157, -0.0013895, -0.0009499, 0.0009324
9: -0.0002668, 0.0012830, -0.0002526, 0.0012851, -0.0010812, 0.0011015

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0010917
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011075
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9905165, 0.9929214, 0.9907099, 0.9929288, -0.0016786, 0.0014471
1: -0.0036270, -0.0030277, -0.0035788, -0.0030259, -0.0004183, 0.0003606
2: 0.0059915, 0.0091671, 0.0059816, 0.0089118, -0.0019109, 0.0022166
3: -0.0054456, -0.0040002, -0.0053294, -0.0039957, -0.0010089, 0.0008698
4: 0.0016875, 0.0023022, 0.0016856, 0.0022527, -0.0003699, 0.0004290
5: 0.0064951, 0.0104892, 0.0064828, 0.0101681, -0.0024034, 0.0027879
6: -0.0011214, -0.0001077, -0.0010399, -0.0001046, -0.0007076, 0.0006100
7: -0.0060391, -0.0034163, -0.0058283, -0.0034082, -0.0018308, 0.0015783
8: -0.0027401, -0.0013607, -0.0026292, -0.0013565, -0.0009628, 0.0008300
9: -0.0002860, 0.0013134, -0.0002910, 0.0011848, -0.0009624, 0.0011164

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011076
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011202
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9905187, 0.9929004, 0.9905595, 0.9928657, -0.0017149, 0.0016442
1: -0.0036264, -0.0030330, -0.0036163, -0.0030416, -0.0004273, 0.0004097
2: 0.0060191, 0.0091642, 0.0060650, 0.0091104, -0.0021712, 0.0022645
3: -0.0054443, -0.0040128, -0.0054198, -0.0040336, -0.0010307, 0.0009882
4: 0.0016929, 0.0023016, 0.0017018, 0.0022912, -0.0004202, 0.0004383
5: 0.0065300, 0.0104856, 0.0065876, 0.0104180, -0.0027308, 0.0028482
6: -0.0011205, -0.0001165, -0.0011034, -0.0001312, -0.0007229, 0.0006931
7: -0.0060368, -0.0034391, -0.0059924, -0.0034770, -0.0018704, 0.0017933
8: -0.0027388, -0.0013728, -0.0027155, -0.0013927, -0.0009836, 0.0009431
9: -0.0002721, 0.0013119, -0.0002490, 0.0012849, -0.0010935, 0.0011405

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011076
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011202
time: 1.22 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905601, 0.9929131, 0.9905588, 0.9929256, -0.0013958, 0.0013737
1: -0.0036161, -0.0030298, -0.0036164, -0.0030267, -0.0003478, 0.0003423
2: 0.0060026, 0.0091096, 0.0059858, 0.0091112, -0.0018140, 0.0018431
3: -0.0054194, -0.0040052, -0.0054201, -0.0039976, -0.0008389, 0.0008256
4: 0.0016897, 0.0022910, 0.0016864, 0.0022913, -0.0003511, 0.0003567
5: 0.0065091, 0.0104169, 0.0064881, 0.0104189, -0.0022815, 0.0023182
6: -0.0011031, -0.0001113, -0.0011036, -0.0001059, -0.0005884, 0.0005791
7: -0.0059917, -0.0034255, -0.0059930, -0.0034116, -0.0015223, 0.0014982
8: -0.0027151, -0.0013656, -0.0027158, -0.0013583, -0.0008006, 0.0007879
9: -0.0002804, 0.0012844, -0.0002888, 0.0012852, -0.0009136, 0.0009283

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0010917
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011075
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905623, 0.9928926, 0.9903901, 0.9928768, -0.0014433, 0.0015673
1: -0.0036156, -0.0030349, -0.0036585, -0.0030389, -0.0003596, 0.0003905
2: 0.0060296, 0.0091067, 0.0060504, 0.0093341, -0.0020696, 0.0019058
3: -0.0054181, -0.0040175, -0.0055216, -0.0040270, -0.0008674, 0.0009420
4: 0.0016949, 0.0022905, 0.0016989, 0.0023345, -0.0004006, 0.0003689
5: 0.0065430, 0.0104132, 0.0065692, 0.0106993, -0.0026031, 0.0023970
6: -0.0011022, -0.0001199, -0.0011748, -0.0001265, -0.0006084, 0.0006607
7: -0.0059893, -0.0034478, -0.0061771, -0.0034649, -0.0015741, 0.0017094
8: -0.0027138, -0.0013773, -0.0028126, -0.0013863, -0.0008278, 0.0008990
9: -0.0002668, 0.0012830, -0.0002563, 0.0013975, -0.0010424, 0.0009599

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0010917
time: 1.36 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011075
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9905165, 0.9929214, 0.9905592, 0.9929214, -0.0014786, 0.0013950
1: -0.0036270, -0.0030277, -0.0036164, -0.0030278, -0.0003684, 0.0003476
2: 0.0059915, 0.0091671, 0.0059916, 0.0091108, -0.0018421, 0.0019525
3: -0.0054456, -0.0040002, -0.0054199, -0.0040003, -0.0008887, 0.0008384
4: 0.0016875, 0.0023022, 0.0016876, 0.0022913, -0.0003565, 0.0003779
5: 0.0064951, 0.0104892, 0.0064954, 0.0104184, -0.0023169, 0.0024557
6: -0.0011214, -0.0001077, -0.0011035, -0.0001078, -0.0006233, 0.0005880
7: -0.0060391, -0.0034163, -0.0059926, -0.0034164, -0.0016126, 0.0015215
8: -0.0027401, -0.0013607, -0.0027156, -0.0013608, -0.0008481, 0.0008001
9: -0.0002860, 0.0013134, -0.0002859, 0.0012850, -0.0009278, 0.0009834

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011076
time: 1.37 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011202
time: 1.36 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9905187, 0.9929004, 0.9903904, 0.9928719, -0.0015262, 0.0015887
1: -0.0036264, -0.0030330, -0.0036584, -0.0030401, -0.0003803, 0.0003959
2: 0.0060191, 0.0091642, 0.0060568, 0.0093337, -0.0020978, 0.0020153
3: -0.0054443, -0.0040128, -0.0055214, -0.0040299, -0.0009173, 0.0009548
4: 0.0016929, 0.0023016, 0.0017002, 0.0023344, -0.0004060, 0.0003901
5: 0.0065300, 0.0104856, 0.0065773, 0.0106988, -0.0026385, 0.0025347
6: -0.0011205, -0.0001165, -0.0011746, -0.0001285, -0.0006433, 0.0006697
7: -0.0060368, -0.0034391, -0.0061768, -0.0034702, -0.0016645, 0.0017327
8: -0.0027388, -0.0013728, -0.0028125, -0.0013891, -0.0008754, 0.0009112
9: -0.0002721, 0.0013119, -0.0002531, 0.0013973, -0.0010566, 0.0010150

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011076
time: 1.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011202
time: 1.25 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.81 seconds
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010521
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010651
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010521
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010651
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010670
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010964, upper bound: 0.0010743
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010670
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0011066, upper bound: 0.0010743
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0010917
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011075
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0010917
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011075
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011076
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011202
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011076
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011202
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0010917
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011075
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0010917
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011075
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011076
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010565, upper bound: 0.0011202
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011076
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 0, lower bound: -0.0010635, upper bound: 0.0011202

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9907144, 0.9928743, 0.9905588, 0.9929256, -0.0014347, 0.0015542
1: -0.0035776, -0.0030395, -0.0036164, -0.0030267, -0.0003575, 0.0003873
2: 0.0060536, 0.0089056, 0.0059858, 0.0091112, -0.0020522, 0.0018945
3: -0.0053266, -0.0040285, -0.0054201, -0.0039976, -0.0008623, 0.0009341
4: 0.0016996, 0.0022516, 0.0016864, 0.0022913, -0.0003972, 0.0003667
5: 0.0065733, 0.0101604, 0.0064881, 0.0104189, -0.0025812, 0.0023827
6: -0.0010380, -0.0001275, -0.0011036, -0.0001059, -0.0006048, 0.0006551
7: -0.0058232, -0.0034676, -0.0059930, -0.0034116, -0.0015647, 0.0016950
8: -0.0026265, -0.0013877, -0.0027158, -0.0013583, -0.0008229, 0.0008914
9: -0.0002547, 0.0011817, -0.0002888, 0.0012852, -0.0010336, 0.0009541

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010657
time: 1.29 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010657
time: 1.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9905645, 0.9928139, 0.9905588, 0.9929256, -0.0016513, 0.0015454
1: -0.0036150, -0.0030546, -0.0036164, -0.0030267, -0.0004115, 0.0003851
2: 0.0061335, 0.0091038, 0.0059858, 0.0091112, -0.0020406, 0.0021805
3: -0.0054168, -0.0040648, -0.0054201, -0.0039976, -0.0009925, 0.0009288
4: 0.0017150, 0.0022899, 0.0016864, 0.0022913, -0.0003950, 0.0004220
5: 0.0066738, 0.0104096, 0.0064881, 0.0104189, -0.0025666, 0.0027425
6: -0.0011012, -0.0001531, -0.0011036, -0.0001059, -0.0006961, 0.0006514
7: -0.0059869, -0.0035336, -0.0059930, -0.0034116, -0.0018010, 0.0016854
8: -0.0027126, -0.0014224, -0.0027158, -0.0013583, -0.0009471, 0.0008864
9: -0.0002145, 0.0012815, -0.0002888, 0.0012852, -0.0010278, 0.0010982

Time for backsubstitution: 1.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010682
time: 1.30 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010682
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9907144, 0.9928743, 0.9903901, 0.9928768, -0.0014111, 0.0017562
1: -0.0035776, -0.0030395, -0.0036585, -0.0030389, -0.0003516, 0.0004376
2: 0.0060536, 0.0089056, 0.0060504, 0.0093341, -0.0023190, 0.0018634
3: -0.0053266, -0.0040285, -0.0055216, -0.0040270, -0.0008481, 0.0010555
4: 0.0016996, 0.0022516, 0.0016989, 0.0023345, -0.0004488, 0.0003607
5: 0.0065733, 0.0101604, 0.0065692, 0.0106993, -0.0029167, 0.0023436
6: -0.0010380, -0.0001275, -0.0011748, -0.0001265, -0.0005948, 0.0007403
7: -0.0058232, -0.0034676, -0.0061771, -0.0034649, -0.0015390, 0.0019154
8: -0.0026265, -0.0013877, -0.0028126, -0.0013863, -0.0008094, 0.0010073
9: -0.0002547, 0.0011817, -0.0002563, 0.0013975, -0.0011680, 0.0009385

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010521
time: 1.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010521
time: 1.49 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9905645, 0.9928139, 0.9903901, 0.9928768, -0.0014808, 0.0016107
1: -0.0036150, -0.0030546, -0.0036585, -0.0030389, -0.0003690, 0.0004013
2: 0.0061335, 0.0091038, 0.0060504, 0.0093341, -0.0021268, 0.0019554
3: -0.0054168, -0.0040648, -0.0055216, -0.0040270, -0.0008900, 0.0009680
4: 0.0017150, 0.0022899, 0.0016989, 0.0023345, -0.0004116, 0.0003785
5: 0.0066738, 0.0104096, 0.0065692, 0.0106993, -0.0026750, 0.0024594
6: -0.0011012, -0.0001531, -0.0011748, -0.0001265, -0.0006242, 0.0006789
7: -0.0059869, -0.0035336, -0.0061771, -0.0034649, -0.0016150, 0.0017566
8: -0.0027126, -0.0014224, -0.0028126, -0.0013863, -0.0008493, 0.0009238
9: -0.0002145, 0.0012815, -0.0002563, 0.0013975, -0.0010712, 0.0009848

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010651
time: 1.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010651
time: 1.61 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9906772, 0.9928792, 0.9905592, 0.9929214, -0.0015164, 0.0015738
1: -0.0035869, -0.0030383, -0.0036164, -0.0030278, -0.0003778, 0.0003921
2: 0.0060472, 0.0089549, 0.0059916, 0.0091108, -0.0020782, 0.0020023
3: -0.0053490, -0.0040256, -0.0054199, -0.0040003, -0.0009114, 0.0009459
4: 0.0016983, 0.0022611, 0.0016876, 0.0022913, -0.0004022, 0.0003875
5: 0.0065653, 0.0102224, 0.0064954, 0.0104184, -0.0026138, 0.0025184
6: -0.0010537, -0.0001255, -0.0011035, -0.0001078, -0.0006392, 0.0006634
7: -0.0058639, -0.0034623, -0.0059926, -0.0034164, -0.0016538, 0.0017165
8: -0.0026479, -0.0013850, -0.0027156, -0.0013608, -0.0008697, 0.0009027
9: -0.0002579, 0.0012065, -0.0002859, 0.0012850, -0.0010467, 0.0010085

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010769
time: 1.41 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010769
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9905291, 0.9928138, 0.9905592, 0.9929214, -0.0017050, 0.0015650
1: -0.0036239, -0.0030546, -0.0036164, -0.0030278, -0.0004248, 0.0003899
2: 0.0061336, 0.0091506, 0.0059916, 0.0091108, -0.0020665, 0.0022514
3: -0.0054381, -0.0040649, -0.0054199, -0.0040003, -0.0010247, 0.0009406
4: 0.0017150, 0.0022990, 0.0016876, 0.0022913, -0.0004000, 0.0004358
5: 0.0066739, 0.0104684, 0.0064954, 0.0104184, -0.0025991, 0.0028317
6: -0.0011162, -0.0001531, -0.0011035, -0.0001078, -0.0007187, 0.0006597
7: -0.0060255, -0.0035337, -0.0059926, -0.0034164, -0.0018595, 0.0017068
8: -0.0027329, -0.0014225, -0.0027156, -0.0013608, -0.0009779, 0.0008976
9: -0.0002144, 0.0013051, -0.0002859, 0.0012850, -0.0010408, 0.0011339

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010781
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010781
time: 1.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9906772, 0.9928792, 0.9903904, 0.9928719, -0.0014911, 0.0017760
1: -0.0035869, -0.0030383, -0.0036584, -0.0030401, -0.0003715, 0.0004425
2: 0.0060472, 0.0089549, 0.0060568, 0.0093337, -0.0023452, 0.0019690
3: -0.0053490, -0.0040256, -0.0055214, -0.0040299, -0.0008962, 0.0010674
4: 0.0016983, 0.0022611, 0.0017002, 0.0023344, -0.0004539, 0.0003811
5: 0.0065653, 0.0102224, 0.0065773, 0.0106988, -0.0029497, 0.0024765
6: -0.0010537, -0.0001255, -0.0011746, -0.0001285, -0.0006286, 0.0007487
7: -0.0058639, -0.0034623, -0.0061768, -0.0034702, -0.0016263, 0.0019370
8: -0.0026479, -0.0013850, -0.0028125, -0.0013891, -0.0008552, 0.0010187
9: -0.0002579, 0.0012065, -0.0002531, 0.0013973, -0.0011812, 0.0009917

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010670
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010670
time: 1.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9905291, 0.9928138, 0.9903904, 0.9928719, -0.0015666, 0.0016262
1: -0.0036239, -0.0030546, -0.0036584, -0.0030401, -0.0003904, 0.0004052
2: 0.0061336, 0.0091506, 0.0060568, 0.0093337, -0.0021474, 0.0020687
3: -0.0054381, -0.0040649, -0.0055214, -0.0040299, -0.0009416, 0.0009774
4: 0.0017150, 0.0022990, 0.0017002, 0.0023344, -0.0004156, 0.0004004
5: 0.0066739, 0.0104684, 0.0065773, 0.0106988, -0.0027009, 0.0026019
6: -0.0011162, -0.0001531, -0.0011746, -0.0001285, -0.0006604, 0.0006855
7: -0.0060255, -0.0035337, -0.0061768, -0.0034702, -0.0017086, 0.0017736
8: -0.0027329, -0.0014225, -0.0028125, -0.0013891, -0.0008986, 0.0009327
9: -0.0002144, 0.0013051, -0.0002531, 0.0013973, -0.0010816, 0.0010419

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010743
time: 1.39 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010743
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9907096, 0.9929338, -0.0018201, 0.0013581
1: -0.0036572, -0.0030540, -0.0035789, -0.0030247, -0.0004535, 0.0003384
2: 0.0061307, 0.0093271, 0.0059751, 0.0089122, -0.0017933, 0.0024034
3: -0.0055184, -0.0040635, -0.0053295, -0.0039927, -0.0010939, 0.0008162
4: 0.0017145, 0.0023331, 0.0016844, 0.0022528, -0.0003471, 0.0004652
5: 0.0066702, 0.0106905, 0.0064746, 0.0101686, -0.0022555, 0.0030229
6: -0.0011725, -0.0001521, -0.0010401, -0.0001025, -0.0007672, 0.0005725
7: -0.0061713, -0.0035313, -0.0058286, -0.0034028, -0.0019851, 0.0014812
8: -0.0028096, -0.0014212, -0.0026293, -0.0013536, -0.0010439, 0.0007789
9: -0.0002159, 0.0013940, -0.0002942, 0.0011850, -0.0009032, 0.0012105

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011181
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011181
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9905592, 0.9928713, -0.0016743, 0.0014226
1: -0.0036572, -0.0030540, -0.0036164, -0.0030403, -0.0004172, 0.0003545
2: 0.0061307, 0.0093271, 0.0060578, 0.0091109, -0.0018785, 0.0022109
3: -0.0055184, -0.0040635, -0.0054200, -0.0040304, -0.0010063, 0.0008550
4: 0.0017145, 0.0023331, 0.0017004, 0.0022913, -0.0003636, 0.0004279
5: 0.0066702, 0.0106905, 0.0065785, 0.0104185, -0.0023626, 0.0027808
6: -0.0011725, -0.0001521, -0.0011035, -0.0001289, -0.0007058, 0.0005997
7: -0.0061713, -0.0035313, -0.0059927, -0.0034710, -0.0018261, 0.0015515
8: -0.0028096, -0.0014212, -0.0027157, -0.0013895, -0.0009603, 0.0008159
9: -0.0002159, 0.0013940, -0.0002526, 0.0012851, -0.0009461, 0.0011135

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011075
time: 1.41 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011075
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9907099, 0.9929288, -0.0016753, 0.0013958
1: -0.0036260, -0.0030396, -0.0035788, -0.0030259, -0.0004175, 0.0003478
2: 0.0060545, 0.0091619, 0.0059816, 0.0089118, -0.0018431, 0.0022123
3: -0.0054432, -0.0040288, -0.0053294, -0.0039957, -0.0010069, 0.0008389
4: 0.0016997, 0.0023012, 0.0016856, 0.0022527, -0.0003567, 0.0004282
5: 0.0065744, 0.0104828, 0.0064828, 0.0101681, -0.0023181, 0.0027825
6: -0.0011198, -0.0001278, -0.0010399, -0.0001046, -0.0007062, 0.0005884
7: -0.0060349, -0.0034683, -0.0058283, -0.0034082, -0.0018272, 0.0015223
8: -0.0027378, -0.0013881, -0.0026292, -0.0013565, -0.0009609, 0.0008006
9: -0.0002543, 0.0013108, -0.0002910, 0.0011848, -0.0009283, 0.0011142

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011299
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011299
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9907099, 0.9929288, -0.0018704, 0.0013764
1: -0.0036673, -0.0030525, -0.0035788, -0.0030259, -0.0004660, 0.0003430
2: 0.0061225, 0.0093805, 0.0059816, 0.0089118, -0.0018175, 0.0024698
3: -0.0055427, -0.0040598, -0.0053294, -0.0039957, -0.0011241, 0.0008273
4: 0.0017129, 0.0023435, 0.0016856, 0.0022527, -0.0003518, 0.0004780
5: 0.0066600, 0.0107577, 0.0064828, 0.0101681, -0.0022860, 0.0031064
6: -0.0011896, -0.0001495, -0.0010399, -0.0001046, -0.0007884, 0.0005802
7: -0.0062154, -0.0035245, -0.0058283, -0.0034082, -0.0020399, 0.0015012
8: -0.0028328, -0.0014177, -0.0026292, -0.0013565, -0.0010728, 0.0007895
9: -0.0002200, 0.0014209, -0.0002910, 0.0011848, -0.0009154, 0.0012439

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011327
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011326
time: 1.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9905595, 0.9928657, -0.0016567, 0.0016102
1: -0.0036260, -0.0030396, -0.0036163, -0.0030416, -0.0004128, 0.0004012
2: 0.0060545, 0.0091619, 0.0060650, 0.0091104, -0.0021262, 0.0021877
3: -0.0054432, -0.0040288, -0.0054198, -0.0040336, -0.0009957, 0.0009677
4: 0.0016997, 0.0023012, 0.0017018, 0.0022912, -0.0004115, 0.0004234
5: 0.0065744, 0.0104828, 0.0065876, 0.0104180, -0.0026742, 0.0027515
6: -0.0011198, -0.0001278, -0.0011034, -0.0001312, -0.0006984, 0.0006787
7: -0.0060349, -0.0034683, -0.0059924, -0.0034770, -0.0018069, 0.0017561
8: -0.0027378, -0.0013881, -0.0027155, -0.0013927, -0.0009502, 0.0009235
9: -0.0002543, 0.0013108, -0.0002490, 0.0012849, -0.0010709, 0.0011018

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011076
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011076
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9905595, 0.9928657, -0.0017332, 0.0014366
1: -0.0036673, -0.0030525, -0.0036163, -0.0030416, -0.0004319, 0.0003580
2: 0.0061225, 0.0093805, 0.0060650, 0.0091104, -0.0018970, 0.0022886
3: -0.0055427, -0.0040598, -0.0054198, -0.0040336, -0.0010417, 0.0008634
4: 0.0017129, 0.0023435, 0.0017018, 0.0022912, -0.0003672, 0.0004430
5: 0.0066600, 0.0107577, 0.0065876, 0.0104180, -0.0023860, 0.0028785
6: -0.0011896, -0.0001495, -0.0011034, -0.0001312, -0.0007306, 0.0006056
7: -0.0062154, -0.0035245, -0.0059924, -0.0034770, -0.0018903, 0.0015668
8: -0.0028328, -0.0014177, -0.0027155, -0.0013927, -0.0009941, 0.0008240
9: -0.0002200, 0.0014209, -0.0002490, 0.0012849, -0.0009554, 0.0011527

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011202
time: 1.25 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011202
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9905588, 0.9929256, -0.0016076, 0.0013136
1: -0.0036572, -0.0030540, -0.0036164, -0.0030267, -0.0004006, 0.0003273
2: 0.0061307, 0.0093271, 0.0059858, 0.0091112, -0.0017346, 0.0021229
3: -0.0055184, -0.0040635, -0.0054201, -0.0039976, -0.0009662, 0.0007895
4: 0.0017145, 0.0023331, 0.0016864, 0.0022913, -0.0003357, 0.0004109
5: 0.0066702, 0.0106905, 0.0064881, 0.0104189, -0.0021816, 0.0026700
6: -0.0011725, -0.0001521, -0.0011036, -0.0001059, -0.0006777, 0.0005537
7: -0.0061713, -0.0035313, -0.0059930, -0.0034116, -0.0017534, 0.0014326
8: -0.0028096, -0.0014212, -0.0027158, -0.0013583, -0.0009221, 0.0007534
9: -0.0002159, 0.0013940, -0.0002888, 0.0012852, -0.0008736, 0.0010692

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011181
time: 1.39 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011181
time: 1.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9903901, 0.9928768, -0.0014530, 0.0013950
1: -0.0036572, -0.0030540, -0.0036585, -0.0030389, -0.0003620, 0.0003476
2: 0.0061307, 0.0093271, 0.0060504, 0.0093341, -0.0018421, 0.0019187
3: -0.0055184, -0.0040635, -0.0055216, -0.0040270, -0.0008733, 0.0008384
4: 0.0017145, 0.0023331, 0.0016989, 0.0023345, -0.0003565, 0.0003714
5: 0.0066702, 0.0106905, 0.0065692, 0.0106993, -0.0023169, 0.0024132
6: -0.0011725, -0.0001521, -0.0011748, -0.0001265, -0.0006125, 0.0005880
7: -0.0061713, -0.0035313, -0.0061771, -0.0034649, -0.0015847, 0.0015215
8: -0.0028096, -0.0014212, -0.0028126, -0.0013863, -0.0008334, 0.0008001
9: -0.0002159, 0.0013940, -0.0002563, 0.0013975, -0.0009278, 0.0009663

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011075
time: 1.36 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011075
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9905592, 0.9929214, -0.0014749, 0.0013560
1: -0.0036260, -0.0030396, -0.0036164, -0.0030278, -0.0003675, 0.0003379
2: 0.0060545, 0.0091619, 0.0059916, 0.0091108, -0.0017906, 0.0019475
3: -0.0054432, -0.0040288, -0.0054199, -0.0040003, -0.0008864, 0.0008150
4: 0.0016997, 0.0023012, 0.0016876, 0.0022913, -0.0003466, 0.0003769
5: 0.0065744, 0.0104828, 0.0064954, 0.0104184, -0.0022521, 0.0024495
6: -0.0011198, -0.0001278, -0.0011035, -0.0001078, -0.0006217, 0.0005716
7: -0.0060349, -0.0034683, -0.0059926, -0.0034164, -0.0016085, 0.0014789
8: -0.0027378, -0.0013881, -0.0027156, -0.0013608, -0.0008459, 0.0007778
9: -0.0002543, 0.0013108, -0.0002859, 0.0012850, -0.0009019, 0.0009809

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011299
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011299
time: 1.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9905592, 0.9929214, -0.0016640, 0.0013339
1: -0.0036673, -0.0030525, -0.0036164, -0.0030278, -0.0004146, 0.0003324
2: 0.0061225, 0.0093805, 0.0059916, 0.0091108, -0.0017614, 0.0021973
3: -0.0055427, -0.0040598, -0.0054199, -0.0040003, -0.0010001, 0.0008017
4: 0.0017129, 0.0023435, 0.0016876, 0.0022913, -0.0003409, 0.0004253
5: 0.0066600, 0.0107577, 0.0064954, 0.0104184, -0.0022153, 0.0027636
6: -0.0011896, -0.0001495, -0.0011035, -0.0001078, -0.0007014, 0.0005623
7: -0.0062154, -0.0035245, -0.0059926, -0.0034164, -0.0018148, 0.0014548
8: -0.0028328, -0.0014177, -0.0027156, -0.0013608, -0.0009544, 0.0007651
9: -0.0002200, 0.0014209, -0.0002859, 0.0012850, -0.0008871, 0.0011067

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011327
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011326
time: 1.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9903904, 0.9928719, -0.0014479, 0.0015696
1: -0.0036260, -0.0030396, -0.0036584, -0.0030401, -0.0003608, 0.0003911
2: 0.0060545, 0.0091619, 0.0060568, 0.0093337, -0.0020726, 0.0019119
3: -0.0054432, -0.0040288, -0.0055214, -0.0040299, -0.0008702, 0.0009434
4: 0.0016997, 0.0023012, 0.0017002, 0.0023344, -0.0004011, 0.0003700
5: 0.0065744, 0.0104828, 0.0065773, 0.0106988, -0.0026068, 0.0024047
6: -0.0011198, -0.0001278, -0.0011746, -0.0001285, -0.0006103, 0.0006616
7: -0.0060349, -0.0034683, -0.0061768, -0.0034702, -0.0015791, 0.0017118
8: -0.0027378, -0.0013881, -0.0028125, -0.0013891, -0.0008304, 0.0009002
9: -0.0002543, 0.0013108, -0.0002531, 0.0013973, -0.0010439, 0.0009629

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011076
time: 1.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011076
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9903904, 0.9928719, -0.0015405, 0.0014132
1: -0.0036673, -0.0030525, -0.0036584, -0.0030401, -0.0003839, 0.0003521
2: 0.0061225, 0.0093805, 0.0060568, 0.0093337, -0.0018661, 0.0020343
3: -0.0055427, -0.0040598, -0.0055214, -0.0040299, -0.0009259, 0.0008494
4: 0.0017129, 0.0023435, 0.0017002, 0.0023344, -0.0003612, 0.0003937
5: 0.0066600, 0.0107577, 0.0065773, 0.0106988, -0.0023471, 0.0025586
6: -0.0011896, -0.0001495, -0.0011746, -0.0001285, -0.0006494, 0.0005957
7: -0.0062154, -0.0035245, -0.0061768, -0.0034702, -0.0016802, 0.0015413
8: -0.0028328, -0.0014177, -0.0028125, -0.0013891, -0.0008836, 0.0008106
9: -0.0002200, 0.0014209, -0.0002531, 0.0013973, -0.0009399, 0.0010246

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011202
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011202
time: 1.59 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 4.18 seconds
NS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010657
NS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010657
NS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010682
NS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010682
NS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010521
NS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010521
NS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010651
NS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010873, upper bound: 0.0010651
NS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010769
NS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010769
NS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010781
NS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010781
NS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010670
NS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010670
NS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010743
NS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010800, upper bound: 0.0010743
NS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011181
NS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011181
NS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011075
NS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011075
NS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011299
NS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011299
NS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011327
NS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011326
NS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011076
NS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011076
NS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011202
NS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011202
NS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011181
NS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011181
NS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011075
NS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010465, upper bound: 0.0011075
NS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011299
NS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011299
NS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011327
NS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011326
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011076
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011076
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011202
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.18
Output dim: 0, lower bound: -0.0010411, upper bound: 0.0011202

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9907144, 0.9928743, -0.0017481, 0.0013463
1: -0.0036572, -0.0030540, -0.0035776, -0.0030395, -0.0004356, 0.0003355
2: 0.0061307, 0.0093271, 0.0060536, 0.0089056, -0.0017777, 0.0023084
3: -0.0055184, -0.0040635, -0.0053266, -0.0040285, -0.0010507, 0.0008092
4: 0.0017145, 0.0023331, 0.0016996, 0.0022516, -0.0003441, 0.0004468
5: 0.0066702, 0.0106905, 0.0065733, 0.0101604, -0.0022359, 0.0029033
6: -0.0011725, -0.0001521, -0.0010380, -0.0001275, -0.0007369, 0.0005675
7: -0.0061713, -0.0035313, -0.0058232, -0.0034676, -0.0019066, 0.0014683
8: -0.0028096, -0.0014212, -0.0026265, -0.0013877, -0.0010026, 0.0007722
9: -0.0002159, 0.0013940, -0.0002547, 0.0011817, -0.0008954, 0.0011626

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010759
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009165, upper bound: 0.0010669
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9906772, 0.9928792, -0.0017890, 0.0014306
1: -0.0036572, -0.0030540, -0.0035869, -0.0030383, -0.0004458, 0.0003565
2: 0.0061307, 0.0093271, 0.0060472, 0.0089549, -0.0018890, 0.0023624
3: -0.0055184, -0.0040635, -0.0053490, -0.0040256, -0.0010752, 0.0008598
4: 0.0017145, 0.0023331, 0.0016983, 0.0022611, -0.0003656, 0.0004572
5: 0.0066702, 0.0106905, 0.0065653, 0.0102224, -0.0023759, 0.0029712
6: -0.0011725, -0.0001521, -0.0010537, -0.0001255, -0.0007541, 0.0006030
7: -0.0061713, -0.0035313, -0.0058639, -0.0034623, -0.0019512, 0.0015602
8: -0.0028096, -0.0014212, -0.0026479, -0.0013850, -0.0010261, 0.0008205
9: -0.0002159, 0.0013940, -0.0002579, 0.0012065, -0.0009514, 0.0011898

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010759
time: 1.76 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009165, upper bound: 0.0010669
time: 1.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9905645, 0.9928139, -0.0016024, 0.0014121
1: -0.0036572, -0.0030540, -0.0036150, -0.0030546, -0.0003993, 0.0003519
2: 0.0061307, 0.0093271, 0.0061335, 0.0091038, -0.0018647, 0.0021159
3: -0.0055184, -0.0040635, -0.0054168, -0.0040648, -0.0009631, 0.0008487
4: 0.0017145, 0.0023331, 0.0017150, 0.0022899, -0.0003609, 0.0004095
5: 0.0066702, 0.0106905, 0.0066738, 0.0104096, -0.0023453, 0.0026613
6: -0.0011725, -0.0001521, -0.0011012, -0.0001531, -0.0006755, 0.0005953
7: -0.0061713, -0.0035313, -0.0059869, -0.0035336, -0.0017476, 0.0015401
8: -0.0028096, -0.0014212, -0.0027126, -0.0014224, -0.0009191, 0.0008099
9: -0.0002159, 0.0013940, -0.0002145, 0.0012815, -0.0009391, 0.0010657

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009971, upper bound: 0.0010601
time: 1.44 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009971, upper bound: 0.0010436
time: 1.33 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9905291, 0.9928138, -0.0016433, 0.0015006
1: -0.0036572, -0.0030540, -0.0036239, -0.0030546, -0.0004095, 0.0003739
2: 0.0061307, 0.0093271, 0.0061336, 0.0091506, -0.0019815, 0.0021699
3: -0.0055184, -0.0040635, -0.0054381, -0.0040649, -0.0009877, 0.0009019
4: 0.0017145, 0.0023331, 0.0017150, 0.0022990, -0.0003835, 0.0004200
5: 0.0066702, 0.0106905, 0.0066739, 0.0104684, -0.0024922, 0.0027292
6: -0.0011725, -0.0001521, -0.0011162, -0.0001531, -0.0006927, 0.0006326
7: -0.0061713, -0.0035313, -0.0060255, -0.0035337, -0.0017922, 0.0016366
8: -0.0028096, -0.0014212, -0.0027329, -0.0014225, -0.0009425, 0.0008607
9: -0.0002159, 0.0013940, -0.0002144, 0.0013051, -0.0009980, 0.0010929

Time for backsubstitution: 1.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009971, upper bound: 0.0010601
time: 1.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009971, upper bound: 0.0010436
time: 1.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9907144, 0.9928743, -0.0016037, 0.0013884
1: -0.0036260, -0.0030396, -0.0035776, -0.0030395, -0.0003996, 0.0003459
2: 0.0060545, 0.0091619, 0.0060536, 0.0089056, -0.0018333, 0.0021177
3: -0.0054432, -0.0040288, -0.0053266, -0.0040285, -0.0009639, 0.0008344
4: 0.0016997, 0.0023012, 0.0016996, 0.0022516, -0.0003548, 0.0004099
5: 0.0065744, 0.0104828, 0.0065733, 0.0101604, -0.0023058, 0.0026635
6: -0.0011198, -0.0001278, -0.0010380, -0.0001275, -0.0006760, 0.0005852
7: -0.0060349, -0.0034683, -0.0058232, -0.0034676, -0.0017491, 0.0015142
8: -0.0027378, -0.0013881, -0.0026265, -0.0013877, -0.0009198, 0.0007963
9: -0.0002543, 0.0013108, -0.0002547, 0.0011817, -0.0009233, 0.0010666

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011078
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011052
time: 1.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9906772, 0.9928792, -0.0015726, 0.0013919
1: -0.0036260, -0.0030396, -0.0035869, -0.0030383, -0.0003918, 0.0003468
2: 0.0060545, 0.0091619, 0.0060472, 0.0089549, -0.0018380, 0.0020766
3: -0.0054432, -0.0040288, -0.0053490, -0.0040256, -0.0009452, 0.0008366
4: 0.0016997, 0.0023012, 0.0016983, 0.0022611, -0.0003558, 0.0004019
5: 0.0065744, 0.0104828, 0.0065653, 0.0102224, -0.0023118, 0.0026118
6: -0.0011198, -0.0001278, -0.0010537, -0.0001255, -0.0006629, 0.0005868
7: -0.0060349, -0.0034683, -0.0058639, -0.0034623, -0.0017151, 0.0015181
8: -0.0027378, -0.0013881, -0.0026479, -0.0013850, -0.0009020, 0.0007984
9: -0.0002543, 0.0013108, -0.0002579, 0.0012065, -0.0009257, 0.0010459

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011078
time: 1.68 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011052
time: 1.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9907144, 0.9928743, -0.0017987, 0.0013593
1: -0.0036673, -0.0030525, -0.0035776, -0.0030395, -0.0004482, 0.0003387
2: 0.0061225, 0.0093805, 0.0060536, 0.0089056, -0.0017949, 0.0023752
3: -0.0055427, -0.0040598, -0.0053266, -0.0040285, -0.0010811, 0.0008170
4: 0.0017129, 0.0023435, 0.0016996, 0.0022516, -0.0003474, 0.0004597
5: 0.0066600, 0.0107577, 0.0065733, 0.0101604, -0.0022575, 0.0029874
6: -0.0011896, -0.0001495, -0.0010380, -0.0001275, -0.0007582, 0.0005730
7: -0.0062154, -0.0035245, -0.0058232, -0.0034676, -0.0019618, 0.0014825
8: -0.0028328, -0.0014177, -0.0026265, -0.0013877, -0.0010317, 0.0007796
9: -0.0002200, 0.0014209, -0.0002547, 0.0011817, -0.0009040, 0.0011963

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009904, upper bound: 0.0010893
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009904, upper bound: 0.0010831
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9906772, 0.9928792, -0.0017764, 0.0013726
1: -0.0036673, -0.0030525, -0.0035869, -0.0030383, -0.0004426, 0.0003420
2: 0.0061225, 0.0093805, 0.0060472, 0.0089549, -0.0018125, 0.0023457
3: -0.0055427, -0.0040598, -0.0053490, -0.0040256, -0.0010677, 0.0008250
4: 0.0017129, 0.0023435, 0.0016983, 0.0022611, -0.0003508, 0.0004540
5: 0.0066600, 0.0107577, 0.0065653, 0.0102224, -0.0022796, 0.0029503
6: -0.0011896, -0.0001495, -0.0010537, -0.0001255, -0.0007488, 0.0005786
7: -0.0062154, -0.0035245, -0.0058639, -0.0034623, -0.0019374, 0.0014970
8: -0.0028328, -0.0014177, -0.0026479, -0.0013850, -0.0010189, 0.0007873
9: -0.0002200, 0.0014209, -0.0002579, 0.0012065, -0.0009129, 0.0011814

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009904, upper bound: 0.0010893
time: 1.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009904, upper bound: 0.0010831
time: 1.36 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9905645, 0.9928139, -0.0015949, 0.0016050
1: -0.0036260, -0.0030396, -0.0036150, -0.0030546, -0.0003974, 0.0003999
2: 0.0060545, 0.0091619, 0.0061335, 0.0091038, -0.0021193, 0.0021061
3: -0.0054432, -0.0040288, -0.0054168, -0.0040648, -0.0009586, 0.0009646
4: 0.0016997, 0.0023012, 0.0017150, 0.0022899, -0.0004102, 0.0004076
5: 0.0065744, 0.0104828, 0.0066738, 0.0104096, -0.0026656, 0.0026489
6: -0.0011198, -0.0001278, -0.0011012, -0.0001531, -0.0006723, 0.0006766
7: -0.0060349, -0.0034683, -0.0059869, -0.0035336, -0.0017395, 0.0017504
8: -0.0027378, -0.0013881, -0.0027126, -0.0014224, -0.0009148, 0.0009205
9: -0.0002543, 0.0013108, -0.0002145, 0.0012815, -0.0010674, 0.0010607

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009254, upper bound: 0.0010628
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010529
time: 1.50 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9905291, 0.9928138, -0.0015637, 0.0016088
1: -0.0036260, -0.0030396, -0.0036239, -0.0030546, -0.0003896, 0.0004009
2: 0.0060545, 0.0091619, 0.0061336, 0.0091506, -0.0021244, 0.0020649
3: -0.0054432, -0.0040288, -0.0054381, -0.0040649, -0.0009399, 0.0009669
4: 0.0016997, 0.0023012, 0.0017150, 0.0022990, -0.0004112, 0.0003997
5: 0.0065744, 0.0104828, 0.0066739, 0.0104684, -0.0026720, 0.0025971
6: -0.0011198, -0.0001278, -0.0011162, -0.0001531, -0.0006592, 0.0006782
7: -0.0060349, -0.0034683, -0.0060255, -0.0035337, -0.0017055, 0.0017546
8: -0.0027378, -0.0013881, -0.0027329, -0.0014225, -0.0008969, 0.0009227
9: -0.0002543, 0.0013108, -0.0002144, 0.0013051, -0.0010700, 0.0010400

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010628
time: 1.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010529
time: 1.75 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9905645, 0.9928139, -0.0016616, 0.0014349
1: -0.0036673, -0.0030525, -0.0036150, -0.0030546, -0.0004140, 0.0003575
2: 0.0061225, 0.0093805, 0.0061335, 0.0091038, -0.0018947, 0.0021942
3: -0.0055427, -0.0040598, -0.0054168, -0.0040648, -0.0009987, 0.0008624
4: 0.0017129, 0.0023435, 0.0017150, 0.0022899, -0.0003667, 0.0004247
5: 0.0066600, 0.0107577, 0.0066738, 0.0104096, -0.0023831, 0.0027597
6: -0.0011896, -0.0001495, -0.0011012, -0.0001531, -0.0007004, 0.0006048
7: -0.0062154, -0.0035245, -0.0059869, -0.0035336, -0.0018123, 0.0015649
8: -0.0028328, -0.0014177, -0.0027126, -0.0014224, -0.0009530, 0.0008230
9: -0.0002200, 0.0014209, -0.0002145, 0.0012815, -0.0009543, 0.0011051

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010703
time: 1.39 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010561
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9905291, 0.9928138, -0.0016259, 0.0014335
1: -0.0036673, -0.0030525, -0.0036239, -0.0030546, -0.0004051, 0.0003572
2: 0.0061225, 0.0093805, 0.0061336, 0.0091506, -0.0018929, 0.0021469
3: -0.0055427, -0.0040598, -0.0054381, -0.0040649, -0.0009772, 0.0008616
4: 0.0017129, 0.0023435, 0.0017150, 0.0022990, -0.0003664, 0.0004155
5: 0.0066600, 0.0107577, 0.0066739, 0.0104684, -0.0023807, 0.0027003
6: -0.0011896, -0.0001495, -0.0011162, -0.0001531, -0.0006854, 0.0006043
7: -0.0062154, -0.0035245, -0.0060255, -0.0035337, -0.0017732, 0.0015634
8: -0.0028328, -0.0014177, -0.0027329, -0.0014225, -0.0009325, 0.0008222
9: -0.0002200, 0.0014209, -0.0002144, 0.0013051, -0.0009534, 0.0010813

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010703
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010561
time: 1.86 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9905640, 0.9928651, -0.0015394, 0.0013021
1: -0.0036572, -0.0030540, -0.0036151, -0.0030418, -0.0003836, 0.0003245
2: 0.0061307, 0.0093271, 0.0060657, 0.0091044, -0.0017195, 0.0020328
3: -0.0055184, -0.0040635, -0.0054170, -0.0040340, -0.0009252, 0.0007826
4: 0.0017145, 0.0023331, 0.0017019, 0.0022900, -0.0003328, 0.0003934
5: 0.0066702, 0.0106905, 0.0065885, 0.0104104, -0.0021626, 0.0025567
6: -0.0011725, -0.0001521, -0.0011014, -0.0001314, -0.0006489, 0.0005489
7: -0.0061713, -0.0035313, -0.0059874, -0.0034776, -0.0016790, 0.0014202
8: -0.0028096, -0.0014212, -0.0027128, -0.0013930, -0.0008830, 0.0007468
9: -0.0002159, 0.0013940, -0.0002486, 0.0012818, -0.0008660, 0.0010238

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010759
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010669
time: 1.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9905204, 0.9928738, -0.0015718, 0.0013868
1: -0.0036572, -0.0030540, -0.0036260, -0.0030396, -0.0003916, 0.0003455
2: 0.0061307, 0.0093271, 0.0060545, 0.0091619, -0.0018312, 0.0020755
3: -0.0055184, -0.0040635, -0.0054432, -0.0040288, -0.0009447, 0.0008335
4: 0.0017145, 0.0023331, 0.0016997, 0.0023012, -0.0003544, 0.0004017
5: 0.0066702, 0.0106905, 0.0065744, 0.0104828, -0.0023032, 0.0026104
6: -0.0011725, -0.0001521, -0.0011198, -0.0001278, -0.0006626, 0.0005846
7: -0.0061713, -0.0035313, -0.0060349, -0.0034683, -0.0017142, 0.0015125
8: -0.0028096, -0.0014212, -0.0027378, -0.0013881, -0.0009015, 0.0007954
9: -0.0002159, 0.0013940, -0.0002543, 0.0013108, -0.0009223, 0.0010453

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010759
time: 1.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010669
time: 1.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9903954, 0.9928160, -0.0013847, 0.0013847
1: -0.0036572, -0.0030540, -0.0036572, -0.0030540, -0.0003450, 0.0003450
2: 0.0061307, 0.0093271, 0.0061307, 0.0093271, -0.0018285, 0.0018285
3: -0.0055184, -0.0040635, -0.0055184, -0.0040635, -0.0008323, 0.0008323
4: 0.0017145, 0.0023331, 0.0017145, 0.0023331, -0.0003539, 0.0003539
5: 0.0066702, 0.0106905, 0.0066702, 0.0106905, -0.0022998, 0.0022998
6: -0.0011725, -0.0001521, -0.0011725, -0.0001521, -0.0005837, 0.0005837
7: -0.0061713, -0.0035313, -0.0061713, -0.0035313, -0.0015103, 0.0015103
8: -0.0028096, -0.0014212, -0.0028096, -0.0014212, -0.0007942, 0.0007942
9: -0.0002159, 0.0013940, -0.0002159, 0.0013940, -0.0009210, 0.0009210

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009970, upper bound: 0.0010601
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009970, upper bound: 0.0010436
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9903954, 0.9928160, 0.9903550, 0.9928222, -0.0014164, 0.0014737
1: -0.0036572, -0.0030540, -0.0036673, -0.0030525, -0.0003529, 0.0003672
2: 0.0061307, 0.0093271, 0.0061225, 0.0093805, -0.0019460, 0.0018704
3: -0.0055184, -0.0040635, -0.0055427, -0.0040598, -0.0008513, 0.0008857
4: 0.0017145, 0.0023331, 0.0017129, 0.0023435, -0.0003766, 0.0003620
5: 0.0066702, 0.0106905, 0.0066600, 0.0107577, -0.0024476, 0.0023524
6: -0.0011725, -0.0001521, -0.0011896, -0.0001495, -0.0005971, 0.0006212
7: -0.0061713, -0.0035313, -0.0062154, -0.0035245, -0.0015448, 0.0016073
8: -0.0028096, -0.0014212, -0.0028328, -0.0014177, -0.0008124, 0.0008453
9: -0.0002159, 0.0013940, -0.0002200, 0.0014209, -0.0009801, 0.0009420

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009970, upper bound: 0.0010601
time: 1.71 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009970, upper bound: 0.0010436
time: 1.24 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9905640, 0.9928651, -0.0014081, 0.0013557
1: -0.0036260, -0.0030396, -0.0036151, -0.0030418, -0.0003508, 0.0003378
2: 0.0060545, 0.0091619, 0.0060657, 0.0091044, -0.0017902, 0.0018593
3: -0.0054432, -0.0040288, -0.0054170, -0.0040340, -0.0008463, 0.0008148
4: 0.0016997, 0.0023012, 0.0017019, 0.0022900, -0.0003465, 0.0003599
5: 0.0065744, 0.0104828, 0.0065885, 0.0104104, -0.0022516, 0.0023385
6: -0.0011198, -0.0001278, -0.0011014, -0.0001314, -0.0005935, 0.0005715
7: -0.0060349, -0.0034683, -0.0059874, -0.0034776, -0.0015357, 0.0014786
8: -0.0027378, -0.0013881, -0.0027128, -0.0013930, -0.0008076, 0.0007776
9: -0.0002543, 0.0013108, -0.0002486, 0.0012818, -0.0009017, 0.0009365

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011078
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011052
time: 1.31 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9905204, 0.9928738, -0.0013524, 0.0013524
1: -0.0036260, -0.0030396, -0.0036260, -0.0030396, -0.0003370, 0.0003370
2: 0.0060545, 0.0091619, 0.0060545, 0.0091619, -0.0017859, 0.0017859
3: -0.0054432, -0.0040288, -0.0054432, -0.0040288, -0.0008129, 0.0008129
4: 0.0016997, 0.0023012, 0.0016997, 0.0023012, -0.0003457, 0.0003457
5: 0.0065744, 0.0104828, 0.0065744, 0.0104828, -0.0022462, 0.0022462
6: -0.0011198, -0.0001278, -0.0011198, -0.0001278, -0.0005701, 0.0005701
7: -0.0060349, -0.0034683, -0.0060349, -0.0034683, -0.0014750, 0.0014750
8: -0.0027378, -0.0013881, -0.0027378, -0.0013881, -0.0007757, 0.0007757
9: -0.0002543, 0.0013108, -0.0002543, 0.0013108, -0.0008995, 0.0008995

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011078
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011052
time: 1.74 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9905640, 0.9928651, -0.0015972, 0.0013184
1: -0.0036673, -0.0030525, -0.0036151, -0.0030418, -0.0003980, 0.0003285
2: 0.0061225, 0.0093805, 0.0060657, 0.0091044, -0.0017409, 0.0021091
3: -0.0055427, -0.0040598, -0.0054170, -0.0040340, -0.0009600, 0.0007924
4: 0.0017129, 0.0023435, 0.0017019, 0.0022900, -0.0003369, 0.0004082
5: 0.0066600, 0.0107577, 0.0065885, 0.0104104, -0.0021896, 0.0026526
6: -0.0011896, -0.0001495, -0.0011014, -0.0001314, -0.0006733, 0.0005557
7: -0.0062154, -0.0035245, -0.0059874, -0.0034776, -0.0017419, 0.0014379
8: -0.0028328, -0.0014177, -0.0027128, -0.0013930, -0.0009161, 0.0007562
9: -0.0002200, 0.0014209, -0.0002486, 0.0012818, -0.0008768, 0.0010622

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009902, upper bound: 0.0010893
time: 1.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009902, upper bound: 0.0010831
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9905204, 0.9928738, -0.0015690, 0.0013303
1: -0.0036673, -0.0030525, -0.0036260, -0.0030396, -0.0003910, 0.0003315
2: 0.0061225, 0.0093805, 0.0060545, 0.0091619, -0.0017566, 0.0020718
3: -0.0055427, -0.0040598, -0.0054432, -0.0040288, -0.0009430, 0.0007995
4: 0.0017129, 0.0023435, 0.0016997, 0.0023012, -0.0003400, 0.0004010
5: 0.0066600, 0.0107577, 0.0065744, 0.0104828, -0.0022094, 0.0026058
6: -0.0011896, -0.0001495, -0.0011198, -0.0001278, -0.0006614, 0.0005608
7: -0.0062154, -0.0035245, -0.0060349, -0.0034683, -0.0017112, 0.0014509
8: -0.0028328, -0.0014177, -0.0027378, -0.0013881, -0.0008999, 0.0007630
9: -0.0002200, 0.0014209, -0.0002543, 0.0013108, -0.0008847, 0.0010435

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009902, upper bound: 0.0010893
time: 1.88 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009902, upper bound: 0.0010831
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9903954, 0.9928160, -0.0013868, 0.0015718
1: -0.0036260, -0.0030396, -0.0036572, -0.0030540, -0.0003455, 0.0003916
2: 0.0060545, 0.0091619, 0.0061307, 0.0093271, -0.0020755, 0.0018312
3: -0.0054432, -0.0040288, -0.0055184, -0.0040635, -0.0008335, 0.0009447
4: 0.0016997, 0.0023012, 0.0017145, 0.0023331, -0.0004017, 0.0003544
5: 0.0065744, 0.0104828, 0.0066702, 0.0106905, -0.0026104, 0.0023032
6: -0.0011198, -0.0001278, -0.0011725, -0.0001521, -0.0005846, 0.0006626
7: -0.0060349, -0.0034683, -0.0061713, -0.0035313, -0.0015125, 0.0017142
8: -0.0027378, -0.0013881, -0.0028096, -0.0014212, -0.0007954, 0.0009015
9: -0.0002543, 0.0013108, -0.0002159, 0.0013940, -0.0010453, 0.0009223

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010628
time: 1.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010529
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905204, 0.9928738, 0.9903550, 0.9928222, -0.0013303, 0.0015690
1: -0.0036260, -0.0030396, -0.0036673, -0.0030525, -0.0003315, 0.0003910
2: 0.0060545, 0.0091619, 0.0061225, 0.0093805, -0.0020718, 0.0017566
3: -0.0054432, -0.0040288, -0.0055427, -0.0040598, -0.0007995, 0.0009430
4: 0.0016997, 0.0023012, 0.0017129, 0.0023435, -0.0004010, 0.0003400
5: 0.0065744, 0.0104828, 0.0066600, 0.0107577, -0.0026058, 0.0022094
6: -0.0011198, -0.0001278, -0.0011896, -0.0001495, -0.0005608, 0.0006614
7: -0.0060349, -0.0034683, -0.0062154, -0.0035245, -0.0014509, 0.0017112
8: -0.0027378, -0.0013881, -0.0028328, -0.0014177, -0.0007630, 0.0008999
9: -0.0002543, 0.0013108, -0.0002200, 0.0014209, -0.0010435, 0.0008847

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010628
time: 1.36 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010529
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9903954, 0.9928160, -0.0014737, 0.0014164
1: -0.0036673, -0.0030525, -0.0036572, -0.0030540, -0.0003672, 0.0003529
2: 0.0061225, 0.0093805, 0.0061307, 0.0093271, -0.0018704, 0.0019460
3: -0.0055427, -0.0040598, -0.0055184, -0.0040635, -0.0008857, 0.0008513
4: 0.0017129, 0.0023435, 0.0017145, 0.0023331, -0.0003620, 0.0003766
5: 0.0066600, 0.0107577, 0.0066702, 0.0106905, -0.0023524, 0.0024476
6: -0.0011896, -0.0001495, -0.0011725, -0.0001521, -0.0006212, 0.0005971
7: -0.0062154, -0.0035245, -0.0061713, -0.0035313, -0.0016073, 0.0015448
8: -0.0028328, -0.0014177, -0.0028096, -0.0014212, -0.0008453, 0.0008124
9: -0.0002200, 0.0014209, -0.0002159, 0.0013940, -0.0009420, 0.0009801

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010703
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010561
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9903550, 0.9928222, 0.9903550, 0.9928222, -0.0014105, 0.0014105
1: -0.0036673, -0.0030525, -0.0036673, -0.0030525, -0.0003515, 0.0003515
2: 0.0061225, 0.0093805, 0.0061225, 0.0093805, -0.0018626, 0.0018626
3: -0.0055427, -0.0040598, -0.0055427, -0.0040598, -0.0008478, 0.0008478
4: 0.0017129, 0.0023435, 0.0017129, 0.0023435, -0.0003605, 0.0003605
5: 0.0066600, 0.0107577, 0.0066600, 0.0107577, -0.0023427, 0.0023427
6: -0.0011896, -0.0001495, -0.0011896, -0.0001495, -0.0005946, 0.0005946
7: -0.0062154, -0.0035245, -0.0062154, -0.0035245, -0.0015384, 0.0015384
8: -0.0028328, -0.0014177, -0.0028328, -0.0014177, -0.0008090, 0.0008090
9: -0.0002200, 0.0014209, -0.0002200, 0.0014209, -0.0009381, 0.0009381

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010703
time: 1.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010561
time: 1.35 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 4.66 seconds
NS_A2_B1_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010759
NS_A2_B1_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009165, upper bound: 0.0010669
NS_A2_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010759
NS_A2_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009165, upper bound: 0.0010669
NS_A2_B1_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009971, upper bound: 0.0010601
NS_A2_B1_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009971, upper bound: 0.0010436
NS_A2_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009971, upper bound: 0.0010601
NS_A2_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009971, upper bound: 0.0010436
NS_A2_B1_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011078
NS_A2_B1_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011052
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011078
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011052
NS_A2_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009904, upper bound: 0.0010893
NS_A2_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009904, upper bound: 0.0010831
NS_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009904, upper bound: 0.0010893
NS_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009904, upper bound: 0.0010831
NS_A2_B1_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009254, upper bound: 0.0010628
NS_A2_B1_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010529
NS_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010628
NS_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010529
NS_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010703
NS_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010561
NS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010703
NS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010561
NS_A2_B2_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010759
NS_A2_B2_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010669
NS_A2_B2_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010759
NS_A2_B2_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009978, upper bound: 0.0010669
NS_A2_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009970, upper bound: 0.0010601
NS_A2_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009970, upper bound: 0.0010436
NS_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009970, upper bound: 0.0010601
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009970, upper bound: 0.0010436
NS_A2_B2_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011078
NS_A2_B2_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011052
NS_A2_B2_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011078
NS_A2_B2_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011052
NS_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009902, upper bound: 0.0010893
NS_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009902, upper bound: 0.0010831
NS_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009902, upper bound: 0.0010893
NS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009902, upper bound: 0.0010831
NS_A2_B2_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010628
NS_A2_B2_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010529
NS_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010628
NS_A2_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0010042, upper bound: 0.0010529
NS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010703
NS_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010561
NS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010703
NS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.66
Output dim: 0, lower bound: -0.0009897, upper bound: 0.0010561

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9907144, 0.9928743, -0.0015994, 0.0013545
1: -0.0036249, -0.0030469, -0.0035776, -0.0030395, -0.0003985, 0.0003375
2: 0.0060932, 0.0091560, 0.0060536, 0.0089056, -0.0017885, 0.0021120
3: -0.0054405, -0.0040465, -0.0053266, -0.0040285, -0.0009613, 0.0008141
4: 0.0017072, 0.0023000, 0.0016996, 0.0022516, -0.0003462, 0.0004088
5: 0.0066231, 0.0104753, 0.0065733, 0.0101604, -0.0022495, 0.0026564
6: -0.0011179, -0.0001402, -0.0010380, -0.0001275, -0.0006742, 0.0005709
7: -0.0060300, -0.0035003, -0.0058232, -0.0034676, -0.0017444, 0.0014772
8: -0.0027353, -0.0014049, -0.0026265, -0.0013877, -0.0009174, 0.0007769
9: -0.0002348, 0.0013078, -0.0002547, 0.0011817, -0.0009008, 0.0010637

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
time: 1.35 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
time: 1.39 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9904792, 0.9928120, 0.9907161, 0.9928608, -0.0016568, 0.0013525
1: -0.0036363, -0.0030550, -0.0035773, -0.0030429, -0.0004128, 0.0003370
2: 0.0061360, 0.0092164, 0.0060715, 0.0089036, -0.0017859, 0.0021878
3: -0.0054680, -0.0040659, -0.0053256, -0.0040366, -0.0009958, 0.0008129
4: 0.0017155, 0.0023117, 0.0017030, 0.0022512, -0.0003457, 0.0004234
5: 0.0066769, 0.0105513, 0.0065959, 0.0101578, -0.0022462, 0.0027517
6: -0.0011372, -0.0001538, -0.0010373, -0.0001333, -0.0006984, 0.0005701
7: -0.0060799, -0.0035356, -0.0058215, -0.0034824, -0.0018070, 0.0014751
8: -0.0027615, -0.0014235, -0.0026256, -0.0013955, -0.0009503, 0.0007757
9: -0.0002132, 0.0013382, -0.0002457, 0.0011807, -0.0008995, 0.0011019

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008153, upper bound: 0.0008738
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007373, upper bound: 0.0007587
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9906772, 0.9928792, -0.0015681, 0.0013607
1: -0.0036249, -0.0030469, -0.0035869, -0.0030383, -0.0003907, 0.0003391
2: 0.0060932, 0.0091560, 0.0060472, 0.0089549, -0.0017968, 0.0020706
3: -0.0054405, -0.0040465, -0.0053490, -0.0040256, -0.0009425, 0.0008178
4: 0.0017072, 0.0023000, 0.0016983, 0.0022611, -0.0003478, 0.0004008
5: 0.0066231, 0.0104753, 0.0065653, 0.0102224, -0.0022600, 0.0026043
6: -0.0011179, -0.0001402, -0.0010537, -0.0001255, -0.0006610, 0.0005736
7: -0.0060300, -0.0035003, -0.0058639, -0.0034623, -0.0017102, 0.0014841
8: -0.0027353, -0.0014049, -0.0026479, -0.0013850, -0.0008994, 0.0007805
9: -0.0002348, 0.0013078, -0.0002579, 0.0012065, -0.0009050, 0.0010429

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
time: 1.49 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9904792, 0.9928120, 0.9906788, 0.9928662, -0.0016188, 0.0013618
1: -0.0036363, -0.0030550, -0.0035865, -0.0030415, -0.0004034, 0.0003393
2: 0.0061360, 0.0092164, 0.0060644, 0.0089528, -0.0017983, 0.0021376
3: -0.0054680, -0.0040659, -0.0053481, -0.0040334, -0.0009729, 0.0008185
4: 0.0017155, 0.0023117, 0.0017016, 0.0022607, -0.0003481, 0.0004137
5: 0.0066769, 0.0105513, 0.0065869, 0.0102197, -0.0022618, 0.0026885
6: -0.0011372, -0.0001538, -0.0010530, -0.0001310, -0.0006824, 0.0005741
7: -0.0060799, -0.0035356, -0.0058622, -0.0034765, -0.0017655, 0.0014853
8: -0.0027615, -0.0014235, -0.0026470, -0.0013924, -0.0009285, 0.0007811
9: -0.0002132, 0.0013382, -0.0002493, 0.0012055, -0.0009057, 0.0010766

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008153, upper bound: 0.0008700
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007369, upper bound: 0.0007528
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9905640, 0.9928651, -0.0014036, 0.0013206
1: -0.0036249, -0.0030469, -0.0036151, -0.0030418, -0.0003497, 0.0003290
2: 0.0060932, 0.0091560, 0.0060657, 0.0091044, -0.0017438, 0.0018535
3: -0.0054405, -0.0040465, -0.0054170, -0.0040340, -0.0008436, 0.0007937
4: 0.0017072, 0.0023000, 0.0017019, 0.0022900, -0.0003375, 0.0003587
5: 0.0066231, 0.0104753, 0.0065885, 0.0104104, -0.0021932, 0.0023312
6: -0.0011179, -0.0001402, -0.0011014, -0.0001314, -0.0005917, 0.0005567
7: -0.0060300, -0.0035003, -0.0059874, -0.0034776, -0.0015309, 0.0014403
8: -0.0027353, -0.0014049, -0.0027128, -0.0013930, -0.0008051, 0.0007574
9: -0.0002348, 0.0013078, -0.0002486, 0.0012818, -0.0008783, 0.0009335

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.9904792, 0.9928120, 0.9905655, 0.9928516, -0.0014741, 0.0013166
1: -0.0036363, -0.0030550, -0.0036148, -0.0030452, -0.0003673, 0.0003281
2: 0.0061360, 0.0092164, 0.0060837, 0.0091024, -0.0017386, 0.0019465
3: -0.0054680, -0.0040659, -0.0054161, -0.0040422, -0.0008860, 0.0007913
4: 0.0017155, 0.0023117, 0.0017054, 0.0022896, -0.0003365, 0.0003767
5: 0.0066769, 0.0105513, 0.0066112, 0.0104079, -0.0021867, 0.0024482
6: -0.0011372, -0.0001538, -0.0011008, -0.0001371, -0.0006214, 0.0005550
7: -0.0060799, -0.0035356, -0.0059857, -0.0034925, -0.0016077, 0.0014360
8: -0.0027615, -0.0014235, -0.0027120, -0.0014008, -0.0008455, 0.0007552
9: -0.0002132, 0.0013382, -0.0002396, 0.0012808, -0.0008757, 0.0009804

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008095, upper bound: 0.0008604
time: 1.51 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007241, upper bound: 0.0007359
time: 1.30 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9905204, 0.9928738, -0.0013474, 0.0013165
1: -0.0036249, -0.0030469, -0.0036260, -0.0030396, -0.0003357, 0.0003280
2: 0.0060932, 0.0091560, 0.0060545, 0.0091619, -0.0017385, 0.0017793
3: -0.0054405, -0.0040465, -0.0054432, -0.0040288, -0.0008098, 0.0007913
4: 0.0017072, 0.0023000, 0.0016997, 0.0023012, -0.0003365, 0.0003444
5: 0.0066231, 0.0104753, 0.0065744, 0.0104828, -0.0021865, 0.0022378
6: -0.0011179, -0.0001402, -0.0011198, -0.0001278, -0.0005680, 0.0005550
7: -0.0060300, -0.0035003, -0.0060349, -0.0034683, -0.0014696, 0.0014359
8: -0.0027353, -0.0014049, -0.0027378, -0.0013881, -0.0007728, 0.0007551
9: -0.0002348, 0.0013078, -0.0002543, 0.0013108, -0.0008756, 0.0008961

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
time: 1.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9904792, 0.9928120, 0.9905220, 0.9928604, -0.0014200, 0.0013143
1: -0.0036363, -0.0030550, -0.0036256, -0.0030430, -0.0003538, 0.0003275
2: 0.0061360, 0.0092164, 0.0060721, 0.0091600, -0.0017355, 0.0018750
3: -0.0054680, -0.0040659, -0.0054423, -0.0040369, -0.0008534, 0.0007899
4: 0.0017155, 0.0023117, 0.0017031, 0.0023008, -0.0003359, 0.0003629
5: 0.0066769, 0.0105513, 0.0065965, 0.0104803, -0.0021828, 0.0023583
6: -0.0011372, -0.0001538, -0.0011192, -0.0001334, -0.0005986, 0.0005540
7: -0.0060799, -0.0035356, -0.0060333, -0.0034829, -0.0015487, 0.0014334
8: -0.0027615, -0.0014235, -0.0027370, -0.0013957, -0.0008144, 0.0007538
9: -0.0002132, 0.0013382, -0.0002454, 0.0013098, -0.0008741, 0.0009444

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008095, upper bound: 0.0008592
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007238, upper bound: 0.0007315
time: 1.40 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 4.21 seconds
NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0008153, upper bound: 0.0008738
NS_A2_B1_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0007373, upper bound: 0.0007587
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0008153, upper bound: 0.0008700
NS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0007369, upper bound: 0.0007528
NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
NS_A2_B2_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0008095, upper bound: 0.0008604
NS_A2_B2_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0007241, upper bound: 0.0007359
NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0010268, upper bound: 0.0011051
NS_A2_B2_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0008095, upper bound: 0.0008592
NS_A2_B2_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 4.21
Output dim: 0, lower bound: -0.0007238, upper bound: 0.0007315

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9907194, 0.9928457, -0.0015636, 0.0013493
1: -0.0036249, -0.0030469, -0.0035764, -0.0030466, -0.0003896, 0.0003362
2: 0.0060932, 0.0091560, 0.0060914, 0.0088992, -0.0017817, 0.0020648
3: -0.0054405, -0.0040465, -0.0053237, -0.0040457, -0.0009398, 0.0008110
4: 0.0017072, 0.0023000, 0.0017069, 0.0022503, -0.0003449, 0.0003996
5: 0.0066231, 0.0104753, 0.0066209, 0.0101523, -0.0022410, 0.0025969
6: -0.0011179, -0.0001402, -0.0010359, -0.0001396, -0.0006591, 0.0005688
7: -0.0060300, -0.0035003, -0.0058179, -0.0034989, -0.0017054, 0.0014716
8: -0.0027353, -0.0014049, -0.0026237, -0.0014042, -0.0008968, 0.0007739
9: -0.0002348, 0.0013078, -0.0002357, 0.0011785, -0.0008974, 0.0010399

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008624, upper bound: 0.0008721
time: 1.34 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007449, upper bound: 0.0007888
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9906793, 0.9928107, -0.0015658, 0.0014315
1: -0.0036249, -0.0030469, -0.0035864, -0.0030554, -0.0003902, 0.0003567
2: 0.0060932, 0.0091560, 0.0061378, 0.0089521, -0.0018903, 0.0020677
3: -0.0054405, -0.0040465, -0.0053477, -0.0040668, -0.0009411, 0.0008604
4: 0.0017072, 0.0023000, 0.0017158, 0.0022605, -0.0003659, 0.0004002
5: 0.0066231, 0.0104753, 0.0066791, 0.0102188, -0.0023775, 0.0026006
6: -0.0011179, -0.0001402, -0.0010528, -0.0001544, -0.0006601, 0.0006034
7: -0.0060300, -0.0035003, -0.0058616, -0.0035371, -0.0017078, 0.0015613
8: -0.0027353, -0.0014049, -0.0026467, -0.0014243, -0.0008981, 0.0008210
9: -0.0002348, 0.0013078, -0.0002123, 0.0012051, -0.0009520, 0.0010414

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008624, upper bound: 0.0008721
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007449, upper bound: 0.0007888
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9906822, 0.9928489, -0.0015330, 0.0013556
1: -0.0036249, -0.0030469, -0.0035857, -0.0030458, -0.0003820, 0.0003378
2: 0.0060932, 0.0091560, 0.0060872, 0.0089484, -0.0017900, 0.0020243
3: -0.0054405, -0.0040465, -0.0053460, -0.0040438, -0.0009214, 0.0008147
4: 0.0017072, 0.0023000, 0.0017061, 0.0022598, -0.0003465, 0.0003918
5: 0.0066231, 0.0104753, 0.0066156, 0.0102142, -0.0022514, 0.0025460
6: -0.0011179, -0.0001402, -0.0010516, -0.0001383, -0.0006462, 0.0005714
7: -0.0060300, -0.0035003, -0.0058585, -0.0034954, -0.0016719, 0.0014785
8: -0.0027353, -0.0014049, -0.0026451, -0.0014023, -0.0008792, 0.0007775
9: -0.0002348, 0.0013078, -0.0002378, 0.0012033, -0.0009016, 0.0010195

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008624, upper bound: 0.0008715
time: 1.12 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007448, upper bound: 0.0007852
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9906371, 0.9928167, -0.0015336, 0.0014376
1: -0.0036249, -0.0030469, -0.0035969, -0.0030538, -0.0003821, 0.0003582
2: 0.0060932, 0.0091560, 0.0061298, 0.0090079, -0.0018983, 0.0020251
3: -0.0054405, -0.0040465, -0.0053731, -0.0040631, -0.0009218, 0.0008640
4: 0.0017072, 0.0023000, 0.0017143, 0.0022713, -0.0003674, 0.0003920
5: 0.0066231, 0.0104753, 0.0066691, 0.0102890, -0.0023876, 0.0025471
6: -0.0011179, -0.0001402, -0.0010706, -0.0001519, -0.0006465, 0.0006060
7: -0.0060300, -0.0035003, -0.0059077, -0.0035305, -0.0016726, 0.0015679
8: -0.0027353, -0.0014049, -0.0026709, -0.0014208, -0.0008796, 0.0008245
9: -0.0002348, 0.0013078, -0.0002164, 0.0012332, -0.0009561, 0.0010200

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008624, upper bound: 0.0008715
time: 1.13 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007448, upper bound: 0.0007852
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9905686, 0.9928367, -0.0013671, 0.0013155
1: -0.0036249, -0.0030469, -0.0036140, -0.0030489, -0.0003406, 0.0003278
2: 0.0060932, 0.0091560, 0.0061034, 0.0090983, -0.0017371, 0.0018052
3: -0.0054405, -0.0040465, -0.0054143, -0.0040511, -0.0008217, 0.0007907
4: 0.0017072, 0.0023000, 0.0017092, 0.0022888, -0.0003362, 0.0003494
5: 0.0066231, 0.0104753, 0.0066359, 0.0104027, -0.0021848, 0.0022705
6: -0.0011179, -0.0001402, -0.0010995, -0.0001434, -0.0005763, 0.0005545
7: -0.0060300, -0.0035003, -0.0059823, -0.0035087, -0.0014910, 0.0014347
8: -0.0027353, -0.0014049, -0.0027102, -0.0014094, -0.0007841, 0.0007545
9: -0.0002348, 0.0013078, -0.0002296, 0.0012788, -0.0008749, 0.0009092

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008524, upper bound: 0.0008632
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007313, upper bound: 0.0007647
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9905261, 0.9928015, -0.0013676, 0.0013980
1: -0.0036249, -0.0030469, -0.0036246, -0.0030576, -0.0003408, 0.0003484
2: 0.0060932, 0.0091560, 0.0061498, 0.0091545, -0.0018461, 0.0018059
3: -0.0054405, -0.0040465, -0.0054399, -0.0040722, -0.0008220, 0.0008403
4: 0.0017072, 0.0023000, 0.0017182, 0.0022997, -0.0003573, 0.0003495
5: 0.0066231, 0.0104753, 0.0066943, 0.0104734, -0.0023219, 0.0022713
6: -0.0011179, -0.0001402, -0.0011174, -0.0001582, -0.0005765, 0.0005893
7: -0.0060300, -0.0035003, -0.0060288, -0.0035471, -0.0014915, 0.0015248
8: -0.0027353, -0.0014049, -0.0027346, -0.0014295, -0.0007844, 0.0008019
9: -0.0002348, 0.0013078, -0.0002063, 0.0013071, -0.0009298, 0.0009095

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008524, upper bound: 0.0008632
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007313, upper bound: 0.0007647
time: 1.15 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9905249, 0.9928444, -0.0013115, 0.0013115
1: -0.0036249, -0.0030469, -0.0036249, -0.0030469, -0.0003268, 0.0003268
2: 0.0060932, 0.0091560, 0.0060932, 0.0091560, -0.0017319, 0.0017319
3: -0.0054405, -0.0040465, -0.0054405, -0.0040465, -0.0007883, 0.0007883
4: 0.0017072, 0.0023000, 0.0017072, 0.0023000, -0.0003352, 0.0003352
5: 0.0066231, 0.0104753, 0.0066231, 0.0104753, -0.0021782, 0.0021782
6: -0.0011179, -0.0001402, -0.0011179, -0.0001402, -0.0005529, 0.0005529
7: -0.0060300, -0.0035003, -0.0060300, -0.0035003, -0.0014304, 0.0014304
8: -0.0027353, -0.0014049, -0.0027353, -0.0014049, -0.0007522, 0.0007522
9: -0.0002348, 0.0013078, -0.0002348, 0.0013078, -0.0008723, 0.0008723

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008524, upper bound: 0.0008628
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007313, upper bound: 0.0007608
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9905249, 0.9928444, 0.9904792, 0.9928120, -0.0013101, 0.0013937
1: -0.0036249, -0.0030469, -0.0036363, -0.0030550, -0.0003265, 0.0003473
2: 0.0060932, 0.0091560, 0.0061360, 0.0092164, -0.0018404, 0.0017300
3: -0.0054405, -0.0040465, -0.0054680, -0.0040659, -0.0007874, 0.0008377
4: 0.0017072, 0.0023000, 0.0017155, 0.0023117, -0.0003562, 0.0003348
5: 0.0066231, 0.0104753, 0.0066769, 0.0105513, -0.0023147, 0.0021759
6: -0.0011179, -0.0001402, -0.0011372, -0.0001538, -0.0005523, 0.0005875
7: -0.0060300, -0.0035003, -0.0060799, -0.0035356, -0.0014289, 0.0015200
8: -0.0027353, -0.0014049, -0.0027615, -0.0014235, -0.0007514, 0.0007994
9: -0.0002348, 0.0013078, -0.0002132, 0.0013382, -0.0009269, 0.0008713

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 89

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008524, upper bound: 0.0008628
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007313, upper bound: 0.0007608
time: 0.97 seconds

## Summary of splitting at layer (split count: 10)
- Time for NS candidates: 3.54 seconds
NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0008624, upper bound: 0.0008721
NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0007449, upper bound: 0.0007888
NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0008624, upper bound: 0.0008721
NS_A2_B1_A2_B2_A2_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0007449, upper bound: 0.0007888
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0008624, upper bound: 0.0008715
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0007448, upper bound: 0.0007852
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0008624, upper bound: 0.0008715
NS_A2_B1_A2_B2_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0007448, upper bound: 0.0007852
NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0008524, upper bound: 0.0008632
NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0007313, upper bound: 0.0007647
NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0008524, upper bound: 0.0008632
NS_A2_B2_A2_B2_A2_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0007313, upper bound: 0.0007647
NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0008524, upper bound: 0.0008628
NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0007313, upper bound: 0.0007608
NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0008524, upper bound: 0.0008628
NS_A2_B2_A2_B2_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 11, time: 3.54
Output dim: 0, lower bound: -0.0007313, upper bound: 0.0007608

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.14 + 508.00 = 511.14 seconds
