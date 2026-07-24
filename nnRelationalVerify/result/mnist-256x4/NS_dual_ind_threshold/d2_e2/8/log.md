## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.012999239999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0060496, 0.0032124, -0.0060496, 0.0032124, -0.0092621, 0.0092621)
1: (-0.0036231, 0.0133442, -0.0036231, 0.0133442, -0.0169673, 0.0169673)
2: (0.0049072, 0.0242831, 0.0049072, 0.0242831, -0.0190535, 0.0190535)
3: (-0.0086418, -0.0017943, -0.0086418, -0.0017943, -0.0068474, 0.0068474)
4: (0.0025300, 0.0077342, 0.0025300, 0.0077342, -0.0052041, 0.0052041)
5: (-0.0062876, 0.0019187, -0.0062876, 0.0019187, -0.0082063, 0.0082063)
6: (-0.0074834, -0.0043177, -0.0074834, -0.0043177, -0.0031656, 0.0031656)
7: (-0.0063982, 0.0016634, -0.0063982, 0.0016634, -0.0080616, 0.0080616)
8: (-0.0105992, -0.0012224, -0.0105992, -0.0012224, -0.0093768, 0.0093768)
9: (0.9911500, 1.0128294, 0.9911500, 1.0128294, -0.0216795, 0.0216795)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.99 + 2.69 = 4.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0144436, upper bound: 0.0144436

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 76

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140630, upper bound: 0.0141704
time: 1.65 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141704, upper bound: 0.0141704
time: 1.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.19 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.19
Output dim: 9, lower bound: -0.0140630, upper bound: 0.0141704
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.19
Output dim: 9, lower bound: -0.0141704, upper bound: 0.0141704

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0055506, 0.0016288, -0.0059988, 0.0030505, -0.0086010, 0.0076276
1: -0.0032378, 0.0116753, -0.0036031, 0.0131742, -0.0164120, 0.0152784
2: 0.0050699, 0.0212018, 0.0049081, 0.0239698, -0.0182414, 0.0158516
3: -0.0074756, -0.0018362, -0.0085247, -0.0017981, -0.0056775, 0.0066885
4: 0.0028859, 0.0077109, 0.0025651, 0.0077340, -0.0048481, 0.0051459
5: -0.0057254, 0.0014994, -0.0062208, 0.0018778, -0.0076032, 0.0077202
6: -0.0070200, -0.0044159, -0.0074363, -0.0043264, -0.0026937, 0.0030204
7: -0.0056909, 0.0010873, -0.0063249, 0.0016107, -0.0073017, 0.0074122
8: -0.0085474, -0.0012354, -0.0103896, -0.0012233, -0.0073241, 0.0091541
9: 0.9930466, 1.0125208, 0.9913414, 1.0128148, -0.0197681, 0.0211794

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140630, upper bound: 0.0140630
time: 1.76 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140630, upper bound: 0.0141704
time: 1.63 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0057894, 0.0024363, -0.0060376, 0.0031758, -0.0089651, 0.0084738
1: -0.0035176, 0.0125586, -0.0036183, 0.0133071, -0.0168247, 0.0161769
2: 0.0049159, 0.0228058, 0.0049076, 0.0242134, -0.0189804, 0.0167345
3: -0.0080411, -0.0018123, -0.0086135, -0.0017952, -0.0062459, 0.0068012
4: 0.0027203, 0.0077327, 0.0025387, 0.0077341, -0.0048723, 0.0051940
5: -0.0059986, 0.0016836, -0.0062727, 0.0019079, -0.0079065, 0.0079564
6: -0.0072517, -0.0043709, -0.0074725, -0.0043202, -0.0029315, 0.0031016
7: -0.0060874, 0.0014033, -0.0063836, 0.0016512, -0.0077386, 0.0077870
8: -0.0096148, -0.0012273, -0.0105528, -0.0012226, -0.0083923, 0.0093255
9: 0.9920993, 1.0127498, 0.9911945, 1.0128260, -0.0207267, 0.0215553

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 76

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141704, upper bound: 0.0140630
time: 2.12 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141704, upper bound: 0.0141704
time: 2.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.94 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.94
Output dim: 9, lower bound: -0.0140630, upper bound: 0.0140630
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.94
Output dim: 9, lower bound: -0.0140630, upper bound: 0.0141704
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.94
Output dim: 9, lower bound: -0.0141704, upper bound: 0.0140630
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.94
Output dim: 9, lower bound: -0.0141704, upper bound: 0.0141704

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0055506, 0.0016288, -0.0055506, 0.0016288, -0.0071793, 0.0071793
1: -0.0032378, 0.0116753, -0.0032378, 0.0116753, -0.0149131, 0.0149131
2: 0.0050699, 0.0212018, 0.0050699, 0.0212018, -0.0153681, 0.0153681
3: -0.0074756, -0.0018362, -0.0074756, -0.0018362, -0.0056394, 0.0056394
4: 0.0028859, 0.0077109, 0.0028859, 0.0077109, -0.0047391, 0.0047391
5: -0.0057254, 0.0014994, -0.0057254, 0.0014994, -0.0072247, 0.0072247
6: -0.0070200, -0.0044159, -0.0070200, -0.0044159, -0.0026041, 0.0026041
7: -0.0056909, 0.0010873, -0.0056909, 0.0010873, -0.0067783, 0.0067783
8: -0.0085474, -0.0012354, -0.0085474, -0.0012354, -0.0073120, 0.0073120
9: 0.9930466, 1.0125208, 0.9930466, 1.0125208, -0.0194741, 0.0194741

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136607, upper bound: 0.0138076
time: 3.00 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138098, upper bound: 0.0138098
time: 2.23 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055506, 0.0016288, -0.0057894, 0.0024363, -0.0079868, 0.0074181
1: -0.0032378, 0.0116753, -0.0035176, 0.0125586, -0.0157965, 0.0151929
2: 0.0050699, 0.0212018, 0.0049159, 0.0228058, -0.0173130, 0.0158430
3: -0.0074756, -0.0018362, -0.0080411, -0.0018123, -0.0056634, 0.0062048
4: 0.0028859, 0.0077109, 0.0027203, 0.0077327, -0.0048468, 0.0049906
5: -0.0057254, 0.0014994, -0.0059986, 0.0016836, -0.0074090, 0.0074980
6: -0.0070200, -0.0044159, -0.0072517, -0.0043709, -0.0026491, 0.0028357
7: -0.0056909, 0.0010873, -0.0060874, 0.0014033, -0.0070943, 0.0071747
8: -0.0085474, -0.0012354, -0.0096148, -0.0012273, -0.0073201, 0.0083794
9: 0.9930466, 1.0125208, 0.9920993, 1.0127498, -0.0197031, 0.0204215

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0136607, upper bound: 0.0139210
time: 2.17 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138098, upper bound: 0.0139213
time: 1.56 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0057894, 0.0024363, -0.0055506, 0.0016288, -0.0074181, 0.0079868
1: -0.0035176, 0.0125586, -0.0032378, 0.0116753, -0.0151929, 0.0157965
2: 0.0049159, 0.0228058, 0.0050699, 0.0212018, -0.0158430, 0.0173130
3: -0.0080411, -0.0018123, -0.0074756, -0.0018362, -0.0062048, 0.0056634
4: 0.0027203, 0.0077327, 0.0028859, 0.0077109, -0.0049906, 0.0048468
5: -0.0059986, 0.0016836, -0.0057254, 0.0014994, -0.0074980, 0.0074090
6: -0.0072517, -0.0043709, -0.0070200, -0.0044159, -0.0028357, 0.0026491
7: -0.0060874, 0.0014033, -0.0056909, 0.0010873, -0.0071747, 0.0070943
8: -0.0096148, -0.0012273, -0.0085474, -0.0012354, -0.0083794, 0.0073201
9: 0.9920993, 1.0127498, 0.9930466, 1.0125208, -0.0204215, 0.0197031

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138049, upper bound: 0.0138076
time: 2.44 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139213, upper bound: 0.0138098
time: 2.03 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057894, 0.0024363, -0.0057894, 0.0024363, -0.0082256, 0.0082256
1: -0.0035176, 0.0125586, -0.0035176, 0.0125586, -0.0160762, 0.0160762
2: 0.0049159, 0.0228058, 0.0049159, 0.0228058, -0.0167255, 0.0167255
3: -0.0080411, -0.0018123, -0.0080411, -0.0018123, -0.0062288, 0.0062288
4: 0.0027203, 0.0077327, 0.0027203, 0.0077327, -0.0048705, 0.0048705
5: -0.0059986, 0.0016836, -0.0059986, 0.0016836, -0.0076822, 0.0076822
6: -0.0072517, -0.0043709, -0.0072517, -0.0043709, -0.0028807, 0.0028807
7: -0.0060874, 0.0014033, -0.0060874, 0.0014033, -0.0074907, 0.0074907
8: -0.0096148, -0.0012273, -0.0096148, -0.0012273, -0.0083875, 0.0083875
9: 0.9920993, 1.0127498, 0.9920993, 1.0127498, -0.0206505, 0.0206505

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138049, upper bound: 0.0138076
time: 2.58 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139213, upper bound: 0.0138098
time: 2.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 7.37 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.37
Output dim: 9, lower bound: -0.0136607, upper bound: 0.0138076
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.37
Output dim: 9, lower bound: -0.0138098, upper bound: 0.0138098
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.37
Output dim: 9, lower bound: -0.0136607, upper bound: 0.0139210
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.37
Output dim: 9, lower bound: -0.0138098, upper bound: 0.0139213
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.37
Output dim: 9, lower bound: -0.0138049, upper bound: 0.0138076
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.37
Output dim: 9, lower bound: -0.0139213, upper bound: 0.0138098
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.37
Output dim: 9, lower bound: -0.0138049, upper bound: 0.0138076
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.37
Output dim: 9, lower bound: -0.0139213, upper bound: 0.0138098

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0051612, 0.0006780, -0.0054845, 0.0014628, -0.0066241, 0.0061625
1: -0.0031417, 0.0107180, -0.0032078, 0.0115149, -0.0146567, 0.0139258
2: 0.0050196, 0.0193333, 0.0050734, 0.0208895, -0.0148806, 0.0134700
3: -0.0065791, -0.0018626, -0.0073258, -0.0018397, -0.0047393, 0.0054631
4: 0.0032635, 0.0077249, 0.0029493, 0.0077103, -0.0043445, 0.0045747
5: -0.0055295, 0.0009983, -0.0056866, 0.0014162, -0.0069458, 0.0066849
6: -0.0067039, -0.0045584, -0.0069668, -0.0044399, -0.0022640, 0.0024085
7: -0.0054427, 0.0006815, -0.0056470, 0.0010132, -0.0064559, 0.0063285
8: -0.0073365, -0.0013333, -0.0083431, -0.0012512, -0.0060853, 0.0070097
9: 0.9944047, 1.0124737, 0.9932769, 1.0124974, -0.0180927, 0.0191968

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131970, upper bound: 0.0132544
time: 2.45 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131420, upper bound: 0.0132807
time: 1.69 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0054196, 0.0012807, -0.0055506, 0.0016288, -0.0070484, 0.0068313
1: -0.0031652, 0.0113252, -0.0032378, 0.0116753, -0.0148405, 0.0145630
2: 0.0050786, 0.0205317, 0.0050699, 0.0212018, -0.0153240, 0.0142315
3: -0.0071701, -0.0018447, -0.0074756, -0.0018362, -0.0053339, 0.0056309
4: 0.0030345, 0.0077094, 0.0028859, 0.0077109, -0.0044046, 0.0047209
5: -0.0056440, 0.0013123, -0.0057254, 0.0014994, -0.0071434, 0.0070376
6: -0.0069107, -0.0044807, -0.0070200, -0.0044159, -0.0024948, 0.0025393
7: -0.0055665, 0.0009319, -0.0056909, 0.0010873, -0.0066539, 0.0066228
8: -0.0081102, -0.0012692, -0.0085474, -0.0012354, -0.0068748, 0.0072783
9: 0.9935288, 1.0124638, 0.9930466, 1.0125208, -0.0189920, 0.0194172

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138076, upper bound: 0.0136606
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138076, upper bound: 0.0138098
time: 1.79 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0051612, 0.0006780, -0.0057294, 0.0022789, -0.0074401, 0.0064074
1: -0.0031417, 0.0107180, -0.0034888, 0.0124065, -0.0155483, 0.0142068
2: 0.0050196, 0.0193333, 0.0049194, 0.0225110, -0.0168512, 0.0139440
3: -0.0065791, -0.0018626, -0.0079005, -0.0018158, -0.0047633, 0.0060379
4: 0.0032635, 0.0077249, 0.0027831, 0.0077321, -0.0044686, 0.0049412
5: -0.0055295, 0.0009983, -0.0059512, 0.0016023, -0.0071318, 0.0069495
6: -0.0067039, -0.0045584, -0.0072025, -0.0043947, -0.0023092, 0.0026441
7: -0.0054427, 0.0006815, -0.0060450, 0.0013315, -0.0067742, 0.0067265
8: -0.0073365, -0.0013333, -0.0094222, -0.0012434, -0.0060931, 0.0080889
9: 0.9944047, 1.0124737, 0.9923125, 1.0127271, -0.0183224, 0.0201612

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131970, upper bound: 0.0133534
time: 1.83 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131420, upper bound: 0.0134122
time: 1.81 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0054196, 0.0012807, -0.0057894, 0.0024363, -0.0078558, 0.0070701
1: -0.0031652, 0.0113252, -0.0035176, 0.0125586, -0.0157238, 0.0148428
2: 0.0050786, 0.0205317, 0.0049159, 0.0228058, -0.0172689, 0.0148104
3: -0.0071701, -0.0018447, -0.0080411, -0.0018123, -0.0053578, 0.0061963
4: 0.0030345, 0.0077094, 0.0027203, 0.0077327, -0.0046175, 0.0049891
5: -0.0056440, 0.0013123, -0.0059986, 0.0016836, -0.0073277, 0.0073109
6: -0.0069107, -0.0044807, -0.0072517, -0.0043709, -0.0025398, 0.0027709
7: -0.0055665, 0.0009319, -0.0060874, 0.0014033, -0.0069699, 0.0070192
8: -0.0081102, -0.0012692, -0.0096148, -0.0012273, -0.0068829, 0.0083457
9: 0.9935288, 1.0124638, 0.9920993, 1.0127498, -0.0192210, 0.0203645

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138076, upper bound: 0.0138049
time: 2.48 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138076, upper bound: 0.0139213
time: 1.88 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054319, 0.0015153, -0.0054845, 0.0014628, -0.0068947, 0.0069998
1: -0.0034755, 0.0116595, -0.0032078, 0.0115149, -0.0149904, 0.0148672
2: 0.0048393, 0.0210627, 0.0050734, 0.0208895, -0.0154586, 0.0155999
3: -0.0072082, -0.0018386, -0.0073258, -0.0018397, -0.0053684, 0.0054872
4: 0.0030858, 0.0077584, 0.0029493, 0.0077103, -0.0046245, 0.0047875
5: -0.0057561, 0.0012026, -0.0056866, 0.0014162, -0.0071724, 0.0068892
6: -0.0069595, -0.0045103, -0.0069668, -0.0044399, -0.0025196, 0.0024565
7: -0.0058557, 0.0010171, -0.0056470, 0.0010132, -0.0068690, 0.0066641
8: -0.0084866, -0.0013259, -0.0083431, -0.0012512, -0.0072354, 0.0070172
9: 0.9933559, 1.0127705, 0.9932769, 1.0124974, -0.0191416, 0.0194936

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133875, upper bound: 0.0132544
time: 2.09 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132993, upper bound: 0.0132807
time: 2.41 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0056496, 0.0020572, -0.0055506, 0.0016288, -0.0072783, 0.0076078
1: -0.0034409, 0.0121849, -0.0032378, 0.0116753, -0.0151163, 0.0154228
2: 0.0049248, 0.0220886, 0.0050699, 0.0212018, -0.0158016, 0.0162193
3: -0.0077169, -0.0018210, -0.0074756, -0.0018362, -0.0058807, 0.0056546
4: 0.0028700, 0.0077311, 0.0028859, 0.0077109, -0.0047880, 0.0048453
5: -0.0058846, 0.0014972, -0.0057254, 0.0014994, -0.0073840, 0.0072226
6: -0.0071338, -0.0044352, -0.0070200, -0.0044159, -0.0027179, 0.0025848
7: -0.0059664, 0.0012373, -0.0056909, 0.0010873, -0.0070537, 0.0069282
8: -0.0091486, -0.0012608, -0.0085474, -0.0012354, -0.0079132, 0.0072866
9: 0.9926134, 1.0126891, 0.9930466, 1.0125208, -0.0199074, 0.0196425

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139211, upper bound: 0.0136606
time: 3.22 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139211, upper bound: 0.0138098
time: 1.95 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054319, 0.0015153, -0.0057294, 0.0022789, -0.0077108, 0.0072447
1: -0.0034755, 0.0116595, -0.0034888, 0.0124065, -0.0158820, 0.0151482
2: 0.0048393, 0.0210627, 0.0049194, 0.0225110, -0.0162548, 0.0149391
3: -0.0072082, -0.0018386, -0.0079005, -0.0018158, -0.0053924, 0.0060619
4: 0.0030858, 0.0077584, 0.0027831, 0.0077321, -0.0044823, 0.0047096
5: -0.0057561, 0.0012026, -0.0059512, 0.0016023, -0.0073584, 0.0071538
6: -0.0069595, -0.0045103, -0.0072025, -0.0043947, -0.0025648, 0.0026922
7: -0.0058557, 0.0010171, -0.0060450, 0.0013315, -0.0071873, 0.0070621
8: -0.0084866, -0.0013259, -0.0094222, -0.0012434, -0.0072431, 0.0080963
9: 0.9933559, 1.0127705, 0.9923125, 1.0127271, -0.0193713, 0.0204580

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133875, upper bound: 0.0132543
time: 2.14 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132993, upper bound: 0.0132807
time: 2.10 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0056496, 0.0020572, -0.0057894, 0.0024363, -0.0080858, 0.0078466
1: -0.0034409, 0.0121849, -0.0035176, 0.0125586, -0.0159996, 0.0157025
2: 0.0049248, 0.0220886, 0.0049159, 0.0228058, -0.0166813, 0.0155634
3: -0.0077169, -0.0018210, -0.0080411, -0.0018123, -0.0059047, 0.0062201
4: 0.0028700, 0.0077311, 0.0027203, 0.0077327, -0.0045262, 0.0048523
5: -0.0058846, 0.0014972, -0.0059986, 0.0016836, -0.0075683, 0.0074958
6: -0.0071338, -0.0044352, -0.0072517, -0.0043709, -0.0027629, 0.0028164
7: -0.0059664, 0.0012373, -0.0060874, 0.0014033, -0.0073698, 0.0073246
8: -0.0091486, -0.0012608, -0.0096148, -0.0012273, -0.0079213, 0.0083540
9: 0.9926134, 1.0126891, 0.9920993, 1.0127498, -0.0201364, 0.0205898

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139211, upper bound: 0.0136606
time: 2.08 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139211, upper bound: 0.0138098
time: 2.29 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.27 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0131970, upper bound: 0.0132544
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0131420, upper bound: 0.0132807
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0138076, upper bound: 0.0136606
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0138076, upper bound: 0.0138098
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0131970, upper bound: 0.0133534
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0131420, upper bound: 0.0134122
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0138076, upper bound: 0.0138049
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0138076, upper bound: 0.0139213
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0133875, upper bound: 0.0132544
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0132993, upper bound: 0.0132807
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0139211, upper bound: 0.0136606
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0139211, upper bound: 0.0138098
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0133875, upper bound: 0.0132543
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0132993, upper bound: 0.0132807
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0139211, upper bound: 0.0136606
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.27
Output dim: 9, lower bound: -0.0139211, upper bound: 0.0138098

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0051612, 0.0006780, -0.0052878, 0.0009420, -0.0061032, 0.0059657
1: -0.0031417, 0.0107180, -0.0031173, 0.0109643, -0.0141061, 0.0138353
2: 0.0050196, 0.0193333, 0.0050874, 0.0198467, -0.0138341, 0.0134519
3: -0.0065791, -0.0018626, -0.0068691, -0.0018531, -0.0047259, 0.0050065
4: 0.0032635, 0.0077249, 0.0031009, 0.0077075, -0.0043400, 0.0043945
5: -0.0055295, 0.0009983, -0.0055758, 0.0012027, -0.0067323, 0.0065741
6: -0.0067039, -0.0045584, -0.0068000, -0.0044836, -0.0022203, 0.0022416
7: -0.0054427, 0.0006815, -0.0054895, 0.0007992, -0.0062419, 0.0061710
8: -0.0073365, -0.0013333, -0.0076613, -0.0012967, -0.0060398, 0.0063279
9: 0.9944047, 1.0124737, 0.9939502, 1.0124261, -0.0180214, 0.0185235

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131041, upper bound: 0.0132514
time: 2.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131041, upper bound: 0.0132514
time: 1.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0051031, 0.0005471, -0.0051809, 0.0008665, -0.0059697, 0.0057280
1: -0.0031170, 0.0105673, -0.0042791, 0.0107348, -0.0138517, 0.0148463
2: 0.0050232, 0.0190455, 0.0042751, 0.0193794, -0.0136772, 0.0141281
3: -0.0064456, -0.0018664, -0.0066395, -0.0017491, -0.0046965, 0.0047731
4: 0.0033090, 0.0077242, 0.0030913, 0.0080344, -0.0047178, 0.0045943
5: -0.0055000, 0.0009340, -0.0059647, 0.0011565, -0.0066565, 0.0068987
6: -0.0066561, -0.0045722, -0.0067172, -0.0044277, -0.0022284, 0.0021449
7: -0.0054007, 0.0006285, -0.0055089, 0.0009060, -0.0063067, 0.0061375
8: -0.0071466, -0.0013479, -0.0073631, -0.0013193, -0.0058274, 0.0060152
9: 0.9946000, 1.0124540, 0.9942069, 1.0139111, -0.0193111, 0.0182471

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124713, upper bound: 0.0128364
time: 2.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125156, upper bound: 0.0126674
time: 2.41 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0054196, 0.0012807, -0.0051612, 0.0006780, -0.0060976, 0.0064420
1: -0.0031652, 0.0113252, -0.0031417, 0.0107180, -0.0138832, 0.0144669
2: 0.0050786, 0.0205317, 0.0050196, 0.0193333, -0.0134317, 0.0145773
3: -0.0071701, -0.0018447, -0.0065791, -0.0018626, -0.0053074, 0.0047343
4: 0.0030345, 0.0077094, 0.0032635, 0.0077249, -0.0045008, 0.0043279
5: -0.0056440, 0.0013123, -0.0055295, 0.0009983, -0.0066423, 0.0068418
6: -0.0069107, -0.0044807, -0.0067039, -0.0045584, -0.0023523, 0.0022231
7: -0.0055665, 0.0009319, -0.0054427, 0.0006815, -0.0062480, 0.0063746
8: -0.0081102, -0.0012692, -0.0073365, -0.0013333, -0.0067769, 0.0060673
9: 0.9935288, 1.0124638, 0.9944047, 1.0124737, -0.0189449, 0.0180591

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132544, upper bound: 0.0131969
time: 2.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132807, upper bound: 0.0131420
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0054196, 0.0012807, -0.0054196, 0.0012807, -0.0067003, 0.0067003
1: -0.0031652, 0.0113252, -0.0031652, 0.0113252, -0.0144904, 0.0144904
2: 0.0050786, 0.0205317, 0.0050786, 0.0205317, -0.0142008, 0.0142008
3: -0.0071701, -0.0018447, -0.0071701, -0.0018447, -0.0053254, 0.0053254
4: 0.0030345, 0.0077094, 0.0030345, 0.0077094, -0.0043932, 0.0043932
5: -0.0056440, 0.0013123, -0.0056440, 0.0013123, -0.0069563, 0.0069563
6: -0.0069107, -0.0044807, -0.0069107, -0.0044807, -0.0024299, 0.0024299
7: -0.0055665, 0.0009319, -0.0055665, 0.0009319, -0.0064984, 0.0064984
8: -0.0081102, -0.0012692, -0.0081102, -0.0012692, -0.0068410, 0.0068410
9: 0.9935288, 1.0124638, 0.9935288, 1.0124638, -0.0189350, 0.0189350

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132544, upper bound: 0.0132249
time: 2.09 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132807, upper bound: 0.0131684
time: 2.19 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0051612, 0.0006780, -0.0055145, 0.0016823, -0.0068435, 0.0061925
1: -0.0031417, 0.0107180, -0.0033949, 0.0118074, -0.0149492, 0.0141129
2: 0.0050196, 0.0193333, 0.0049325, 0.0213711, -0.0157128, 0.0139265
3: -0.0065791, -0.0018626, -0.0074037, -0.0018300, -0.0047491, 0.0055410
4: 0.0032635, 0.0077249, 0.0029462, 0.0077293, -0.0044658, 0.0047531
5: -0.0055295, 0.0009983, -0.0057825, 0.0013756, -0.0069052, 0.0067808
6: -0.0067039, -0.0045584, -0.0070190, -0.0044412, -0.0022627, 0.0024606
7: -0.0054427, 0.0006815, -0.0058705, 0.0011021, -0.0065448, 0.0065520
8: -0.0073365, -0.0013333, -0.0086731, -0.0012880, -0.0060485, 0.0073398
9: 0.9944047, 1.0124737, 0.9930506, 1.0126522, -0.0182474, 0.0194231

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131041, upper bound: 0.0133534
time: 2.17 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131041, upper bound: 0.0133534
time: 1.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0051031, 0.0005471, -0.0054535, 0.0015771, -0.0066803, 0.0060006
1: -0.0031170, 0.0105673, -0.0046893, 0.0116926, -0.0148096, 0.0152566
2: 0.0050232, 0.0190455, 0.0040536, 0.0211283, -0.0156953, 0.0146239
3: -0.0064456, -0.0018664, -0.0072785, -0.0017239, -0.0047217, 0.0054121
4: 0.0033090, 0.0077242, 0.0029036, 0.0080830, -0.0047740, 0.0048206
5: -0.0055000, 0.0009340, -0.0061826, 0.0013676, -0.0068676, 0.0071166
6: -0.0066561, -0.0045722, -0.0069727, -0.0043745, -0.0022816, 0.0024005
7: -0.0054007, 0.0006285, -0.0059198, 0.0012623, -0.0066630, 0.0065484
8: -0.0071466, -0.0013479, -0.0085184, -0.0013127, -0.0058339, 0.0071705
9: 0.9946000, 1.0124540, 0.9931573, 1.0143013, -0.0197013, 0.0192968

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124713, upper bound: 0.0130064
time: 1.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0125156, upper bound: 0.0128324
time: 1.99 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0054196, 0.0012807, -0.0054319, 0.0015153, -0.0069349, 0.0067126
1: -0.0031652, 0.0113252, -0.0034755, 0.0116595, -0.0148246, 0.0148007
2: 0.0050786, 0.0205317, 0.0048393, 0.0210627, -0.0155616, 0.0151552
3: -0.0071701, -0.0018447, -0.0072082, -0.0018386, -0.0053315, 0.0053635
4: 0.0030345, 0.0077094, 0.0030858, 0.0077584, -0.0047137, 0.0046236
5: -0.0056440, 0.0013123, -0.0057561, 0.0012026, -0.0068466, 0.0070684
6: -0.0069107, -0.0044807, -0.0069595, -0.0045103, -0.0024004, 0.0024788
7: -0.0055665, 0.0009319, -0.0058557, 0.0010171, -0.0065836, 0.0067876
8: -0.0081102, -0.0012692, -0.0084866, -0.0013259, -0.0067843, 0.0072174
9: 0.9935288, 1.0124638, 0.9933559, 1.0127705, -0.0192418, 0.0191079

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132544, upper bound: 0.0133875
time: 2.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132807, upper bound: 0.0132993
time: 2.53 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0054196, 0.0012807, -0.0056496, 0.0020572, -0.0074768, 0.0069303
1: -0.0031652, 0.0113252, -0.0034409, 0.0121849, -0.0153501, 0.0147661
2: 0.0050786, 0.0205317, 0.0049248, 0.0220886, -0.0161885, 0.0147786
3: -0.0071701, -0.0018447, -0.0077169, -0.0018210, -0.0053491, 0.0058722
4: 0.0030345, 0.0077094, 0.0028700, 0.0077311, -0.0046059, 0.0047767
5: -0.0056440, 0.0013123, -0.0058846, 0.0014972, -0.0071412, 0.0071969
6: -0.0069107, -0.0044807, -0.0071338, -0.0044352, -0.0024755, 0.0026531
7: -0.0055665, 0.0009319, -0.0059664, 0.0012373, -0.0068038, 0.0068983
8: -0.0081102, -0.0012692, -0.0091486, -0.0012608, -0.0068494, 0.0078795
9: 0.9935288, 1.0124638, 0.9926134, 1.0126891, -0.0191603, 0.0198504

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132544, upper bound: 0.0134070
time: 2.08 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132807, upper bound: 0.0133195
time: 2.05 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054319, 0.0015153, -0.0052878, 0.0009420, -0.0063739, 0.0068030
1: -0.0034755, 0.0116595, -0.0031173, 0.0109643, -0.0144398, 0.0147768
2: 0.0048393, 0.0210627, 0.0050874, 0.0198467, -0.0144121, 0.0155819
3: -0.0072082, -0.0018386, -0.0068691, -0.0018531, -0.0053550, 0.0050305
4: 0.0030858, 0.0077584, 0.0031009, 0.0077075, -0.0046217, 0.0046074
5: -0.0057561, 0.0012026, -0.0055758, 0.0012027, -0.0069589, 0.0067784
6: -0.0069595, -0.0045103, -0.0068000, -0.0044836, -0.0024759, 0.0022897
7: -0.0058557, 0.0010171, -0.0054895, 0.0007992, -0.0066549, 0.0065066
8: -0.0084866, -0.0013259, -0.0076613, -0.0012967, -0.0071898, 0.0063353
9: 0.9933559, 1.0127705, 0.9939502, 1.0124261, -0.0190703, 0.0188203

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132374, upper bound: 0.0132514
time: 1.49 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132374, upper bound: 0.0132514
time: 2.61 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0053801, 0.0013760, -0.0051809, 0.0008665, -0.0062466, 0.0065569
1: -0.0034523, 0.0115219, -0.0042791, 0.0107348, -0.0141870, 0.0158010
2: 0.0048429, 0.0207964, 0.0042751, 0.0193794, -0.0142551, 0.0162701
3: -0.0070866, -0.0018422, -0.0066395, -0.0017491, -0.0053375, 0.0047972
4: 0.0031292, 0.0077577, 0.0030913, 0.0080344, -0.0049052, 0.0046664
5: -0.0057209, 0.0011415, -0.0059647, 0.0011565, -0.0068775, 0.0071062
6: -0.0069153, -0.0045238, -0.0067172, -0.0044277, -0.0024877, 0.0021933
7: -0.0058151, 0.0009629, -0.0055089, 0.0009060, -0.0067211, 0.0064719
8: -0.0083104, -0.0013410, -0.0073631, -0.0013193, -0.0069912, 0.0060221
9: 0.9935376, 1.0127519, 0.9942069, 1.0139111, -0.0203735, 0.0185450

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126105, upper bound: 0.0128364
time: 1.86 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127067, upper bound: 0.0126674
time: 1.96 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056496, 0.0020572, -0.0051612, 0.0006780, -0.0063275, 0.0072185
1: -0.0034409, 0.0121849, -0.0031417, 0.0107180, -0.0141590, 0.0153267
2: 0.0049248, 0.0220886, 0.0050196, 0.0193333, -0.0139093, 0.0164588
3: -0.0077169, -0.0018210, -0.0065791, -0.0018626, -0.0058543, 0.0047581
4: 0.0028700, 0.0077311, 0.0032635, 0.0077249, -0.0048474, 0.0044676
5: -0.0058846, 0.0014972, -0.0055295, 0.0009983, -0.0068829, 0.0070267
6: -0.0071338, -0.0044352, -0.0067039, -0.0045584, -0.0025755, 0.0022686
7: -0.0059664, 0.0012373, -0.0054427, 0.0006815, -0.0066479, 0.0066800
8: -0.0091486, -0.0012608, -0.0073365, -0.0013333, -0.0078153, 0.0060757
9: 0.9926134, 1.0126891, 0.9944047, 1.0124737, -0.0198603, 0.0182844

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131970
time: 2.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134122, upper bound: 0.0131420
time: 2.07 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056496, 0.0020572, -0.0054196, 0.0012807, -0.0069303, 0.0074768
1: -0.0034409, 0.0121849, -0.0031652, 0.0113252, -0.0147661, 0.0153501
2: 0.0049248, 0.0220886, 0.0050786, 0.0205317, -0.0147786, 0.0161885
3: -0.0077169, -0.0018210, -0.0071701, -0.0018447, -0.0058722, 0.0053491
4: 0.0028700, 0.0077311, 0.0030345, 0.0077094, -0.0047767, 0.0046059
5: -0.0058846, 0.0014972, -0.0056440, 0.0013123, -0.0071969, 0.0071412
6: -0.0071338, -0.0044352, -0.0069107, -0.0044807, -0.0026531, 0.0024755
7: -0.0059664, 0.0012373, -0.0055665, 0.0009319, -0.0068983, 0.0068038
8: -0.0091486, -0.0012608, -0.0081102, -0.0012692, -0.0078795, 0.0068494
9: 0.9926134, 1.0126891, 0.9935288, 1.0124638, -0.0198504, 0.0191603

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0132249
time: 2.09 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134122, upper bound: 0.0131684
time: 2.15 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054319, 0.0015153, -0.0055145, 0.0016823, -0.0071141, 0.0070298
1: -0.0034755, 0.0116595, -0.0033949, 0.0118074, -0.0152830, 0.0150543
2: 0.0048393, 0.0210627, 0.0049325, 0.0213711, -0.0151197, 0.0149211
3: -0.0072082, -0.0018386, -0.0074037, -0.0018300, -0.0053782, 0.0055651
4: 0.0030858, 0.0077584, 0.0029462, 0.0077293, -0.0044778, 0.0045248
5: -0.0057561, 0.0012026, -0.0057825, 0.0013756, -0.0071318, 0.0069851
6: -0.0069595, -0.0045103, -0.0070190, -0.0044412, -0.0025183, 0.0025087
7: -0.0058557, 0.0010171, -0.0058705, 0.0011021, -0.0069578, 0.0068876
8: -0.0084866, -0.0013259, -0.0086731, -0.0012880, -0.0071986, 0.0073472
9: 0.9933559, 1.0127705, 0.9930506, 1.0126522, -0.0192963, 0.0197200

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132374, upper bound: 0.0132514
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132374, upper bound: 0.0132514
time: 2.03 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0053801, 0.0013760, -0.0054535, 0.0015771, -0.0069572, 0.0068296
1: -0.0034523, 0.0115219, -0.0046893, 0.0116926, -0.0151449, 0.0162112
2: 0.0048429, 0.0207964, 0.0040536, 0.0211283, -0.0151564, 0.0156239
3: -0.0070866, -0.0018422, -0.0072785, -0.0017239, -0.0053626, 0.0054363
4: 0.0031292, 0.0077577, 0.0029036, 0.0080830, -0.0048599, 0.0047543
5: -0.0057209, 0.0011415, -0.0061826, 0.0013676, -0.0070885, 0.0073241
6: -0.0069153, -0.0045238, -0.0069727, -0.0043745, -0.0025408, 0.0024489
7: -0.0058151, 0.0009629, -0.0059198, 0.0012623, -0.0070774, 0.0068827
8: -0.0083104, -0.0013410, -0.0085184, -0.0013127, -0.0069978, 0.0071774
9: 0.9935376, 1.0127519, 0.9931573, 1.0143013, -0.0207637, 0.0195947

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126151, upper bound: 0.0128466
time: 2.49 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127135, upper bound: 0.0127199
time: 1.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056496, 0.0020572, -0.0054319, 0.0015153, -0.0071648, 0.0074891
1: -0.0034409, 0.0121849, -0.0034755, 0.0116595, -0.0151004, 0.0156604
2: 0.0049248, 0.0220886, 0.0048393, 0.0210627, -0.0149007, 0.0159015
3: -0.0077169, -0.0018210, -0.0072082, -0.0018386, -0.0058783, 0.0053872
4: 0.0028700, 0.0077311, 0.0030858, 0.0077584, -0.0046335, 0.0044656
5: -0.0058846, 0.0014972, -0.0057561, 0.0012026, -0.0070872, 0.0072533
6: -0.0071338, -0.0044352, -0.0069595, -0.0045103, -0.0026235, 0.0025243
7: -0.0059664, 0.0012373, -0.0058557, 0.0010171, -0.0069835, 0.0070930
8: -0.0091486, -0.0012608, -0.0084866, -0.0013259, -0.0078227, 0.0072257
9: 0.9926134, 1.0126891, 0.9933559, 1.0127705, -0.0201572, 0.0193332

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131969
time: 2.17 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134122, upper bound: 0.0131422
time: 2.06 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056496, 0.0020572, -0.0056496, 0.0020572, -0.0077068, 0.0077068
1: -0.0034409, 0.0121849, -0.0034409, 0.0121849, -0.0156259, 0.0156259
2: 0.0049248, 0.0220886, 0.0049248, 0.0220886, -0.0155323, 0.0155323
3: -0.0077169, -0.0018210, -0.0077169, -0.0018210, -0.0058959, 0.0058959
4: 0.0028700, 0.0077311, 0.0028700, 0.0077311, -0.0045146, 0.0045146
5: -0.0058846, 0.0014972, -0.0058846, 0.0014972, -0.0073818, 0.0073818
6: -0.0071338, -0.0044352, -0.0071338, -0.0044352, -0.0026986, 0.0026986
7: -0.0059664, 0.0012373, -0.0059664, 0.0012373, -0.0072037, 0.0072037
8: -0.0091486, -0.0012608, -0.0091486, -0.0012608, -0.0078878, 0.0078878
9: 0.9926134, 1.0126891, 0.9926134, 1.0126891, -0.0200757, 0.0200757

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0132249
time: 2.04 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134122, upper bound: 0.0131684
time: 3.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.21 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0131041, upper bound: 0.0132514
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0131041, upper bound: 0.0132514
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0124713, upper bound: 0.0128364
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0125156, upper bound: 0.0126674
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132544, upper bound: 0.0131969
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132807, upper bound: 0.0131420
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132544, upper bound: 0.0132249
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132807, upper bound: 0.0131684
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0131041, upper bound: 0.0133534
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0131041, upper bound: 0.0133534
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0124713, upper bound: 0.0130064
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0125156, upper bound: 0.0128324
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132544, upper bound: 0.0133875
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132807, upper bound: 0.0132993
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132544, upper bound: 0.0134070
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132807, upper bound: 0.0133195
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132374, upper bound: 0.0132514
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132374, upper bound: 0.0132514
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0126105, upper bound: 0.0128364
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0127067, upper bound: 0.0126674
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131970
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0134122, upper bound: 0.0131420
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0132249
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0134122, upper bound: 0.0131684
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132374, upper bound: 0.0132514
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0132374, upper bound: 0.0132514
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0126151, upper bound: 0.0128466
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0127135, upper bound: 0.0127199
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131969
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0134122, upper bound: 0.0131422
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0132249
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.21
Output dim: 9, lower bound: -0.0134122, upper bound: 0.0131684

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0049609, 0.0003057, -0.0052878, 0.0009420, -0.0059029, 0.0055934
1: -0.0030504, 0.0101631, -0.0031173, 0.0109643, -0.0140147, 0.0132804
2: 0.0050342, 0.0182746, 0.0050874, 0.0198467, -0.0138167, 0.0123755
3: -0.0061099, -0.0018762, -0.0068691, -0.0018531, -0.0042568, 0.0049930
4: 0.0034217, 0.0077218, 0.0031009, 0.0077075, -0.0041550, 0.0043903
5: -0.0054237, 0.0007782, -0.0055758, 0.0012027, -0.0066264, 0.0063541
6: -0.0065347, -0.0046030, -0.0068000, -0.0044836, -0.0020511, 0.0021970
7: -0.0052814, 0.0004984, -0.0054895, 0.0007992, -0.0060806, 0.0059879
8: -0.0066433, -0.0013796, -0.0076613, -0.0012967, -0.0053466, 0.0062817
9: 0.9950958, 1.0124000, 0.9939502, 1.0124261, -0.0173303, 0.0184498

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127830, upper bound: 0.0126334
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127330, upper bound: 0.0127193
time: 1.67 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0048713, 0.0006532, -0.0052878, 0.0009420, -0.0058133, 0.0059409
1: -0.0042528, 0.0099823, -0.0031173, 0.0109643, -0.0152171, 0.0130996
2: 0.0041961, 0.0179093, 0.0050874, 0.0198467, -0.0148338, 0.0122059
3: -0.0059269, -0.0017667, -0.0068691, -0.0018531, -0.0040738, 0.0051024
4: 0.0033944, 0.0080598, 0.0031009, 0.0077075, -0.0042766, 0.0048516
5: -0.0058538, 0.0007553, -0.0055758, 0.0012027, -0.0070565, 0.0063312
6: -0.0064800, -0.0045439, -0.0068000, -0.0044836, -0.0019964, 0.0022561
7: -0.0053103, 0.0006915, -0.0054895, 0.0007992, -0.0061095, 0.0061811
8: -0.0064084, -0.0013998, -0.0076613, -0.0012967, -0.0051117, 0.0062615
9: 0.9952855, 1.0139272, 0.9939502, 1.0124261, -0.0171407, 0.0199770

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127830, upper bound: 0.0126334
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127330, upper bound: 0.0127193
time: 1.97 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0051612, 0.0006780, -0.0059011, 0.0059291
1: -0.0030752, 0.0107814, -0.0031417, 0.0107180, -0.0137932, 0.0139232
2: 0.0050926, 0.0194936, 0.0050196, 0.0193333, -0.0134149, 0.0135329
3: -0.0067108, -0.0018580, -0.0065791, -0.0018626, -0.0048482, 0.0047210
4: 0.0031853, 0.0077066, 0.0032635, 0.0077249, -0.0043219, 0.0043236
5: -0.0055368, 0.0011005, -0.0055295, 0.0009983, -0.0065351, 0.0066301
6: -0.0067431, -0.0045243, -0.0067039, -0.0045584, -0.0021847, 0.0021795
7: -0.0054098, 0.0007239, -0.0054427, 0.0006815, -0.0060913, 0.0061666
8: -0.0074292, -0.0013145, -0.0073365, -0.0013333, -0.0060959, 0.0060220
9: 0.9942025, 1.0123928, 0.9944047, 1.0124737, -0.0182712, 0.0179880

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132514, upper bound: 0.0131041
time: 2.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132514, upper bound: 0.0131420
time: 1.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0051103, 0.0007790, -0.0051031, 0.0005471, -0.0056575, 0.0058821
1: -0.0042372, 0.0105434, -0.0031170, 0.0105673, -0.0148045, 0.0136604
2: 0.0042801, 0.0190120, 0.0050232, 0.0190455, -0.0140958, 0.0133684
3: -0.0064745, -0.0017543, -0.0064456, -0.0018664, -0.0046081, 0.0046914
4: 0.0031767, 0.0080335, 0.0033090, 0.0077242, -0.0045205, 0.0047033
5: -0.0059276, 0.0010535, -0.0055000, 0.0009340, -0.0068616, 0.0065535
6: -0.0066593, -0.0044693, -0.0066561, -0.0045722, -0.0020871, 0.0021868
7: -0.0054251, 0.0008406, -0.0054007, 0.0006285, -0.0060536, 0.0062413
8: -0.0071200, -0.0013386, -0.0071466, -0.0013479, -0.0057721, 0.0058080
9: 0.9944739, 1.0138772, 0.9946000, 1.0124540, -0.0179802, 0.0192772

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128364, upper bound: 0.0124713
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126674, upper bound: 0.0125156
time: 2.28 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0054196, 0.0012807, -0.0065039, 0.0061874
1: -0.0030752, 0.0107814, -0.0031652, 0.0113252, -0.0144004, 0.0139466
2: 0.0050926, 0.0194936, 0.0050786, 0.0205317, -0.0141836, 0.0131491
3: -0.0067108, -0.0018580, -0.0071701, -0.0018447, -0.0048661, 0.0053121
4: 0.0031853, 0.0077066, 0.0030345, 0.0077094, -0.0042077, 0.0043892
5: -0.0055368, 0.0011005, -0.0056440, 0.0013123, -0.0068490, 0.0067445
6: -0.0067431, -0.0045243, -0.0069107, -0.0044807, -0.0022623, 0.0023863
7: -0.0054098, 0.0007239, -0.0055665, 0.0009319, -0.0063417, 0.0062904
8: -0.0074292, -0.0013145, -0.0081102, -0.0012692, -0.0061600, 0.0067957
9: 0.9942025, 1.0123928, 0.9935288, 1.0124638, -0.0182613, 0.0188640

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132531, upper bound: 0.0131355
time: 1.97 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132531, upper bound: 0.0131684
time: 2.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0051103, 0.0007790, -0.0053606, 0.0011278, -0.0062381, 0.0061396
1: -0.0042372, 0.0105434, -0.0031397, 0.0111696, -0.0154068, 0.0136831
2: 0.0042801, 0.0190120, 0.0050822, 0.0202320, -0.0149200, 0.0129637
3: -0.0064745, -0.0017543, -0.0070331, -0.0018485, -0.0046260, 0.0052789
4: 0.0031767, 0.0080335, 0.0030827, 0.0077087, -0.0043901, 0.0048088
5: -0.0059276, 0.0010535, -0.0056117, 0.0012454, -0.0071730, 0.0066652
6: -0.0066593, -0.0044693, -0.0068612, -0.0044950, -0.0021643, 0.0023919
7: -0.0054251, 0.0008406, -0.0055231, 0.0008671, -0.0062922, 0.0063637
8: -0.0071200, -0.0013386, -0.0079144, -0.0012844, -0.0058356, 0.0065758
9: 0.9944739, 1.0138772, 0.9937307, 1.0124439, -0.0179700, 0.0201465

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128437, upper bound: 0.0125325
time: 2.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126870, upper bound: 0.0125840
time: 1.47 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0049609, 0.0003057, -0.0055145, 0.0016823, -0.0066432, 0.0058202
1: -0.0030504, 0.0101631, -0.0033949, 0.0118074, -0.0148578, 0.0135580
2: 0.0050342, 0.0182746, 0.0049325, 0.0213711, -0.0156954, 0.0128500
3: -0.0061099, -0.0018762, -0.0074037, -0.0018300, -0.0042799, 0.0055275
4: 0.0034217, 0.0077218, 0.0029462, 0.0077293, -0.0043075, 0.0047489
5: -0.0054237, 0.0007782, -0.0057825, 0.0013756, -0.0067993, 0.0065608
6: -0.0065347, -0.0046030, -0.0070190, -0.0044412, -0.0020935, 0.0024160
7: -0.0052814, 0.0004984, -0.0058705, 0.0011021, -0.0063835, 0.0063689
8: -0.0066433, -0.0013796, -0.0086731, -0.0012880, -0.0053553, 0.0072935
9: 0.9950958, 1.0124000, 0.9930506, 1.0126522, -0.0175563, 0.0193495

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127830, upper bound: 0.0127171
time: 2.04 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127330, upper bound: 0.0128367
time: 1.37 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0048713, 0.0006532, -0.0055145, 0.0016823, -0.0065536, 0.0061677
1: -0.0042528, 0.0099823, -0.0033949, 0.0118074, -0.0160602, 0.0133772
2: 0.0041961, 0.0179093, 0.0049325, 0.0213711, -0.0167125, 0.0126805
3: -0.0059269, -0.0017667, -0.0074037, -0.0018300, -0.0040969, 0.0056370
4: 0.0033944, 0.0080598, 0.0029462, 0.0077293, -0.0043349, 0.0051136
5: -0.0058538, 0.0007553, -0.0057825, 0.0013756, -0.0072294, 0.0065379
6: -0.0064800, -0.0045439, -0.0070190, -0.0044412, -0.0020387, 0.0024751
7: -0.0053103, 0.0006915, -0.0058705, 0.0011021, -0.0064124, 0.0065620
8: -0.0064084, -0.0013998, -0.0086731, -0.0012880, -0.0051204, 0.0072733
9: 0.9952855, 1.0139272, 0.9930506, 1.0126522, -0.0173667, 0.0208766

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127830, upper bound: 0.0127171
time: 1.37 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127330, upper bound: 0.0128367
time: 1.47 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0049249, 0.0002318, -0.0054535, 0.0015771, -0.0065021, 0.0056853
1: -0.0029030, 0.0102336, -0.0046893, 0.0116926, -0.0145956, 0.0149229
2: 0.0050393, 0.0183484, 0.0040536, 0.0211283, -0.0156819, 0.0138526
3: -0.0060307, -0.0019027, -0.0072785, -0.0017239, -0.0043068, 0.0053758
4: 0.0037200, 0.0077219, 0.0029036, 0.0080830, -0.0043630, 0.0048184
5: -0.0053537, 0.0004760, -0.0061826, 0.0013676, -0.0067213, 0.0066586
6: -0.0065249, -0.0048131, -0.0069727, -0.0043745, -0.0021504, 0.0021596
7: -0.0052333, 0.0003302, -0.0059198, 0.0012623, -0.0064956, 0.0062500
8: -0.0066987, -0.0014507, -0.0085184, -0.0013127, -0.0053860, 0.0070677
9: 0.9953943, 1.0122818, 0.9931573, 1.0143013, -0.0189070, 0.0191245

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124703, upper bound: 0.0127544
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124703, upper bound: 0.0128324
time: 2.26 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0054319, 0.0015153, -0.0067384, 0.0061997
1: -0.0030752, 0.0107814, -0.0034755, 0.0116595, -0.0147346, 0.0142569
2: 0.0050926, 0.0194936, 0.0048393, 0.0210627, -0.0155448, 0.0141108
3: -0.0067108, -0.0018580, -0.0072082, -0.0018386, -0.0048722, 0.0053501
4: 0.0031853, 0.0077066, 0.0030858, 0.0077584, -0.0045348, 0.0046208
5: -0.0055368, 0.0011005, -0.0057561, 0.0012026, -0.0067393, 0.0068567
6: -0.0067431, -0.0045243, -0.0069595, -0.0045103, -0.0022328, 0.0024352
7: -0.0054098, 0.0007239, -0.0058557, 0.0010171, -0.0064269, 0.0065796
8: -0.0074292, -0.0013145, -0.0084866, -0.0013259, -0.0061033, 0.0071720
9: 0.9942025, 1.0123928, 0.9933559, 1.0127705, -0.0185680, 0.0190369

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132514, upper bound: 0.0132374
time: 1.55 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132514, upper bound: 0.0132993
time: 1.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0051103, 0.0007790, -0.0053801, 0.0013760, -0.0064864, 0.0061591
1: -0.0042372, 0.0105434, -0.0034523, 0.0115219, -0.0157592, 0.0139957
2: 0.0042801, 0.0190120, 0.0048429, 0.0207964, -0.0162379, 0.0139463
3: -0.0064745, -0.0017543, -0.0070866, -0.0018422, -0.0046323, 0.0053323
4: 0.0031767, 0.0080335, 0.0031292, 0.0077577, -0.0045811, 0.0049043
5: -0.0059276, 0.0010535, -0.0057209, 0.0011415, -0.0070691, 0.0067745
6: -0.0066593, -0.0044693, -0.0069153, -0.0045238, -0.0021355, 0.0024460
7: -0.0054251, 0.0008406, -0.0058151, 0.0009629, -0.0063880, 0.0066556
8: -0.0071200, -0.0013386, -0.0083104, -0.0013410, -0.0057789, 0.0069718
9: 0.9944739, 1.0138772, 0.9935376, 1.0127519, -0.0182781, 0.0203395

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128364, upper bound: 0.0126105
time: 1.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126674, upper bound: 0.0127067
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0056496, 0.0020572, -0.0072804, 0.0064174
1: -0.0030752, 0.0107814, -0.0034409, 0.0121849, -0.0152601, 0.0142224
2: 0.0050926, 0.0194936, 0.0049248, 0.0220886, -0.0161714, 0.0137269
3: -0.0067108, -0.0018580, -0.0077169, -0.0018210, -0.0048898, 0.0058589
4: 0.0031853, 0.0077066, 0.0028700, 0.0077311, -0.0044204, 0.0047726
5: -0.0055368, 0.0011005, -0.0058846, 0.0014972, -0.0070340, 0.0069852
6: -0.0067431, -0.0045243, -0.0071338, -0.0044352, -0.0023079, 0.0026095
7: -0.0054098, 0.0007239, -0.0059664, 0.0012373, -0.0066471, 0.0066903
8: -0.0074292, -0.0013145, -0.0091486, -0.0012608, -0.0061684, 0.0078341
9: 0.9942025, 1.0123928, 0.9926134, 1.0126891, -0.0184866, 0.0197794

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132531, upper bound: 0.0132563
time: 3.22 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132531, upper bound: 0.0133195
time: 2.10 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0051103, 0.0007790, -0.0055950, 0.0019089, -0.0070192, 0.0063740
1: -0.0042372, 0.0105434, -0.0034173, 0.0120430, -0.0162802, 0.0139607
2: 0.0042801, 0.0190120, 0.0049282, 0.0218124, -0.0169224, 0.0135414
3: -0.0064745, -0.0017543, -0.0075897, -0.0018247, -0.0046498, 0.0058354
4: 0.0031767, 0.0080335, 0.0029144, 0.0077305, -0.0045538, 0.0051191
5: -0.0059276, 0.0010535, -0.0058420, 0.0014338, -0.0073614, 0.0068955
6: -0.0066593, -0.0044693, -0.0070880, -0.0044489, -0.0022104, 0.0026187
7: -0.0054251, 0.0008406, -0.0059251, 0.0011729, -0.0065980, 0.0067657
8: -0.0071200, -0.0013386, -0.0089655, -0.0012763, -0.0058437, 0.0076269
9: 0.9944739, 1.0138772, 0.9927991, 1.0126704, -0.0181965, 0.0210781

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128437, upper bound: 0.0126567
time: 2.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126870, upper bound: 0.0127594
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0052072, 0.0009276, -0.0052878, 0.0009420, -0.0061492, 0.0062153
1: -0.0033785, 0.0110444, -0.0031173, 0.0109643, -0.0143428, 0.0141617
2: 0.0048537, 0.0198871, 0.0050874, 0.0198467, -0.0143943, 0.0144119
3: -0.0066934, -0.0018529, -0.0068691, -0.0018531, -0.0048403, 0.0050162
4: 0.0032535, 0.0077554, 0.0031009, 0.0077075, -0.0044540, 0.0046029
5: -0.0056140, 0.0009710, -0.0055758, 0.0012027, -0.0068167, 0.0065468
6: -0.0067691, -0.0045577, -0.0068000, -0.0044836, -0.0022855, 0.0022423
7: -0.0056762, 0.0008122, -0.0054895, 0.0007992, -0.0064754, 0.0063018
8: -0.0077117, -0.0013715, -0.0076613, -0.0012967, -0.0064150, 0.0062898
9: 0.9941286, 1.0126923, 0.9939502, 1.0124261, -0.0182975, 0.0187421

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130332, upper bound: 0.0126334
time: 2.21 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129891, upper bound: 0.0127193
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0051662, 0.0010100, -0.0052878, 0.0009420, -0.0061082, 0.0062977
1: -0.0047385, 0.0109705, -0.0031173, 0.0109643, -0.0157028, 0.0140878
2: 0.0039256, 0.0197224, 0.0050874, 0.0198467, -0.0154236, 0.0143533
3: -0.0066060, -0.0017421, -0.0068691, -0.0018531, -0.0047529, 0.0051271
4: 0.0031991, 0.0081321, 0.0031009, 0.0077075, -0.0045084, 0.0050312
5: -0.0060753, 0.0009770, -0.0055758, 0.0012027, -0.0072781, 0.0065528
6: -0.0067385, -0.0044876, -0.0068000, -0.0044836, -0.0022549, 0.0023124
7: -0.0057272, 0.0010442, -0.0054895, 0.0007992, -0.0065263, 0.0065338
8: -0.0076038, -0.0013933, -0.0076613, -0.0012967, -0.0063071, 0.0062680
9: 0.9941720, 1.0144305, 0.9939502, 1.0124261, -0.0182542, 0.0204803

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130332, upper bound: 0.0126334
time: 1.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129891, upper bound: 0.0127193
time: 1.54 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0051612, 0.0006780, -0.0061120, 0.0066204
1: -0.0033474, 0.0115872, -0.0031417, 0.0107180, -0.0140655, 0.0147290
2: 0.0049381, 0.0209475, 0.0050196, 0.0193333, -0.0138927, 0.0153197
3: -0.0072176, -0.0018351, -0.0065791, -0.0018626, -0.0053549, 0.0047440
4: 0.0030337, 0.0077283, 0.0032635, 0.0077249, -0.0046611, 0.0044648
5: -0.0057231, 0.0012698, -0.0055295, 0.0009983, -0.0067214, 0.0067994
6: -0.0069494, -0.0044826, -0.0067039, -0.0045584, -0.0023910, 0.0022213
7: -0.0057918, 0.0010129, -0.0054427, 0.0006815, -0.0064733, 0.0064556
8: -0.0083977, -0.0013057, -0.0073365, -0.0013333, -0.0070644, 0.0060308
9: 0.9933559, 1.0126143, 0.9944047, 1.0124737, -0.0191178, 0.0182095

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131041
time: 2.20 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131420
time: 1.63 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0053676, 0.0013628, -0.0051031, 0.0005471, -0.0059147, 0.0064659
1: -0.0046402, 0.0114577, -0.0031170, 0.0105673, -0.0152075, 0.0145747
2: 0.0040588, 0.0206815, 0.0050232, 0.0190455, -0.0145941, 0.0152982
3: -0.0070806, -0.0017295, -0.0064456, -0.0018664, -0.0052142, 0.0047161
4: 0.0029936, 0.0080821, 0.0033090, 0.0077242, -0.0047306, 0.0047731
5: -0.0061322, 0.0012563, -0.0055000, 0.0009340, -0.0070662, 0.0067563
6: -0.0069009, -0.0044170, -0.0066561, -0.0045722, -0.0023287, 0.0022391
7: -0.0058313, 0.0011818, -0.0054007, 0.0006285, -0.0064598, 0.0065825
8: -0.0082289, -0.0013316, -0.0071466, -0.0013479, -0.0068810, 0.0058150
9: 0.9934702, 1.0142620, 0.9946000, 1.0124540, -0.0189838, 0.0196620

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130064, upper bound: 0.0124713
time: 3.10 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128324, upper bound: 0.0125156
time: 1.50 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0054196, 0.0012807, -0.0067147, 0.0068788
1: -0.0033474, 0.0115872, -0.0031652, 0.0113252, -0.0146726, 0.0147524
2: 0.0049381, 0.0209475, 0.0050786, 0.0205317, -0.0147612, 0.0150433
3: -0.0072176, -0.0018351, -0.0071701, -0.0018447, -0.0053729, 0.0053350
4: 0.0030337, 0.0077283, 0.0030345, 0.0077094, -0.0045862, 0.0046016
5: -0.0057231, 0.0012698, -0.0056440, 0.0013123, -0.0070354, 0.0069138
6: -0.0069494, -0.0044826, -0.0069107, -0.0044807, -0.0024687, 0.0024281
7: -0.0057918, 0.0010129, -0.0055665, 0.0009319, -0.0067237, 0.0065794
8: -0.0083977, -0.0013057, -0.0081102, -0.0012692, -0.0071286, 0.0068045
9: 0.9933559, 1.0126143, 0.9935288, 1.0124638, -0.0191079, 0.0190855

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133540, upper bound: 0.0131355
time: 2.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133540, upper bound: 0.0131684
time: 1.92 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0053676, 0.0013628, -0.0053606, 0.0011278, -0.0064954, 0.0067234
1: -0.0046402, 0.0114577, -0.0031397, 0.0111696, -0.0158098, 0.0145974
2: 0.0040588, 0.0206815, 0.0050822, 0.0202320, -0.0155103, 0.0150066
3: -0.0070806, -0.0017295, -0.0070331, -0.0018485, -0.0052320, 0.0053037
4: 0.0029936, 0.0080821, 0.0030827, 0.0077087, -0.0047151, 0.0049995
5: -0.0061322, 0.0012563, -0.0056117, 0.0012454, -0.0073777, 0.0068680
6: -0.0069009, -0.0044170, -0.0068612, -0.0044950, -0.0024059, 0.0024442
7: -0.0058313, 0.0011818, -0.0055231, 0.0008671, -0.0066984, 0.0067049
8: -0.0082289, -0.0013316, -0.0079144, -0.0012844, -0.0069445, 0.0065828
9: 0.9934702, 1.0142620, 0.9937307, 1.0124439, -0.0189737, 0.0205313

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130096, upper bound: 0.0125325
time: 1.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128428, upper bound: 0.0125840
time: 2.18 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0052072, 0.0009276, -0.0055145, 0.0016823, -0.0068895, 0.0064421
1: -0.0033785, 0.0110444, -0.0033949, 0.0118074, -0.0151859, 0.0144392
2: 0.0048537, 0.0198871, 0.0049325, 0.0213711, -0.0151025, 0.0137529
3: -0.0066934, -0.0018529, -0.0074037, -0.0018300, -0.0048634, 0.0055507
4: 0.0032535, 0.0077554, 0.0029462, 0.0077293, -0.0042874, 0.0045206
5: -0.0056140, 0.0009710, -0.0057825, 0.0013756, -0.0069896, 0.0067535
6: -0.0067691, -0.0045577, -0.0070190, -0.0044412, -0.0023278, 0.0024613
7: -0.0056762, 0.0008122, -0.0058705, 0.0011021, -0.0067782, 0.0066827
8: -0.0077117, -0.0013715, -0.0086731, -0.0012880, -0.0064237, 0.0073016
9: 0.9941286, 1.0126923, 0.9930506, 1.0126522, -0.0185235, 0.0196418

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130332, upper bound: 0.0126514
time: 2.14 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129891, upper bound: 0.0127505
time: 2.03 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0051662, 0.0010100, -0.0055145, 0.0016823, -0.0068485, 0.0065245
1: -0.0047385, 0.0109705, -0.0033949, 0.0118074, -0.0165460, 0.0143654
2: 0.0039256, 0.0197224, 0.0049325, 0.0213711, -0.0161344, 0.0137086
3: -0.0066060, -0.0017421, -0.0074037, -0.0018300, -0.0047760, 0.0056616
4: 0.0031991, 0.0081321, 0.0029462, 0.0077293, -0.0044197, 0.0049874
5: -0.0060753, 0.0009770, -0.0057825, 0.0013756, -0.0074510, 0.0067595
6: -0.0067385, -0.0044876, -0.0070190, -0.0044412, -0.0022973, 0.0025314
7: -0.0057272, 0.0010442, -0.0058705, 0.0011021, -0.0068292, 0.0069147
8: -0.0076038, -0.0013933, -0.0086731, -0.0012880, -0.0063158, 0.0072799
9: 0.9941720, 1.0144305, 0.9930506, 1.0126522, -0.0184802, 0.0213799

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130332, upper bound: 0.0126514
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0129891, upper bound: 0.0127505
time: 3.98 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0054319, 0.0015153, -0.0069493, 0.0068911
1: -0.0033474, 0.0115872, -0.0034755, 0.0116595, -0.0150069, 0.0150628
2: 0.0049381, 0.0209475, 0.0048393, 0.0210627, -0.0148838, 0.0147719
3: -0.0072176, -0.0018351, -0.0072082, -0.0018386, -0.0053790, 0.0053731
4: 0.0030337, 0.0077283, 0.0030858, 0.0077584, -0.0044510, 0.0044614
5: -0.0057231, 0.0012698, -0.0057561, 0.0012026, -0.0069257, 0.0070260
6: -0.0069494, -0.0044826, -0.0069595, -0.0045103, -0.0024391, 0.0024769
7: -0.0057918, 0.0010129, -0.0058557, 0.0010171, -0.0068089, 0.0068686
8: -0.0083977, -0.0013057, -0.0084866, -0.0013259, -0.0070718, 0.0071809
9: 0.9933559, 1.0126143, 0.9933559, 1.0127705, -0.0194147, 0.0192584

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131049
time: 2.28 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131422
time: 2.09 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0053676, 0.0013628, -0.0053801, 0.0013760, -0.0067436, 0.0067429
1: -0.0046402, 0.0114577, -0.0034523, 0.0115219, -0.0161622, 0.0149100
2: 0.0040588, 0.0206815, 0.0048429, 0.0207964, -0.0155918, 0.0147931
3: -0.0070806, -0.0017295, -0.0070866, -0.0018422, -0.0052383, 0.0053571
4: 0.0029936, 0.0080821, 0.0031292, 0.0077577, -0.0046780, 0.0048454
5: -0.0061322, 0.0012563, -0.0057209, 0.0011415, -0.0072737, 0.0069772
6: -0.0069009, -0.0044170, -0.0069153, -0.0045238, -0.0023771, 0.0024983
7: -0.0058313, 0.0011818, -0.0058151, 0.0009629, -0.0067942, 0.0069968
8: -0.0082289, -0.0013316, -0.0083104, -0.0013410, -0.0068879, 0.0069789
9: 0.9934702, 1.0142620, 0.9935376, 1.0127519, -0.0192817, 0.0207244

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130064, upper bound: 0.0125202
time: 2.03 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128381, upper bound: 0.0125820
time: 1.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0056496, 0.0020572, -0.0074912, 0.0071087
1: -0.0033474, 0.0115872, -0.0034409, 0.0121849, -0.0155324, 0.0150282
2: 0.0049381, 0.0209475, 0.0049248, 0.0220886, -0.0155154, 0.0143951
3: -0.0072176, -0.0018351, -0.0077169, -0.0018210, -0.0053966, 0.0058819
4: 0.0030337, 0.0077283, 0.0028700, 0.0077311, -0.0043245, 0.0045106
5: -0.0057231, 0.0012698, -0.0058846, 0.0014972, -0.0072203, 0.0071545
6: -0.0069494, -0.0044826, -0.0071338, -0.0044352, -0.0025142, 0.0026512
7: -0.0057918, 0.0010129, -0.0059664, 0.0012373, -0.0070291, 0.0069793
8: -0.0083977, -0.0013057, -0.0091486, -0.0012608, -0.0071369, 0.0078430
9: 0.9933559, 1.0126143, 0.9926134, 1.0126891, -0.0193332, 0.0200009

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133540, upper bound: 0.0131355
time: 2.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133540, upper bound: 0.0131684
time: 2.11 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0053676, 0.0013628, -0.0055950, 0.0019089, -0.0072765, 0.0069578
1: -0.0046402, 0.0114577, -0.0034173, 0.0120430, -0.0166833, 0.0148750
2: 0.0040588, 0.0206815, 0.0049282, 0.0218124, -0.0162824, 0.0144203
3: -0.0070806, -0.0017295, -0.0075897, -0.0018247, -0.0052559, 0.0058602
4: 0.0029936, 0.0080821, 0.0029144, 0.0077305, -0.0045452, 0.0049360
5: -0.0061322, 0.0012563, -0.0058420, 0.0014338, -0.0075660, 0.0070983
6: -0.0069009, -0.0044170, -0.0070880, -0.0044489, -0.0024520, 0.0026710
7: -0.0058313, 0.0011818, -0.0059251, 0.0011729, -0.0070042, 0.0071069
8: -0.0082289, -0.0013316, -0.0089655, -0.0012763, -0.0069526, 0.0076339
9: 0.9934702, 1.0142620, 0.9927991, 1.0126704, -0.0192002, 0.0214629

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130096, upper bound: 0.0125741
time: 2.19 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128489, upper bound: 0.0126451
time: 2.26 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.20 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0127830, upper bound: 0.0126334
NS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0127330, upper bound: 0.0127193
NS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0127830, upper bound: 0.0126334
NS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0127330, upper bound: 0.0127193
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0132514, upper bound: 0.0131041
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0132514, upper bound: 0.0131420
NS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0128364, upper bound: 0.0124713
NS_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0126674, upper bound: 0.0125156
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0132531, upper bound: 0.0131355
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0132531, upper bound: 0.0131684
NS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0128437, upper bound: 0.0125325
NS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0126870, upper bound: 0.0125840
NS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0127830, upper bound: 0.0127171
NS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0127330, upper bound: 0.0128367
NS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0127830, upper bound: 0.0127171
NS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0127330, upper bound: 0.0128367
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0124703, upper bound: 0.0127544
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0124703, upper bound: 0.0128324
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0132514, upper bound: 0.0132374
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0132514, upper bound: 0.0132993
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0128364, upper bound: 0.0126105
NS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0126674, upper bound: 0.0127067
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0132531, upper bound: 0.0132563
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0132531, upper bound: 0.0133195
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0128437, upper bound: 0.0126567
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0126870, upper bound: 0.0127594
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0130332, upper bound: 0.0126334
NS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0129891, upper bound: 0.0127193
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0130332, upper bound: 0.0126334
NS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0129891, upper bound: 0.0127193
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131041
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131420
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0130064, upper bound: 0.0124713
NS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0128324, upper bound: 0.0125156
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0133540, upper bound: 0.0131355
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0133540, upper bound: 0.0131684
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0130096, upper bound: 0.0125325
NS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0128428, upper bound: 0.0125840
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0130332, upper bound: 0.0126514
NS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0129891, upper bound: 0.0127505
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0130332, upper bound: 0.0126514
NS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0129891, upper bound: 0.0127505
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131049
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0133534, upper bound: 0.0131422
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0130064, upper bound: 0.0125202
NS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0128381, upper bound: 0.0125820
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0133540, upper bound: 0.0131355
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0133540, upper bound: 0.0131684
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0130096, upper bound: 0.0125741
NS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 6.20
Output dim: 9, lower bound: -0.0128489, upper bound: 0.0126451

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0049609, 0.0003057, -0.0055288, 0.0057288
1: -0.0030752, 0.0107814, -0.0030504, 0.0101631, -0.0132383, 0.0138318
2: 0.0050926, 0.0194936, 0.0050342, 0.0182746, -0.0123385, 0.0135154
3: -0.0067108, -0.0018580, -0.0061099, -0.0018762, -0.0048347, 0.0042519
4: 0.0031853, 0.0077066, 0.0034217, 0.0077218, -0.0043178, 0.0041386
5: -0.0055368, 0.0011005, -0.0054237, 0.0007782, -0.0063150, 0.0065242
6: -0.0067431, -0.0045243, -0.0065347, -0.0046030, -0.0021401, 0.0020104
7: -0.0054098, 0.0007239, -0.0052814, 0.0004984, -0.0059082, 0.0060053
8: -0.0074292, -0.0013145, -0.0066433, -0.0013796, -0.0060496, 0.0053287
9: 0.9942025, 1.0123928, 0.9950958, 1.0124000, -0.0181975, 0.0172969

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126334, upper bound: 0.0127830
time: 3.20 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127193, upper bound: 0.0127330
time: 1.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0048713, 0.0006532, -0.0058763, 0.0056392
1: -0.0030752, 0.0107814, -0.0042528, 0.0099823, -0.0130575, 0.0150342
2: 0.0050926, 0.0194936, 0.0041961, 0.0179093, -0.0121689, 0.0145325
3: -0.0067108, -0.0018580, -0.0059269, -0.0017667, -0.0049441, 0.0040689
4: 0.0031853, 0.0077066, 0.0033944, 0.0080598, -0.0047791, 0.0042602
5: -0.0055368, 0.0011005, -0.0058538, 0.0007553, -0.0062921, 0.0069543
6: -0.0067431, -0.0045243, -0.0064800, -0.0045439, -0.0021992, 0.0019556
7: -0.0054098, 0.0007239, -0.0053103, 0.0006915, -0.0061013, 0.0060342
8: -0.0074292, -0.0013145, -0.0064084, -0.0013998, -0.0060294, 0.0050939
9: 0.9942025, 1.0123928, 0.9952855, 1.0139272, -0.0197247, 0.0171073

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126334, upper bound: 0.0127830
time: 2.14 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127193, upper bound: 0.0127330
time: 1.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0052232, 0.0007678, -0.0059910, 0.0059910
1: -0.0030752, 0.0107814, -0.0030752, 0.0107814, -0.0138566, 0.0138566
2: 0.0050926, 0.0194936, 0.0050926, 0.0194936, -0.0131319, 0.0131319
3: -0.0067108, -0.0018580, -0.0067108, -0.0018580, -0.0048528, 0.0048528
4: 0.0031853, 0.0077066, 0.0031853, 0.0077066, -0.0042037, 0.0042037
5: -0.0055368, 0.0011005, -0.0055368, 0.0011005, -0.0066373, 0.0066373
6: -0.0067431, -0.0045243, -0.0067431, -0.0045243, -0.0022187, 0.0022187
7: -0.0054098, 0.0007239, -0.0054098, 0.0007239, -0.0061337, 0.0061337
8: -0.0074292, -0.0013145, -0.0074292, -0.0013145, -0.0061147, 0.0061147
9: 0.9942025, 1.0123928, 0.9942025, 1.0123928, -0.0181903, 0.0181903

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0128296
time: 1.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0127957
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0051103, 0.0007790, -0.0060021, 0.0058782
1: -0.0030752, 0.0107814, -0.0042372, 0.0105434, -0.0136186, 0.0150186
2: 0.0050926, 0.0194936, 0.0042801, 0.0190120, -0.0128884, 0.0141469
3: -0.0067108, -0.0018580, -0.0064745, -0.0017543, -0.0049566, 0.0046165
4: 0.0031853, 0.0077066, 0.0031767, 0.0080335, -0.0046654, 0.0043241
5: -0.0055368, 0.0011005, -0.0059276, 0.0010535, -0.0065903, 0.0070281
6: -0.0067431, -0.0045243, -0.0066593, -0.0044693, -0.0022738, 0.0021350
7: -0.0054098, 0.0007239, -0.0054251, 0.0008406, -0.0062504, 0.0061490
8: -0.0074292, -0.0013145, -0.0071200, -0.0013386, -0.0060906, 0.0058054
9: 0.9942025, 1.0123928, 0.9944739, 1.0138772, -0.0196747, 0.0179189

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0128296
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0127957
time: 2.29 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0052072, 0.0009276, -0.0061507, 0.0059750
1: -0.0030752, 0.0107814, -0.0033785, 0.0110444, -0.0141196, 0.0141599
2: 0.0050926, 0.0194936, 0.0048537, 0.0198871, -0.0143749, 0.0140930
3: -0.0067108, -0.0018580, -0.0066934, -0.0018529, -0.0048579, 0.0048354
4: 0.0031853, 0.0077066, 0.0032535, 0.0077554, -0.0045304, 0.0044531
5: -0.0055368, 0.0011005, -0.0056140, 0.0009710, -0.0065078, 0.0067145
6: -0.0067431, -0.0045243, -0.0067691, -0.0045577, -0.0021854, 0.0022447
7: -0.0054098, 0.0007239, -0.0056762, 0.0008122, -0.0062220, 0.0064001
8: -0.0074292, -0.0013145, -0.0077117, -0.0013715, -0.0060577, 0.0063972
9: 0.9942025, 1.0123928, 0.9941286, 1.0126923, -0.0184898, 0.0182641

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126334, upper bound: 0.0130332
time: 2.06 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127193, upper bound: 0.0129891
time: 2.42 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0051662, 0.0010100, -0.0062331, 0.0059340
1: -0.0030752, 0.0107814, -0.0047385, 0.0109705, -0.0140457, 0.0155199
2: 0.0050926, 0.0194936, 0.0039256, 0.0197224, -0.0143162, 0.0151223
3: -0.0067108, -0.0018580, -0.0066060, -0.0017421, -0.0049687, 0.0047480
4: 0.0031853, 0.0077066, 0.0031991, 0.0081321, -0.0049468, 0.0045075
5: -0.0055368, 0.0011005, -0.0060753, 0.0009770, -0.0065138, 0.0071759
6: -0.0067431, -0.0045243, -0.0067385, -0.0044876, -0.0022555, 0.0022142
7: -0.0054098, 0.0007239, -0.0057272, 0.0010442, -0.0064540, 0.0064510
8: -0.0074292, -0.0013145, -0.0076038, -0.0013933, -0.0060359, 0.0062893
9: 0.9942025, 1.0123928, 0.9941720, 1.0144305, -0.0202280, 0.0182208

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126334, upper bound: 0.0130332
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127193, upper bound: 0.0129891
time: 2.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0054340, 0.0014592, -0.0066823, 0.0062019
1: -0.0030752, 0.0107814, -0.0033474, 0.0115872, -0.0146624, 0.0141289
2: 0.0050926, 0.0194936, 0.0049381, 0.0209475, -0.0150262, 0.0137095
3: -0.0067108, -0.0018580, -0.0072176, -0.0018351, -0.0048755, 0.0053596
4: 0.0031853, 0.0077066, 0.0030337, 0.0077283, -0.0044161, 0.0045821
5: -0.0055368, 0.0011005, -0.0057231, 0.0012698, -0.0068066, 0.0068237
6: -0.0067431, -0.0045243, -0.0069494, -0.0044826, -0.0022605, 0.0024251
7: -0.0054098, 0.0007239, -0.0057918, 0.0010129, -0.0064227, 0.0065157
8: -0.0074292, -0.0013145, -0.0083977, -0.0013057, -0.0061235, 0.0070832
9: 0.9942025, 1.0123928, 0.9933559, 1.0126143, -0.0184118, 0.0190369

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0130667
time: 2.91 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0130325
time: 2.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0052232, 0.0007678, -0.0053676, 0.0013628, -0.0065860, 0.0061354
1: -0.0030752, 0.0107814, -0.0046402, 0.0114577, -0.0145329, 0.0154216
2: 0.0050926, 0.0194936, 0.0040588, 0.0206815, -0.0149243, 0.0147372
3: -0.0067108, -0.0018580, -0.0070806, -0.0017295, -0.0049814, 0.0052225
4: 0.0031853, 0.0077066, 0.0029936, 0.0080821, -0.0048697, 0.0046963
5: -0.0055368, 0.0011005, -0.0061322, 0.0012563, -0.0067931, 0.0072327
6: -0.0067431, -0.0045243, -0.0069009, -0.0044170, -0.0023261, 0.0023766
7: -0.0054098, 0.0007239, -0.0058313, 0.0011818, -0.0065916, 0.0065552
8: -0.0074292, -0.0013145, -0.0082289, -0.0013316, -0.0060976, 0.0069144
9: 0.9942025, 1.0123928, 0.9934702, 1.0142620, -0.0200595, 0.0189226

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0130667
time: 3.57 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0130325
time: 3.20 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0052072, 0.0009276, -0.0051086, 0.0005514, -0.0057586, 0.0060362
1: -0.0033785, 0.0110444, -0.0028904, 0.0106294, -0.0140078, 0.0139348
2: 0.0048537, 0.0198871, 0.0051037, 0.0191469, -0.0136097, 0.0143965
3: -0.0066934, -0.0018529, -0.0064506, -0.0018900, -0.0048033, 0.0045976
4: 0.0032535, 0.0077554, 0.0035202, 0.0077052, -0.0044518, 0.0041009
5: -0.0056140, 0.0009710, -0.0054219, 0.0007363, -0.0063503, 0.0063929
6: -0.0067691, -0.0045577, -0.0066669, -0.0047344, -0.0020347, 0.0021092
7: -0.0056762, 0.0008122, -0.0053129, 0.0004773, -0.0061535, 0.0061251
8: -0.0077117, -0.0013715, -0.0072109, -0.0013980, -0.0063137, 0.0058394
9: 0.9941286, 1.0126923, 0.9947575, 1.0122440, -0.0181153, 0.0179348

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130453, upper bound: 0.0131080
time: 2.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130453, upper bound: 0.0131080
time: 2.40 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0051662, 0.0010100, -0.0051086, 0.0005514, -0.0057176, 0.0061186
1: -0.0047385, 0.0109705, -0.0028904, 0.0106294, -0.0153679, 0.0138610
2: 0.0039256, 0.0197224, 0.0051037, 0.0191469, -0.0146390, 0.0143379
3: -0.0066060, -0.0017421, -0.0064506, -0.0018900, -0.0047160, 0.0047085
4: 0.0031991, 0.0081321, 0.0035202, 0.0077052, -0.0045061, 0.0045537
5: -0.0060753, 0.0009770, -0.0054219, 0.0007363, -0.0068116, 0.0063989
6: -0.0067385, -0.0044876, -0.0066669, -0.0047344, -0.0020041, 0.0021793
7: -0.0057272, 0.0010442, -0.0053129, 0.0004773, -0.0062045, 0.0063571
8: -0.0076038, -0.0013933, -0.0072109, -0.0013980, -0.0062058, 0.0058176
9: 0.9941720, 1.0144305, 0.9947575, 1.0122440, -0.0180720, 0.0196730

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128621, upper bound: 0.0126334
time: 2.54 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128621, upper bound: 0.0126334
time: 2.58 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0049609, 0.0003057, -0.0057397, 0.0064201
1: -0.0033474, 0.0115872, -0.0030504, 0.0101631, -0.0135106, 0.0146376
2: 0.0049381, 0.0209475, 0.0050342, 0.0182746, -0.0128162, 0.0153023
3: -0.0072176, -0.0018351, -0.0061099, -0.0018762, -0.0053414, 0.0042749
4: 0.0030337, 0.0077283, 0.0034217, 0.0077218, -0.0046570, 0.0042957
5: -0.0057231, 0.0012698, -0.0054237, 0.0007782, -0.0065014, 0.0066935
6: -0.0069494, -0.0044826, -0.0065347, -0.0046030, -0.0023464, 0.0020521
7: -0.0057918, 0.0010129, -0.0052814, 0.0004984, -0.0062902, 0.0062943
8: -0.0083977, -0.0013057, -0.0066433, -0.0013796, -0.0070182, 0.0053376
9: 0.9933559, 1.0126143, 0.9950958, 1.0124000, -0.0190442, 0.0175184

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127171, upper bound: 0.0127830
time: 2.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128367, upper bound: 0.0127330
time: 1.97 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0048713, 0.0006532, -0.0060872, 0.0063305
1: -0.0033474, 0.0115872, -0.0042528, 0.0099823, -0.0133298, 0.0158400
2: 0.0049381, 0.0209475, 0.0041961, 0.0179093, -0.0126467, 0.0163194
3: -0.0072176, -0.0018351, -0.0059269, -0.0017667, -0.0054509, 0.0040918
4: 0.0030337, 0.0077283, 0.0033944, 0.0080598, -0.0050261, 0.0043339
5: -0.0057231, 0.0012698, -0.0058538, 0.0007553, -0.0064785, 0.0071236
6: -0.0069494, -0.0044826, -0.0064800, -0.0045439, -0.0024056, 0.0019974
7: -0.0057918, 0.0010129, -0.0053103, 0.0006915, -0.0064834, 0.0063232
8: -0.0083977, -0.0013057, -0.0064084, -0.0013998, -0.0069980, 0.0051027
9: 0.9933559, 1.0126143, 0.9952855, 1.0139272, -0.0205714, 0.0173288

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127171, upper bound: 0.0127830
time: 2.35 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128367, upper bound: 0.0127330
time: 1.93 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0053676, 0.0013628, -0.0049249, 0.0002318, -0.0055994, 0.0062877
1: -0.0046402, 0.0114577, -0.0029030, 0.0102336, -0.0148738, 0.0143607
2: 0.0040588, 0.0206815, 0.0050393, 0.0183484, -0.0138229, 0.0152849
3: -0.0070806, -0.0017295, -0.0060307, -0.0019027, -0.0051779, 0.0043013
4: 0.0029936, 0.0080821, 0.0037200, 0.0077219, -0.0047283, 0.0043622
5: -0.0061322, 0.0012563, -0.0053537, 0.0004760, -0.0066082, 0.0066100
6: -0.0069009, -0.0044170, -0.0065249, -0.0048131, -0.0020878, 0.0021079
7: -0.0058313, 0.0011818, -0.0052333, 0.0003302, -0.0061615, 0.0064151
8: -0.0082289, -0.0013316, -0.0066987, -0.0014507, -0.0067782, 0.0053671
9: 0.9934702, 1.0142620, 0.9953943, 1.0122818, -0.0188116, 0.0188677

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127544, upper bound: 0.0124703
time: 2.02 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127544, upper bound: 0.0124703
time: 3.08 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0052232, 0.0007678, -0.0062019, 0.0066823
1: -0.0033474, 0.0115872, -0.0030752, 0.0107814, -0.0141289, 0.0146624
2: 0.0049381, 0.0209475, 0.0050926, 0.0194936, -0.0137095, 0.0150262
3: -0.0072176, -0.0018351, -0.0067108, -0.0018580, -0.0053596, 0.0048755
4: 0.0030337, 0.0077283, 0.0031853, 0.0077066, -0.0045821, 0.0044161
5: -0.0057231, 0.0012698, -0.0055368, 0.0011005, -0.0068237, 0.0068066
6: -0.0069494, -0.0044826, -0.0067431, -0.0045243, -0.0024251, 0.0022605
7: -0.0057918, 0.0010129, -0.0054098, 0.0007239, -0.0065157, 0.0064227
8: -0.0083977, -0.0013057, -0.0074292, -0.0013145, -0.0070832, 0.0061235
9: 0.9933559, 1.0126143, 0.9942025, 1.0123928, -0.0190369, 0.0184118

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127234, upper bound: 0.0128296
time: 1.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128472, upper bound: 0.0127957
time: 2.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0051103, 0.0007790, -0.0062130, 0.0065695
1: -0.0033474, 0.0115872, -0.0042372, 0.0105434, -0.0138909, 0.0158245
2: 0.0049381, 0.0209475, 0.0042801, 0.0190120, -0.0134660, 0.0160412
3: -0.0072176, -0.0018351, -0.0064745, -0.0017543, -0.0054633, 0.0046394
4: 0.0030337, 0.0077283, 0.0031767, 0.0080335, -0.0049998, 0.0045366
5: -0.0057231, 0.0012698, -0.0059276, 0.0010535, -0.0067767, 0.0071974
6: -0.0069494, -0.0044826, -0.0066593, -0.0044693, -0.0024802, 0.0021767
7: -0.0057918, 0.0010129, -0.0054251, 0.0008406, -0.0066324, 0.0064380
8: -0.0083977, -0.0013057, -0.0071200, -0.0013386, -0.0070591, 0.0058143
9: 0.9933559, 1.0126143, 0.9944739, 1.0138772, -0.0205213, 0.0181404

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127234, upper bound: 0.0128296
time: 1.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128472, upper bound: 0.0127957
time: 1.97 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0053676, 0.0013628, -0.0051842, 0.0007455, -0.0061131, 0.0065470
1: -0.0046402, 0.0114577, -0.0029059, 0.0108433, -0.0154835, 0.0143636
2: 0.0040588, 0.0206815, 0.0050979, 0.0195519, -0.0147424, 0.0149931
3: -0.0070806, -0.0017295, -0.0066237, -0.0018859, -0.0051947, 0.0048943
4: 0.0029936, 0.0080821, 0.0034946, 0.0077066, -0.0047129, 0.0045098
5: -0.0061322, 0.0012563, -0.0054547, 0.0007899, -0.0069221, 0.0067110
6: -0.0069009, -0.0044170, -0.0067312, -0.0047397, -0.0021613, 0.0023142
7: -0.0058313, 0.0011818, -0.0053494, 0.0005430, -0.0063743, 0.0065312
8: -0.0082289, -0.0013316, -0.0074742, -0.0013840, -0.0068449, 0.0061426
9: 0.9934702, 1.0142620, 0.9945230, 1.0122563, -0.0187861, 0.0197389

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127608, upper bound: 0.0125317
time: 2.06 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127608, upper bound: 0.0125317
time: 2.22 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0052072, 0.0009276, -0.0053218, 0.0012641, -0.0064713, 0.0062493
1: -0.0033785, 0.0110444, -0.0031472, 0.0114519, -0.0148304, 0.0141915
2: 0.0048537, 0.0198871, 0.0049488, 0.0206292, -0.0142850, 0.0137376
3: -0.0066934, -0.0018529, -0.0069618, -0.0018669, -0.0048265, 0.0051089
4: 0.0032535, 0.0077554, 0.0033653, 0.0077270, -0.0042818, 0.0040270
5: -0.0056140, 0.0009710, -0.0055977, 0.0009028, -0.0065167, 0.0065687
6: -0.0067691, -0.0045577, -0.0068776, -0.0046847, -0.0020843, 0.0023199
7: -0.0056762, 0.0008122, -0.0057072, 0.0007710, -0.0064472, 0.0065195
8: -0.0077117, -0.0013715, -0.0081947, -0.0013883, -0.0063234, 0.0068232
9: 0.9941286, 1.0126923, 0.9938785, 1.0124531, -0.0183244, 0.0188138

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130453, upper bound: 0.0131062
time: 2.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130453, upper bound: 0.0131062
time: 2.27 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0051662, 0.0010100, -0.0053218, 0.0012641, -0.0064303, 0.0063317
1: -0.0047385, 0.0109705, -0.0031472, 0.0114519, -0.0161904, 0.0141177
2: 0.0039256, 0.0197224, 0.0049488, 0.0206292, -0.0153170, 0.0136933
3: -0.0066060, -0.0017421, -0.0069618, -0.0018669, -0.0047391, 0.0052197
4: 0.0031991, 0.0081321, 0.0033653, 0.0077270, -0.0044141, 0.0044938
5: -0.0060753, 0.0009770, -0.0055977, 0.0009028, -0.0069781, 0.0065747
6: -0.0067385, -0.0044876, -0.0068776, -0.0046847, -0.0020538, 0.0023900
7: -0.0057272, 0.0010442, -0.0057072, 0.0007710, -0.0064982, 0.0067515
8: -0.0076038, -0.0013933, -0.0081947, -0.0013883, -0.0062155, 0.0068014
9: 0.9941720, 1.0144305, 0.9938785, 1.0124531, -0.0182811, 0.0205520

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128621, upper bound: 0.0126514
time: 1.88 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128621, upper bound: 0.0126514
time: 2.23 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0052072, 0.0009276, -0.0063616, 0.0066664
1: -0.0033474, 0.0115872, -0.0033785, 0.0110444, -0.0143918, 0.0149657
2: 0.0049381, 0.0209475, 0.0048537, 0.0198871, -0.0137157, 0.0147547
3: -0.0072176, -0.0018351, -0.0066934, -0.0018529, -0.0053646, 0.0048583
4: 0.0030337, 0.0077283, 0.0032535, 0.0077554, -0.0044469, 0.0042710
5: -0.0057231, 0.0012698, -0.0056140, 0.0009710, -0.0066941, 0.0068838
6: -0.0069494, -0.0044826, -0.0067691, -0.0045577, -0.0023917, 0.0022865
7: -0.0057918, 0.0010129, -0.0056762, 0.0008122, -0.0066041, 0.0066891
8: -0.0083977, -0.0013057, -0.0077117, -0.0013715, -0.0070262, 0.0064060
9: 0.9933559, 1.0126143, 0.9941286, 1.0126923, -0.0193365, 0.0184856

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127182, upper bound: 0.0127900
time: 2.36 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128411, upper bound: 0.0127544
time: 2.42 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0051662, 0.0010100, -0.0064440, 0.0066254
1: -0.0033474, 0.0115872, -0.0047385, 0.0109705, -0.0143180, 0.0163258
2: 0.0049381, 0.0209475, 0.0039256, 0.0197224, -0.0136713, 0.0157866
3: -0.0072176, -0.0018351, -0.0066060, -0.0017421, -0.0054755, 0.0047709
4: 0.0030337, 0.0077283, 0.0031991, 0.0081321, -0.0049136, 0.0044033
5: -0.0057231, 0.0012698, -0.0060753, 0.0009770, -0.0067001, 0.0073452
6: -0.0069494, -0.0044826, -0.0067385, -0.0044876, -0.0024618, 0.0022559
7: -0.0057918, 0.0010129, -0.0057272, 0.0010442, -0.0068361, 0.0067401
8: -0.0083977, -0.0013057, -0.0076038, -0.0013933, -0.0070045, 0.0062981
9: 0.9933559, 1.0126143, 0.9941720, 1.0144305, -0.0210747, 0.0184423

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127182, upper bound: 0.0127900
time: 2.26 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128411, upper bound: 0.0127544
time: 2.56 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0053676, 0.0013628, -0.0051899, 0.0009601, -0.0063277, 0.0065527
1: -0.0046402, 0.0114577, -0.0032120, 0.0111712, -0.0158114, 0.0146697
2: 0.0040588, 0.0206815, 0.0048593, 0.0200606, -0.0147851, 0.0147799
3: -0.0070806, -0.0017295, -0.0066482, -0.0018786, -0.0052020, 0.0049187
4: 0.0029936, 0.0080821, 0.0035332, 0.0077555, -0.0046727, 0.0043575
5: -0.0061322, 0.0012563, -0.0055465, 0.0006820, -0.0068143, 0.0068028
6: -0.0069009, -0.0044170, -0.0067744, -0.0047561, -0.0021448, 0.0023574
7: -0.0058313, 0.0011818, -0.0056619, 0.0006466, -0.0064779, 0.0068437
8: -0.0082289, -0.0013316, -0.0078332, -0.0014430, -0.0067859, 0.0065017
9: 0.9934702, 1.0142620, 0.9943468, 1.0125587, -0.0190885, 0.0199152

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127575, upper bound: 0.0125199
time: 1.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127575, upper bound: 0.0125199
time: 3.06 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0054340, 0.0014592, -0.0068932, 0.0068932
1: -0.0033474, 0.0115872, -0.0033474, 0.0115872, -0.0149347, 0.0149347
2: 0.0049381, 0.0209475, 0.0049381, 0.0209475, -0.0143782, 0.0143782
3: -0.0072176, -0.0018351, -0.0072176, -0.0018351, -0.0053825, 0.0053825
4: 0.0030337, 0.0077283, 0.0030337, 0.0077283, -0.0043204, 0.0043204
5: -0.0057231, 0.0012698, -0.0057231, 0.0012698, -0.0069930, 0.0069930
6: -0.0069494, -0.0044826, -0.0069494, -0.0044826, -0.0024668, 0.0024668
7: -0.0057918, 0.0010129, -0.0057918, 0.0010129, -0.0068048, 0.0068048
8: -0.0083977, -0.0013057, -0.0083977, -0.0013057, -0.0070921, 0.0070921
9: 0.9933559, 1.0126143, 0.9933559, 1.0126143, -0.0192584, 0.0192584

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0128344
time: 2.51 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128524, upper bound: 0.0128083
time: 2.27 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0054340, 0.0014592, -0.0053676, 0.0013628, -0.0067968, 0.0068268
1: -0.0033474, 0.0115872, -0.0046402, 0.0114577, -0.0148052, 0.0162275
2: 0.0049381, 0.0209475, 0.0040588, 0.0206815, -0.0142837, 0.0154080
3: -0.0072176, -0.0018351, -0.0070806, -0.0017295, -0.0054881, 0.0052455
4: 0.0030337, 0.0077283, 0.0029936, 0.0080821, -0.0047883, 0.0044482
5: -0.0057231, 0.0012698, -0.0061322, 0.0012563, -0.0069794, 0.0074020
6: -0.0069494, -0.0044826, -0.0069009, -0.0044170, -0.0025324, 0.0024183
7: -0.0057918, 0.0010129, -0.0058313, 0.0011818, -0.0069736, 0.0068442
8: -0.0083977, -0.0013057, -0.0082289, -0.0013316, -0.0070662, 0.0069232
9: 0.9933559, 1.0126143, 0.9934702, 1.0142620, -0.0209061, 0.0191441

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0128344
time: 2.53 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128524, upper bound: 0.0128083
time: 2.52 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0053676, 0.0013628, -0.0054097, 0.0015013, -0.0068689, 0.0067725
1: -0.0046402, 0.0114577, -0.0031624, 0.0116949, -0.0163351, 0.0146201
2: 0.0040588, 0.0206815, 0.0049439, 0.0210881, -0.0154868, 0.0144070
3: -0.0070806, -0.0017295, -0.0071601, -0.0018621, -0.0052185, 0.0054307
4: 0.0029936, 0.0080821, 0.0033236, 0.0077283, -0.0045398, 0.0044430
5: -0.0061322, 0.0012563, -0.0056515, 0.0009747, -0.0071069, 0.0069078
6: -0.0069009, -0.0044170, -0.0069510, -0.0046843, -0.0022166, 0.0025340
7: -0.0058313, 0.0011818, -0.0057661, 0.0008403, -0.0066716, 0.0069479
8: -0.0082289, -0.0013316, -0.0085006, -0.0013751, -0.0068538, 0.0071690
9: 0.9934702, 1.0142620, 0.9936050, 1.0124654, -0.0189952, 0.0206569

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127639, upper bound: 0.0125741
time: 1.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127639, upper bound: 0.0125741
time: 2.21 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.69 seconds
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0126334, upper bound: 0.0127830
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127193, upper bound: 0.0127330
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0126334, upper bound: 0.0127830
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127193, upper bound: 0.0127330
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0128296
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0127957
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0128296
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0127957
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0126334, upper bound: 0.0130332
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127193, upper bound: 0.0129891
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0126334, upper bound: 0.0130332
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127193, upper bound: 0.0129891
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0130667
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0130325
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0130667
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0130325
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0130453, upper bound: 0.0131080
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0130453, upper bound: 0.0131080
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128621, upper bound: 0.0126334
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128621, upper bound: 0.0126334
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127171, upper bound: 0.0127830
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128367, upper bound: 0.0127330
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127171, upper bound: 0.0127830
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128367, upper bound: 0.0127330
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127544, upper bound: 0.0124703
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127544, upper bound: 0.0124703
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127234, upper bound: 0.0128296
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128472, upper bound: 0.0127957
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127234, upper bound: 0.0128296
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128472, upper bound: 0.0127957
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127608, upper bound: 0.0125317
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127608, upper bound: 0.0125317
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0130453, upper bound: 0.0131062
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0130453, upper bound: 0.0131062
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128621, upper bound: 0.0126514
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128621, upper bound: 0.0126514
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127182, upper bound: 0.0127900
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128411, upper bound: 0.0127544
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127182, upper bound: 0.0127900
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128411, upper bound: 0.0127544
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127575, upper bound: 0.0125199
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127575, upper bound: 0.0125199
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0128344
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128524, upper bound: 0.0128083
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127246, upper bound: 0.0128344
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0128524, upper bound: 0.0128083
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127639, upper bound: 0.0125741
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.69
Output dim: 9, lower bound: -0.0127639, upper bound: 0.0125741

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0050441, 0.0003909, -0.0052072, 0.0009276, -0.0059716, 0.0055981
1: -0.0028513, 0.0104438, -0.0033785, 0.0110444, -0.0138957, 0.0138223
2: 0.0051091, 0.0187944, 0.0048537, 0.0198871, -0.0143604, 0.0133151
3: -0.0062963, -0.0018953, -0.0066934, -0.0018529, -0.0044434, 0.0047981
4: 0.0036019, 0.0077043, 0.0032535, 0.0077554, -0.0040384, 0.0044508
5: -0.0053856, 0.0006408, -0.0056140, 0.0009710, -0.0063566, 0.0062548
6: -0.0066107, -0.0047713, -0.0067691, -0.0045577, -0.0020529, 0.0019978
7: -0.0052340, 0.0004082, -0.0056762, 0.0008122, -0.0060462, 0.0060844
8: -0.0069803, -0.0014147, -0.0077117, -0.0013715, -0.0056088, 0.0062970
9: 0.9950126, 1.0122132, 0.9941286, 1.0126923, -0.0176797, 0.0180846

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131080, upper bound: 0.0130453
time: 3.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131080, upper bound: 0.0132919
time: 1.99 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0050441, 0.0003909, -0.0051662, 0.0010100, -0.0060540, 0.0055571
1: -0.0028513, 0.0104438, -0.0047385, 0.0109705, -0.0138218, 0.0151823
2: 0.0051091, 0.0187944, 0.0039256, 0.0197224, -0.0143017, 0.0143444
3: -0.0062963, -0.0018953, -0.0066060, -0.0017421, -0.0045542, 0.0047107
4: 0.0036019, 0.0077043, 0.0031991, 0.0081321, -0.0044912, 0.0045052
5: -0.0053856, 0.0006408, -0.0060753, 0.0009770, -0.0063626, 0.0067161
6: -0.0066107, -0.0047713, -0.0067385, -0.0044876, -0.0021230, 0.0019672
7: -0.0052340, 0.0004082, -0.0057272, 0.0010442, -0.0062782, 0.0061354
8: -0.0069803, -0.0014147, -0.0076038, -0.0013933, -0.0055870, 0.0061891
9: 0.9950126, 1.0122132, 0.9941720, 1.0144305, -0.0194179, 0.0180413

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124702, upper bound: 0.0128621
time: 1.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124702, upper bound: 0.0129891
time: 2.42 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0050441, 0.0003909, -0.0054340, 0.0014592, -0.0065033, 0.0058250
1: -0.0028513, 0.0104438, -0.0033474, 0.0115872, -0.0144385, 0.0137913
2: 0.0051091, 0.0187944, 0.0049381, 0.0209475, -0.0150119, 0.0129272
3: -0.0062963, -0.0018953, -0.0072176, -0.0018351, -0.0044414, 0.0053223
4: 0.0036019, 0.0077043, 0.0030337, 0.0077283, -0.0039070, 0.0045766
5: -0.0053856, 0.0006408, -0.0057231, 0.0012698, -0.0066554, 0.0063639
6: -0.0066107, -0.0047713, -0.0069494, -0.0044826, -0.0021280, 0.0021781
7: -0.0052340, 0.0004082, -0.0057918, 0.0010129, -0.0062469, 0.0062001
8: -0.0069803, -0.0014147, -0.0083977, -0.0013057, -0.0056746, 0.0069830
9: 0.9950126, 1.0122132, 0.9933559, 1.0126143, -0.0176016, 0.0188574

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131121, upper bound: 0.0130621
time: 3.09 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131121, upper bound: 0.0133272
time: 2.11 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050494, 0.0005016, -0.0054065, 0.0013961, -0.0064455, 0.0059080
1: -0.0032541, 0.0104475, -0.0033031, 0.0115300, -0.0147841, 0.0137506
2: 0.0048604, 0.0188066, 0.0049417, 0.0208313, -0.0151121, 0.0130433
3: -0.0063182, -0.0017625, -0.0071523, -0.0018417, -0.0044764, 0.0053898
4: 0.0035452, 0.0078751, 0.0031051, 0.0077278, -0.0040251, 0.0047023
5: -0.0056878, 0.0006980, -0.0056925, 0.0011947, -0.0068825, 0.0063905
6: -0.0066158, -0.0047351, -0.0069284, -0.0045282, -0.0020876, 0.0021933
7: -0.0053120, 0.0004784, -0.0057591, 0.0009519, -0.0062639, 0.0062375
8: -0.0069938, -0.0013965, -0.0083229, -0.0013225, -0.0056712, 0.0069264
9: 0.9949242, 1.0128475, 0.9934869, 1.0125785, -0.0176542, 0.0193607

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132832, upper bound: 0.0130621
time: 2.49 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132832, upper bound: 0.0133272
time: 1.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0050441, 0.0003909, -0.0053676, 0.0013628, -0.0064069, 0.0057585
1: -0.0028513, 0.0104438, -0.0046402, 0.0114577, -0.0143090, 0.0150841
2: 0.0051091, 0.0187944, 0.0040588, 0.0206815, -0.0149100, 0.0139549
3: -0.0062963, -0.0018953, -0.0070806, -0.0017295, -0.0045668, 0.0051853
4: 0.0036019, 0.0077043, 0.0029936, 0.0080821, -0.0043606, 0.0046908
5: -0.0053856, 0.0006408, -0.0061322, 0.0012563, -0.0066419, 0.0067730
6: -0.0066107, -0.0047713, -0.0069009, -0.0044170, -0.0021937, 0.0021296
7: -0.0052340, 0.0004082, -0.0058313, 0.0011818, -0.0064157, 0.0062395
8: -0.0069803, -0.0014147, -0.0082289, -0.0013316, -0.0056487, 0.0068141
9: 0.9950126, 1.0122132, 0.9934702, 1.0142620, -0.0192493, 0.0187430

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0128973
time: 2.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0130325
time: 2.09 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0050494, 0.0005016, -0.0053380, 0.0012958, -0.0063452, 0.0058395
1: -0.0032541, 0.0104475, -0.0045983, 0.0113959, -0.0146500, 0.0150458
2: 0.0048604, 0.0188066, 0.0040627, 0.0205553, -0.0149939, 0.0140710
3: -0.0063182, -0.0017625, -0.0070096, -0.0017362, -0.0045819, 0.0052471
4: 0.0035452, 0.0078751, 0.0030654, 0.0080816, -0.0044789, 0.0048096
5: -0.0056878, 0.0006980, -0.0061041, 0.0011787, -0.0068665, 0.0068022
6: -0.0066158, -0.0047351, -0.0068782, -0.0044622, -0.0021536, 0.0021431
7: -0.0053120, 0.0004784, -0.0057970, 0.0011227, -0.0064347, 0.0062754
8: -0.0069938, -0.0013965, -0.0081482, -0.0013491, -0.0056446, 0.0067517
9: 0.9949242, 1.0128475, 0.9936076, 1.0142281, -0.0193039, 0.0192400

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0128973
time: 2.25 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0130325
time: 2.41 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0050143, 0.0005183, -0.0051086, 0.0005514, -0.0055658, 0.0056269
1: -0.0031505, 0.0106850, -0.0028904, 0.0106294, -0.0137798, 0.0135754
2: 0.0048707, 0.0191374, 0.0051037, 0.0191469, -0.0135935, 0.0135785
3: -0.0062496, -0.0018891, -0.0064506, -0.0018900, -0.0043596, 0.0045526
4: 0.0036633, 0.0077530, 0.0035202, 0.0077052, -0.0040420, 0.0040957
5: -0.0054492, 0.0005028, -0.0054219, 0.0007363, -0.0061855, 0.0059247
6: -0.0066274, -0.0047932, -0.0066669, -0.0047344, -0.0018931, 0.0018737
7: -0.0055207, 0.0005132, -0.0053129, 0.0004773, -0.0059980, 0.0058261
8: -0.0072304, -0.0014740, -0.0072109, -0.0013980, -0.0058324, 0.0057369
9: 0.9949508, 1.0125092, 0.9947575, 1.0122440, -0.0172932, 0.0177517

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0129291
time: 1.57 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0131080
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050623, 0.0006684, -0.0051086, 0.0005514, -0.0056138, 0.0057771
1: -0.0036095, 0.0107399, -0.0028904, 0.0106294, -0.0142388, 0.0136303
2: 0.0045802, 0.0192696, 0.0051037, 0.0191469, -0.0138622, 0.0137722
3: -0.0063586, -0.0017561, -0.0064506, -0.0018900, -0.0044685, 0.0046945
4: 0.0035831, 0.0079551, 0.0035202, 0.0077052, -0.0041221, 0.0043080
5: -0.0057807, 0.0006074, -0.0054219, 0.0007363, -0.0065171, 0.0060293
6: -0.0066578, -0.0047618, -0.0066669, -0.0047344, -0.0019234, 0.0019052
7: -0.0055772, 0.0005985, -0.0053129, 0.0004773, -0.0060545, 0.0059114
8: -0.0073216, -0.0014500, -0.0072109, -0.0013980, -0.0059236, 0.0057609
9: 0.9947715, 1.0132531, 0.9947575, 1.0122440, -0.0174724, 0.0184956

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0129291
time: 2.15 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0131080
time: 2.07 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0050143, 0.0005183, -0.0053218, 0.0012641, -0.0062785, 0.0058401
1: -0.0031505, 0.0106850, -0.0031472, 0.0114519, -0.0146024, 0.0138322
2: 0.0048707, 0.0191374, 0.0049488, 0.0206292, -0.0142711, 0.0129141
3: -0.0062496, -0.0018891, -0.0069618, -0.0018669, -0.0043827, 0.0050727
4: 0.0036633, 0.0077530, 0.0033653, 0.0077270, -0.0037890, 0.0040217
5: -0.0054492, 0.0005028, -0.0055977, 0.0009028, -0.0063519, 0.0061005
6: -0.0066274, -0.0047932, -0.0068776, -0.0046847, -0.0019427, 0.0020844
7: -0.0055207, 0.0005132, -0.0057072, 0.0007710, -0.0062917, 0.0062204
8: -0.0072304, -0.0014740, -0.0081947, -0.0013883, -0.0058421, 0.0067207
9: 0.9949508, 1.0125092, 0.9938785, 1.0124531, -0.0175023, 0.0186307

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0129276
time: 1.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0131062
time: 1.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0050623, 0.0006684, -0.0053218, 0.0012641, -0.0063265, 0.0059902
1: -0.0036095, 0.0107399, -0.0031472, 0.0114519, -0.0150614, 0.0138871
2: 0.0045802, 0.0192696, 0.0049488, 0.0206292, -0.0145062, 0.0131128
3: -0.0063586, -0.0017561, -0.0069618, -0.0018669, -0.0044917, 0.0052057
4: 0.0035831, 0.0079551, 0.0033653, 0.0077270, -0.0039032, 0.0042286
5: -0.0057807, 0.0006074, -0.0055977, 0.0009028, -0.0066835, 0.0062051
6: -0.0066578, -0.0047618, -0.0068776, -0.0046847, -0.0019731, 0.0021159
7: -0.0055772, 0.0005985, -0.0057072, 0.0007710, -0.0063482, 0.0063057
8: -0.0073216, -0.0014500, -0.0081947, -0.0013883, -0.0059333, 0.0067447
9: 0.9947715, 1.0132531, 0.9938785, 1.0124531, -0.0176815, 0.0193745

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0129276
time: 2.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0131062
time: 2.19 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 6.16 seconds
NS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131080, upper bound: 0.0130453
NS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131080, upper bound: 0.0132919
NS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0124702, upper bound: 0.0128621
NS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0124702, upper bound: 0.0129891
NS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131121, upper bound: 0.0130621
NS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131121, upper bound: 0.0133272
NS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0132832, upper bound: 0.0130621
NS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0132832, upper bound: 0.0133272
NS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0128973
NS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0126440, upper bound: 0.0130325
NS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0128973
NS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0127369, upper bound: 0.0130325
NS_A2_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0129291
NS_A2_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0131080
NS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0129291
NS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0131080
NS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0129276
NS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0131062
NS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0129276
NS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.16
Output dim: 9, lower bound: -0.0131699, upper bound: 0.0131062

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0050441, 0.0003909, -0.0050143, 0.0005183, -0.0055624, 0.0054053
1: -0.0028513, 0.0104438, -0.0031505, 0.0106850, -0.0135363, 0.0135943
2: 0.0051091, 0.0187944, 0.0048707, 0.0191374, -0.0135424, 0.0132989
3: -0.0062963, -0.0018953, -0.0062496, -0.0018891, -0.0044023, 0.0043543
4: 0.0036019, 0.0077043, 0.0036633, 0.0077530, -0.0040331, 0.0040268
5: -0.0053856, 0.0006408, -0.0054492, 0.0005028, -0.0058884, 0.0060899
6: -0.0066107, -0.0047713, -0.0066274, -0.0047932, -0.0018174, 0.0018561
7: -0.0052340, 0.0004082, -0.0055207, 0.0005132, -0.0057471, 0.0059289
8: -0.0069803, -0.0014147, -0.0072304, -0.0014740, -0.0055063, 0.0058157
9: 0.9950126, 1.0122132, 0.9949508, 1.0125092, -0.0174966, 0.0172625

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128677, upper bound: 0.0127261
time: 1.50 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128677, upper bound: 0.0129529
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0050441, 0.0003909, -0.0050623, 0.0006684, -0.0057125, 0.0054533
1: -0.0028513, 0.0104438, -0.0036095, 0.0107399, -0.0135912, 0.0140533
2: 0.0051091, 0.0187944, 0.0045802, 0.0192696, -0.0137361, 0.0135675
3: -0.0062963, -0.0018953, -0.0063586, -0.0017561, -0.0045402, 0.0044633
4: 0.0036019, 0.0077043, 0.0035831, 0.0079551, -0.0042455, 0.0041212
5: -0.0053856, 0.0006408, -0.0057807, 0.0006074, -0.0059930, 0.0064215
6: -0.0066107, -0.0047713, -0.0066578, -0.0047618, -0.0018489, 0.0018865
7: -0.0052340, 0.0004082, -0.0055772, 0.0005985, -0.0058325, 0.0059854
8: -0.0069803, -0.0014147, -0.0073216, -0.0014500, -0.0055303, 0.0059069
9: 0.9950126, 1.0122132, 0.9947715, 1.0132531, -0.0182405, 0.0174417

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128677, upper bound: 0.0128320
time: 2.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128677, upper bound: 0.0130485
time: 2.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0050441, 0.0003909, -0.0052448, 0.0010475, -0.0060916, 0.0056358
1: -0.0028513, 0.0104438, -0.0031036, 0.0112319, -0.0140832, 0.0135474
2: 0.0051091, 0.0187944, 0.0049545, 0.0202091, -0.0141987, 0.0129110
3: -0.0062963, -0.0018953, -0.0067821, -0.0018722, -0.0043696, 0.0048868
4: 0.0036019, 0.0077043, 0.0034459, 0.0077260, -0.0039018, 0.0040846
5: -0.0053856, 0.0006408, -0.0055446, 0.0008050, -0.0061906, 0.0061854
6: -0.0066107, -0.0047713, -0.0068098, -0.0047201, -0.0018905, 0.0020385
7: -0.0052340, 0.0004082, -0.0056302, 0.0006892, -0.0059231, 0.0060384
8: -0.0069803, -0.0014147, -0.0079241, -0.0014053, -0.0055750, 0.0065093
9: 0.9950126, 1.0122132, 0.9941750, 1.0124183, -0.0174056, 0.0180383

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128736, upper bound: 0.0127407
time: 1.87 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128736, upper bound: 0.0129700
time: 2.18 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0050441, 0.0003909, -0.0052790, 0.0011314, -0.0061755, 0.0056699
1: -0.0028513, 0.0104438, -0.0035298, 0.0112762, -0.0141275, 0.0139736
2: 0.0051091, 0.0187944, 0.0046848, 0.0203137, -0.0143868, 0.0131793
3: -0.0062963, -0.0018953, -0.0068624, -0.0017443, -0.0045473, 0.0049671
4: 0.0036019, 0.0077043, 0.0033834, 0.0079169, -0.0041146, 0.0042011
5: -0.0053856, 0.0006408, -0.0058449, 0.0008793, -0.0062649, 0.0064857
6: -0.0066107, -0.0047713, -0.0068334, -0.0046908, -0.0019199, 0.0020621
7: -0.0052340, 0.0004082, -0.0056917, 0.0007699, -0.0060039, 0.0060999
8: -0.0069803, -0.0014147, -0.0079949, -0.0013893, -0.0055909, 0.0065802
9: 0.9950126, 1.0122132, 0.9940212, 1.0131137, -0.0181011, 0.0181921

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0128736, upper bound: 0.0128574
time: 2.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0128736, upper bound: 0.0130836
time: 2.39 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0050494, 0.0005016, -0.0052448, 0.0010475, -0.0060969, 0.0057464
1: -0.0032541, 0.0104475, -0.0031036, 0.0112319, -0.0144860, 0.0135511
2: 0.0048604, 0.0188066, 0.0049545, 0.0202091, -0.0144221, 0.0130063
3: -0.0063182, -0.0017625, -0.0067821, -0.0018722, -0.0044239, 0.0050196
4: 0.0035452, 0.0078751, 0.0034459, 0.0077260, -0.0040531, 0.0042847
5: -0.0056878, 0.0006980, -0.0055446, 0.0008050, -0.0064928, 0.0062426
6: -0.0066158, -0.0047351, -0.0068098, -0.0047201, -0.0018957, 0.0020748
7: -0.0053120, 0.0004784, -0.0056302, 0.0006892, -0.0060011, 0.0061085
8: -0.0069938, -0.0013965, -0.0079241, -0.0014053, -0.0055885, 0.0065276
9: 0.9949242, 1.0128475, 0.9941750, 1.0124183, -0.0174940, 0.0186726

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130401, upper bound: 0.0126660
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130401, upper bound: 0.0128314
time: 2.12 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0050494, 0.0005016, -0.0052790, 0.0011314, -0.0061808, 0.0057805
1: -0.0032541, 0.0104475, -0.0035298, 0.0112762, -0.0145303, 0.0139773
2: 0.0048604, 0.0188066, 0.0046848, 0.0203137, -0.0144493, 0.0130940
3: -0.0063182, -0.0017625, -0.0068624, -0.0017443, -0.0044384, 0.0050409
4: 0.0035452, 0.0078751, 0.0033834, 0.0079169, -0.0040236, 0.0041856
5: -0.0056878, 0.0006980, -0.0058449, 0.0008793, -0.0065671, 0.0065430
6: -0.0066158, -0.0047351, -0.0068334, -0.0046908, -0.0019250, 0.0020984
7: -0.0053120, 0.0004784, -0.0056917, 0.0007699, -0.0060819, 0.0061700
8: -0.0069938, -0.0013965, -0.0079949, -0.0013893, -0.0056044, 0.0065985
9: 0.9949242, 1.0128475, 0.9940212, 1.0131137, -0.0181895, 0.0188264

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130401, upper bound: 0.0126660
time: 1.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130401, upper bound: 0.0128314
time: 2.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0050441, 0.0003909, -0.0052023, 0.0011002, -0.0061443, 0.0055933
1: -0.0028513, 0.0104438, -0.0047793, 0.0111100, -0.0139613, 0.0152231
2: 0.0051091, 0.0187944, 0.0038354, 0.0199857, -0.0141872, 0.0141548
3: -0.0062963, -0.0018953, -0.0066934, -0.0016511, -0.0046452, 0.0047981
4: 0.0036019, 0.0077043, 0.0033505, 0.0082512, -0.0045087, 0.0042821
5: -0.0053856, 0.0006408, -0.0062480, 0.0008568, -0.0062424, 0.0068888
6: -0.0066107, -0.0047713, -0.0067765, -0.0046364, -0.0019743, 0.0020052
7: -0.0052340, 0.0004082, -0.0057246, 0.0009354, -0.0061694, 0.0061328
8: -0.0069803, -0.0014147, -0.0077935, -0.0014223, -0.0055580, 0.0063787
9: 0.9950126, 1.0122132, 0.9941667, 1.0146743, -0.0196617, 0.0180466

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0123834, upper bound: 0.0126119
time: 2.41 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0123826, upper bound: 0.0128308
time: 2.09 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0050494, 0.0005016, -0.0052023, 0.0011002, -0.0061496, 0.0057039
1: -0.0032541, 0.0104475, -0.0047793, 0.0111100, -0.0143641, 0.0152268
2: 0.0048604, 0.0188066, 0.0038354, 0.0199857, -0.0143026, 0.0141171
3: -0.0063182, -0.0017625, -0.0066934, -0.0016511, -0.0046167, 0.0048939
4: 0.0035452, 0.0078751, 0.0033505, 0.0082512, -0.0044780, 0.0043083
5: -0.0056878, 0.0006980, -0.0062480, 0.0008568, -0.0065446, 0.0069461
6: -0.0066158, -0.0047351, -0.0067765, -0.0046364, -0.0019795, 0.0020414
7: -0.0053120, 0.0004784, -0.0057246, 0.0009354, -0.0062474, 0.0062030
8: -0.0069938, -0.0013965, -0.0077935, -0.0014223, -0.0055715, 0.0063970
9: 0.9949242, 1.0128475, 0.9941667, 1.0146743, -0.0197501, 0.0186809

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 134

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124660, upper bound: 0.0125073
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0124595, upper bound: 0.0126535
time: 2.43 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.68 + 598.75 = 603.43 seconds
