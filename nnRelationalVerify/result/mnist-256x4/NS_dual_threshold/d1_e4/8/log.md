## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0015192279999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0010538, 0.0006285, -0.0010538, 0.0006285, -0.0016823, 0.0016823)
1: (-0.0069846, -0.0027157, -0.0069846, -0.0027157, -0.0038504, 0.0038504)
2: (0.0306967, 0.0333452, 0.0306967, 0.0333452, -0.0026485, 0.0026485)
3: (-0.0027015, 0.0022440, -0.0027015, 0.0022440, -0.0046414, 0.0046414)
4: (-0.0059976, -0.0011409, -0.0059976, -0.0011409, -0.0040695, 0.0040695)
5: (0.0114665, 0.0131112, 0.0114665, 0.0131112, -0.0016447, 0.0016447)
6: (-0.0035554, 0.0032360, -0.0035554, 0.0032360, -0.0056837, 0.0056837)
7: (0.9759318, 0.9803236, 0.9759318, 0.9803236, -0.0039407, 0.0039407)
8: (-0.0123692, -0.0076604, -0.0123692, -0.0076604, -0.0047088, 0.0047088)
9: (0.0000605, 0.0031709, 0.0000605, 0.0031709, -0.0031104, 0.0031104)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.65 + 2.72 = 4.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0016162, upper bound: 0.0016162

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015303, upper bound: 0.0015812
time: 1.81 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015812, upper bound: 0.0015812
time: 1.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.80 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.80
Output dim: 2, lower bound: -0.0015303, upper bound: 0.0015812
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.80
Output dim: 2, lower bound: -0.0015812, upper bound: 0.0015812

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0008951, 0.0005296, -0.0009888, 0.0006195, -0.0012381, 0.0015184
1: -0.0065819, -0.0029667, -0.0068197, -0.0027386, -0.0031418, 0.0033231
2: 0.0309466, 0.0331895, 0.0307990, 0.0333310, -0.0019492, 0.0023143
3: -0.0024107, 0.0017774, -0.0026749, 0.0020529, -0.0040052, 0.0036396
4: -0.0055879, -0.0019106, -0.0058298, -0.0012397, -0.0035666, 0.0031276
5: 0.0116216, 0.0130145, 0.0115300, 0.0131024, -0.0012104, 0.0014845
6: -0.0026714, 0.0026439, -0.0034461, 0.0029935, -0.0045207, 0.0049876
7: 0.9761900, 0.9799094, 0.9759553, 0.9801540, -0.0034013, 0.0032322
8: -0.0120923, -0.0081046, -0.0123439, -0.0078423, -0.0042500, 0.0034655
9: 0.0003539, 0.0029881, 0.0001807, 0.0031542, -0.0022891, 0.0028074

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014490, upper bound: 0.0015123
time: 2.34 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0014689, upper bound: 0.0015194
time: 1.78 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0010127, 0.0006190, -0.0010389, 0.0006253, -0.0016379, 0.0016579
1: -0.0068804, -0.0027398, -0.0069470, -0.0027239, -0.0033099, 0.0037837
2: 0.0307614, 0.0333303, 0.0307200, 0.0333401, -0.0023445, 0.0025988
3: -0.0026735, 0.0021231, -0.0026920, 0.0022004, -0.0045442, 0.0040211
4: -0.0058915, -0.0012447, -0.0059593, -0.0011761, -0.0034817, 0.0039577
5: 0.0115066, 0.0131019, 0.0114810, 0.0131080, -0.0016014, 0.0016210
6: -0.0034405, 0.0030827, -0.0035164, 0.0031807, -0.0055544, 0.0048512
7: 0.9759565, 0.9802164, 0.9759402, 0.9802849, -0.0038742, 0.0033844
8: -0.0123426, -0.0077754, -0.0123601, -0.0077019, -0.0046407, 0.0045847
9: 0.0001365, 0.0031534, 0.0000879, 0.0031650, -0.0030285, 0.0030654

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014996, upper bound: 0.0015123
time: 1.85 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0015195, upper bound: 0.0015194
time: 1.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.34 seconds
NS_A1_A1, status: Status.VERIFIED, split count: 2, time: 5.34
Output dim: 2, lower bound: -0.0014490, upper bound: 0.0015123
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 5.34
Output dim: 2, lower bound: -0.0014689, upper bound: 0.0015194
NS_A2_A1, status: Status.VERIFIED, split count: 2, time: 5.34
Output dim: 2, lower bound: -0.0014996, upper bound: 0.0015123
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.34
Output dim: 2, lower bound: -0.0015195, upper bound: 0.0015194

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.0008942, 0.0004551, -0.0009885, 0.0005951, -0.0012160, 0.0014424
1: -0.0065798, -0.0031558, -0.0068190, -0.0028005, -0.0030857, 0.0029284
2: 0.0309479, 0.0330721, 0.0307994, 0.0332926, -0.0019144, 0.0019215
3: -0.0021916, 0.0017749, -0.0026032, 0.0020521, -0.0034634, 0.0035746
4: -0.0055857, -0.0021030, -0.0058291, -0.0015065, -0.0033350, 0.0028525
5: 0.0116225, 0.0129416, 0.0115303, 0.0130785, -0.0011888, 0.0013887
6: -0.0023932, 0.0026407, -0.0031510, 0.0029925, -0.0041230, 0.0047325
7: 0.9763846, 0.9799071, 0.9760190, 0.9801533, -0.0030042, 0.0031745
8: -0.0118837, -0.0081070, -0.0122756, -0.0078430, -0.0039447, 0.0034036
9: 0.0003555, 0.0028502, 0.0001812, 0.0031091, -0.0022483, 0.0025126

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0013974, upper bound: 0.0014589
time: 2.00 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014084, upper bound: 0.0014577
time: 1.99 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0010118, 0.0005423, -0.0010387, 0.0006009, -0.0011746, 0.0015810
1: -0.0068781, -0.0029343, -0.0069464, -0.0027858, -0.0029808, 0.0034011
2: 0.0307628, 0.0332096, 0.0307205, 0.0333017, -0.0018493, 0.0022336
3: -0.0024482, 0.0021206, -0.0026202, 0.0021996, -0.0040235, 0.0034531
4: -0.0058892, -0.0018777, -0.0059586, -0.0014433, -0.0032575, 0.0033190
5: 0.0115075, 0.0130270, 0.0114812, 0.0130842, -0.0011484, 0.0015458
6: -0.0027189, 0.0030794, -0.0032209, 0.0031797, -0.0047973, 0.0046066
7: 0.9761568, 0.9802141, 0.9760039, 0.9802843, -0.0034895, 0.0030666
8: -0.0121280, -0.0077779, -0.0122918, -0.0077026, -0.0044254, 0.0032879
9: 0.0001381, 0.0030116, 0.0000884, 0.0031198, -0.0021718, 0.0029126

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014482, upper bound: 0.0014589
time: 1.82 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0014578, upper bound: 0.0014575
time: 1.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.13 seconds
NS_A1_A2_A1, status: Status.VERIFIED, split count: 3, time: 5.13
Output dim: 2, lower bound: -0.0013974, upper bound: 0.0014589
NS_A1_A2_A2, status: Status.VERIFIED, split count: 3, time: 5.13
Output dim: 2, lower bound: -0.0014084, upper bound: 0.0014577
NS_A2_A2_A1, status: Status.VERIFIED, split count: 3, time: 5.13
Output dim: 2, lower bound: -0.0014482, upper bound: 0.0014589
NS_A2_A2_A2, status: Status.VERIFIED, split count: 3, time: 5.13
Output dim: 2, lower bound: -0.0014578, upper bound: 0.0014575

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.37 + 25.62 = 29.99 seconds
