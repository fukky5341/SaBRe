## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.279429504


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142)
1: (-0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979)
2: (-0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389)
3: (-0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414)
4: (-0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357)
5: (-0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473)
6: (-0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181)
7: (-0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449)
8: (-0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810)
9: (0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.11 + 3.57 = 5.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.2910724, upper bound: 0.2910724

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 199
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2852680, upper bound: 0.2834033
time: 2.37 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2867381, upper bound: 0.2867381
time: 1.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.56 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 4.56
Output dim: 9, lower bound: -0.2852680, upper bound: 0.2834033
NS_B2, status: Status.UNKNOWN, split count: 1, time: 4.56
Output dim: 9, lower bound: -0.2867381, upper bound: 0.2867381

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -0.0129652, 0.0413731, -0.0111703, 0.0279490, -0.0409142, 0.0525434
1: -0.0704899, 0.0537955, -0.0460892, 0.0438015, -0.1142914, 0.0998848
2: -0.0256158, 0.0595230, -0.0189219, 0.0536682, -0.0792840, 0.0784448
3: -0.0347380, 0.0319183, -0.0240653, 0.0215755, -0.0563135, 0.0559836
4: -0.0362096, 0.0583846, -0.0275770, 0.0265261, -0.0627357, 0.0859616
5: -0.0363896, 0.0585115, -0.0291737, 0.0434824, -0.0798720, 0.0876852
6: -0.0187536, 0.0350304, -0.0139825, 0.0266369, -0.0453905, 0.0490129
7: -0.0580633, 0.0540047, -0.0458733, 0.0353847, -0.0934481, 0.0998780
8: -0.0608374, 0.0472186, -0.0419444, 0.0333508, -0.0941882, 0.0891630
9: 0.8384492, 1.0964333, 0.8880799, 1.0745114, -0.2360622, 0.2083533

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 232
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 199
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2757960, upper bound: 0.2720876
time: 2.89 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2728784, upper bound: 0.2719351
time: 1.98 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -0.0131966, 0.0424699, -0.0126155, 0.0370694, -0.0502660, 0.0550854
1: -0.0760783, 0.0563151, -0.0665617, 0.0526308, -0.1287090, 0.1228767
2: -0.0265481, 0.0601549, -0.0239718, 0.0584493, -0.0849974, 0.0841267
3: -0.0353133, 0.0349413, -0.0304972, 0.0314492, -0.0667626, 0.0654385
4: -0.0387851, 0.0598488, -0.0363188, 0.0437496, -0.0825347, 0.0961676
5: -0.0371361, 0.0625069, -0.0340900, 0.0579129, -0.0950490, 0.0965970
6: -0.0190883, 0.0376673, -0.0171751, 0.0346812, -0.0537695, 0.0548424
7: -0.0595605, 0.0555874, -0.0552582, 0.0474112, -0.1069717, 0.1108456
8: -0.0664674, 0.0495082, -0.0594778, 0.0447232, -0.1111906, 0.1089859
9: 0.8316510, 1.1053393, 0.8488286, 1.0980957, -0.2664446, 0.2565107

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 185
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 232
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 232
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 128
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2767259, upper bound: 0.2735138
time: 2.17 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2733644, upper bound: 0.2733641
time: 2.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.62 seconds
NS_B1_A1, status: Status.VERIFIED, split count: 2, time: 6.62
Output dim: 9, lower bound: -0.2757960, upper bound: 0.2720876
NS_B1_A2, status: Status.VERIFIED, split count: 2, time: 6.62
Output dim: 9, lower bound: -0.2728784, upper bound: 0.2719351
NS_B2_A1, status: Status.VERIFIED, split count: 2, time: 6.62
Output dim: 9, lower bound: -0.2767259, upper bound: 0.2735138
NS_B2_A2, status: Status.VERIFIED, split count: 2, time: 6.62
Output dim: 9, lower bound: -0.2733644, upper bound: 0.2733641

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.68 + 18.10 = 23.78 seconds
