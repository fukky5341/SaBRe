## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.075013435


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0166849, 0.0250697, -0.0166849, 0.0250697, -0.0417546, 0.0417546)
1: (-0.0139894, 0.0534196, -0.0139894, 0.0534196, -0.0674090, 0.0674090)
2: (0.0022807, 0.0438471, 0.0022807, 0.0438471, -0.0415664, 0.0415664)
3: (-0.0161741, 0.0294320, -0.0161741, 0.0294320, -0.0456061, 0.0456061)
4: (-0.0388691, 0.0247383, -0.0388691, 0.0247383, -0.0636074, 0.0636074)
5: (-0.0199401, 0.0418346, -0.0199401, 0.0418346, -0.0617746, 0.0617746)
6: (-0.0128523, 0.0287888, -0.0128523, 0.0287888, -0.0416411, 0.0416411)
7: (-0.0376090, 0.0265432, -0.0376090, 0.0265432, -0.0641521, 0.0641521)
8: (-0.0134290, 0.0488630, -0.0134290, 0.0488630, -0.0622921, 0.0622921)
9: (0.8556051, 1.0088843, 0.8556051, 1.0088843, -0.1532792, 0.1532792)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 3.10 = 4.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0882511, upper bound: 0.0882511

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0835640, upper bound: 0.0858858
time: 1.97 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0858858, upper bound: 0.0858858
time: 2.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.09 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.09
Output dim: 9, lower bound: -0.0835640, upper bound: 0.0858858
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.09
Output dim: 9, lower bound: -0.0858858, upper bound: 0.0858858

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0156531, 0.0210691, -0.0163677, 0.0240460, -0.0396990, 0.0374368
1: -0.0082205, 0.0496574, -0.0128529, 0.0522128, -0.0604333, 0.0625103
2: 0.0062132, 0.0412265, 0.0038805, 0.0431521, -0.0369388, 0.0373460
3: -0.0128219, 0.0271098, -0.0156291, 0.0286838, -0.0415057, 0.0427389
4: -0.0343439, 0.0149274, -0.0374880, 0.0221238, -0.0564677, 0.0524154
5: -0.0137598, 0.0390376, -0.0186054, 0.0409384, -0.0546983, 0.0576430
6: -0.0101697, 0.0250146, -0.0124672, 0.0276359, -0.0378056, 0.0374819
7: -0.0345220, 0.0207985, -0.0366429, 0.0251738, -0.0596958, 0.0574414
8: -0.0105467, 0.0421653, -0.0130583, 0.0468509, -0.0573976, 0.0541922
9: 0.8679127, 0.9999187, 0.8594437, 1.0080948, -0.1325663, 0.1404750

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0835640, upper bound: 0.0835640
time: 1.74 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0835640, upper bound: 0.0858858
time: 1.51 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0159680, 0.0231623, -0.0164571, 0.0244779, -0.0404459, 0.0396193
1: -0.0118651, 0.0514170, -0.0133277, 0.0527647, -0.0646299, 0.0647447
2: 0.0050116, 0.0425785, 0.0033087, 0.0434461, -0.0384345, 0.0392699
3: -0.0151811, 0.0281883, -0.0158656, 0.0290181, -0.0441992, 0.0440538
4: -0.0364268, 0.0197677, -0.0381023, 0.0231755, -0.0596022, 0.0578700
5: -0.0174391, 0.0403556, -0.0191603, 0.0413453, -0.0587843, 0.0595159
6: -0.0121520, 0.0268643, -0.0126340, 0.0281719, -0.0403239, 0.0394983
7: -0.0360575, 0.0239444, -0.0371033, 0.0257391, -0.0617965, 0.0610477
8: -0.0127840, 0.0453564, -0.0132304, 0.0477483, -0.0605323, 0.0585868
9: 0.8617136, 1.0074937, 0.8575721, 1.0084556, -0.1467420, 0.1499217

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0858858, upper bound: 0.0835640
time: 2.17 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0858858, upper bound: 0.0858858
time: 1.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.32 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.32
Output dim: 9, lower bound: -0.0835640, upper bound: 0.0835640
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.32
Output dim: 9, lower bound: -0.0835640, upper bound: 0.0858858
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.32
Output dim: 9, lower bound: -0.0858858, upper bound: 0.0835640
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.32
Output dim: 9, lower bound: -0.0858858, upper bound: 0.0858858

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0156531, 0.0210691, -0.0156531, 0.0210691, -0.0367222, 0.0367222
1: -0.0082205, 0.0496574, -0.0082205, 0.0496574, -0.0578779, 0.0578779
2: 0.0062132, 0.0412265, 0.0062132, 0.0412265, -0.0350132, 0.0350132
3: -0.0128219, 0.0271098, -0.0128219, 0.0271098, -0.0399317, 0.0399317
4: -0.0343439, 0.0149274, -0.0343439, 0.0149274, -0.0492712, 0.0492712
5: -0.0137598, 0.0390376, -0.0137598, 0.0390376, -0.0527975, 0.0527975
6: -0.0101697, 0.0250146, -0.0101697, 0.0250146, -0.0351844, 0.0351844
7: -0.0345220, 0.0207985, -0.0345220, 0.0207985, -0.0553205, 0.0553205
8: -0.0105467, 0.0421653, -0.0105467, 0.0421653, -0.0518592, 0.0518592
9: 0.8679127, 0.9999187, 0.8679127, 0.9999187, -0.1253071, 0.1253070

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0798641
time: 1.63 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0808305, upper bound: 0.0808439
time: 1.13 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0156531, 0.0210691, -0.0159680, 0.0231623, -0.0388153, 0.0370371
1: -0.0082205, 0.0496574, -0.0118651, 0.0514170, -0.0596376, 0.0615225
2: 0.0062132, 0.0412265, 0.0050116, 0.0425785, -0.0363653, 0.0362149
3: -0.0128219, 0.0271098, -0.0151811, 0.0281883, -0.0410101, 0.0422909
4: -0.0343439, 0.0149274, -0.0364268, 0.0197677, -0.0541115, 0.0513541
5: -0.0137598, 0.0390376, -0.0174391, 0.0403556, -0.0541154, 0.0564767
6: -0.0101697, 0.0250146, -0.0121520, 0.0268643, -0.0370341, 0.0371603
7: -0.0345220, 0.0207985, -0.0360575, 0.0239444, -0.0584664, 0.0568559
8: -0.0105467, 0.0421653, -0.0127840, 0.0453564, -0.0559031, 0.0539001
9: 0.8679127, 0.9999187, 0.8617136, 1.0074937, -0.1318834, 0.1382051

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0818174
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0808305, upper bound: 0.0829000
time: 1.14 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0159680, 0.0231623, -0.0156531, 0.0210691, -0.0370371, 0.0388153
1: -0.0118651, 0.0514170, -0.0082205, 0.0496574, -0.0615225, 0.0596376
2: 0.0050116, 0.0425785, 0.0062132, 0.0412265, -0.0362149, 0.0363653
3: -0.0151811, 0.0281883, -0.0128219, 0.0271098, -0.0422909, 0.0410101
4: -0.0364268, 0.0197677, -0.0343439, 0.0149274, -0.0513541, 0.0541115
5: -0.0174391, 0.0403556, -0.0137598, 0.0390376, -0.0564767, 0.0541154
6: -0.0121520, 0.0268643, -0.0101697, 0.0250146, -0.0371603, 0.0370341
7: -0.0360575, 0.0239444, -0.0345220, 0.0207985, -0.0568559, 0.0584664
8: -0.0127840, 0.0453564, -0.0105467, 0.0421653, -0.0539001, 0.0559031
9: 0.8617136, 1.0074937, 0.8679127, 0.9999187, -0.1382051, 0.1318834

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
time: 2.11 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0829000, upper bound: 0.0808305
time: 1.15 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0159680, 0.0231623, -0.0159680, 0.0231623, -0.0391302, 0.0391302
1: -0.0118651, 0.0514170, -0.0118651, 0.0514170, -0.0632822, 0.0632822
2: 0.0050116, 0.0425785, 0.0050116, 0.0425785, -0.0375669, 0.0375669
3: -0.0151811, 0.0281883, -0.0151811, 0.0281883, -0.0433694, 0.0433694
4: -0.0364268, 0.0197677, -0.0364268, 0.0197677, -0.0561944, 0.0561944
5: -0.0174391, 0.0403556, -0.0174391, 0.0403556, -0.0577946, 0.0577946
6: -0.0121520, 0.0268643, -0.0121520, 0.0268643, -0.0390164, 0.0390164
7: -0.0360575, 0.0239444, -0.0360575, 0.0239444, -0.0600019, 0.0600019
8: -0.0127840, 0.0453564, -0.0127840, 0.0453564, -0.0581404, 0.0581404
9: 0.8617136, 1.0074937, 0.8617136, 1.0074937, -0.1457801, 0.1457801

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
time: 2.07 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0829000, upper bound: 0.0808305
time: 1.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.48 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.48
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0798641
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.48
Output dim: 9, lower bound: -0.0808305, upper bound: 0.0808439
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.48
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0818174
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.48
Output dim: 9, lower bound: -0.0808305, upper bound: 0.0829000
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.48
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.48
Output dim: 9, lower bound: -0.0829000, upper bound: 0.0808305
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.48
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.48
Output dim: 9, lower bound: -0.0829000, upper bound: 0.0808305

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0158237, 0.0127893, -0.0153956, 0.0182535, -0.0340772, 0.0281848
1: -0.0054148, 0.0461315, -0.0052599, 0.0484950, -0.0539098, 0.0513913
2: 0.0053458, 0.0376629, 0.0068169, 0.0400608, -0.0314882, 0.0294597
3: -0.0125702, 0.0245662, -0.0116096, 0.0263214, -0.0385880, 0.0361758
4: -0.0247824, 0.0048562, -0.0313829, 0.0076923, -0.0316874, 0.0327754
5: -0.0092804, 0.0363421, -0.0100342, 0.0381541, -0.0474345, 0.0463763
6: -0.0104036, 0.0199739, -0.0093593, 0.0228200, -0.0321791, 0.0293332
7: -0.0316892, 0.0117341, -0.0333524, 0.0171090, -0.0487982, 0.0450865
8: -0.0124057, 0.0326010, -0.0100123, 0.0379489, -0.0488121, 0.0425092
9: 0.8859093, 1.0092524, 0.8740575, 0.9988016, -0.1042092, 0.1223920

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0759387, upper bound: 0.0759387
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0759387, upper bound: 0.0798641
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0155506, 0.0200678, -0.0156531, 0.0210691, -0.0366197, 0.0357209
1: -0.0071896, 0.0491037, -0.0082205, 0.0496574, -0.0568470, 0.0573243
2: 0.0065629, 0.0408218, 0.0062132, 0.0412265, -0.0346636, 0.0346085
3: -0.0124121, 0.0267453, -0.0128219, 0.0271098, -0.0395219, 0.0395672
4: -0.0333054, 0.0123178, -0.0343439, 0.0149274, -0.0482328, 0.0466616
5: -0.0124739, 0.0386236, -0.0137598, 0.0390376, -0.0515115, 0.0523835
6: -0.0099007, 0.0241867, -0.0101697, 0.0250146, -0.0348822, 0.0336319
7: -0.0340049, 0.0195085, -0.0345220, 0.0207985, -0.0548034, 0.0540305
8: -0.0103662, 0.0406451, -0.0105467, 0.0421653, -0.0515112, 0.0482888
9: 0.8701790, 0.9995501, 0.8679127, 0.9999187, -0.1118884, 0.1249174

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798641, upper bound: 0.0759387
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798641, upper bound: 0.0808439
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0158237, 0.0127893, -0.0156892, 0.0202269, -0.0360506, 0.0284785
1: -0.0054148, 0.0461315, -0.0089824, 0.0502319, -0.0556467, 0.0551139
2: 0.0053458, 0.0376629, 0.0062521, 0.0412116, -0.0358657, 0.0314051
3: -0.0125702, 0.0245662, -0.0139937, 0.0273950, -0.0399652, 0.0385599
4: -0.0247824, 0.0048562, -0.0335060, 0.0123646, -0.0367722, 0.0349841
5: -0.0092804, 0.0363421, -0.0138692, 0.0394606, -0.0487410, 0.0502112
6: -0.0104036, 0.0199739, -0.0113581, 0.0246142, -0.0350178, 0.0313320
7: -0.0316892, 0.0117341, -0.0347635, 0.0203312, -0.0520204, 0.0464975
8: -0.0124057, 0.0326010, -0.0122177, 0.0410441, -0.0534498, 0.0446064
9: 0.8859093, 1.0092524, 0.8678197, 1.0062842, -0.1106654, 0.1414328

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0783961
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0818174
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0155506, 0.0200678, -0.0159680, 0.0231623, -0.0387129, 0.0360358
1: -0.0071896, 0.0491037, -0.0118651, 0.0514170, -0.0586066, 0.0609689
2: 0.0065629, 0.0408218, 0.0050116, 0.0425785, -0.0360157, 0.0358102
3: -0.0124121, 0.0267453, -0.0151811, 0.0281883, -0.0406004, 0.0419264
4: -0.0333054, 0.0123178, -0.0364268, 0.0197677, -0.0530731, 0.0487445
5: -0.0124739, 0.0386236, -0.0174391, 0.0403556, -0.0528294, 0.0560627
6: -0.0099007, 0.0241867, -0.0121520, 0.0268643, -0.0367650, 0.0355213
7: -0.0340049, 0.0195085, -0.0360575, 0.0239444, -0.0579493, 0.0555660
8: -0.0103662, 0.0406451, -0.0127840, 0.0453564, -0.0557227, 0.0507418
9: 0.8701790, 0.9995501, 0.8617136, 1.0074937, -0.1208160, 0.1378365

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798544, upper bound: 0.0783961
time: 2.28 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798544, upper bound: 0.0829000
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0160097, 0.0154769, -0.0153956, 0.0182535, -0.0342632, 0.0308724
1: -0.0083188, 0.0477993, -0.0052599, 0.0484950, -0.0568138, 0.0530592
2: 0.0047449, 0.0397154, 0.0068169, 0.0400608, -0.0322294, 0.0328985
3: -0.0148234, 0.0256052, -0.0116096, 0.0263214, -0.0408985, 0.0372148
4: -0.0270774, 0.0069391, -0.0313829, 0.0076923, -0.0339065, 0.0363754
5: -0.0119602, 0.0376129, -0.0100342, 0.0381541, -0.0501143, 0.0476471
6: -0.0123997, 0.0210322, -0.0093593, 0.0228200, -0.0341805, 0.0303915
7: -0.0329190, 0.0151685, -0.0333524, 0.0171090, -0.0500280, 0.0485209
8: -0.0145032, 0.0343645, -0.0100123, 0.0379489, -0.0509295, 0.0443768
9: 0.8804663, 1.0167246, 0.8740575, 0.9988016, -0.1183353, 0.1308804

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0757888
time: 2.04 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
time: 1.73 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0158550, 0.0220735, -0.0156531, 0.0210691, -0.0369241, 0.0377266
1: -0.0108087, 0.0508285, -0.0082205, 0.0496574, -0.0604661, 0.0590490
2: 0.0057412, 0.0420804, 0.0062132, 0.0412265, -0.0354853, 0.0358672
3: -0.0147648, 0.0277965, -0.0128219, 0.0271098, -0.0418746, 0.0406184
4: -0.0352937, 0.0169752, -0.0343439, 0.0149274, -0.0502211, 0.0513190
5: -0.0161317, 0.0399140, -0.0137598, 0.0390376, -0.0551693, 0.0536738
6: -0.0118801, 0.0259476, -0.0101697, 0.0250146, -0.0365969, 0.0361173
7: -0.0354761, 0.0226024, -0.0345220, 0.0207985, -0.0562746, 0.0571244
8: -0.0125955, 0.0436554, -0.0105467, 0.0421653, -0.0535138, 0.0542021
9: 0.8642685, 1.0071003, 0.8679127, 0.9999187, -0.1356502, 0.1314623

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818173, upper bound: 0.0757888
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818173, upper bound: 0.0808305
time: 1.34 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0160097, 0.0154769, -0.0156892, 0.0202269, -0.0362366, 0.0311661
1: -0.0083188, 0.0477993, -0.0089824, 0.0502319, -0.0585507, 0.0567818
2: 0.0047449, 0.0397154, 0.0062521, 0.0412116, -0.0364667, 0.0334634
3: -0.0148234, 0.0256052, -0.0139937, 0.0273950, -0.0422184, 0.0395989
4: -0.0270774, 0.0069391, -0.0335060, 0.0123646, -0.0383315, 0.0371412
5: -0.0119602, 0.0376129, -0.0138692, 0.0394606, -0.0514208, 0.0514820
6: -0.0123997, 0.0210322, -0.0113581, 0.0246142, -0.0370139, 0.0323903
7: -0.0329190, 0.0151685, -0.0347635, 0.0203312, -0.0532502, 0.0499320
8: -0.0145032, 0.0343645, -0.0122177, 0.0410441, -0.0555474, 0.0465821
9: 0.8804663, 1.0167246, 0.8678197, 1.0062842, -0.1258179, 0.1489049

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0757973
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
time: 1.62 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0158550, 0.0220735, -0.0159680, 0.0231623, -0.0390173, 0.0380415
1: -0.0108087, 0.0508285, -0.0118651, 0.0514170, -0.0622257, 0.0626936
2: 0.0057412, 0.0420804, 0.0050116, 0.0425785, -0.0368374, 0.0370688
3: -0.0147648, 0.0277965, -0.0151811, 0.0281883, -0.0429531, 0.0429776
4: -0.0352937, 0.0169752, -0.0364268, 0.0197677, -0.0550614, 0.0534019
5: -0.0161317, 0.0399140, -0.0174391, 0.0403556, -0.0564872, 0.0573530
6: -0.0118801, 0.0259476, -0.0121520, 0.0268643, -0.0387444, 0.0380996
7: -0.0354761, 0.0226024, -0.0360575, 0.0239444, -0.0594205, 0.0586598
8: -0.0125955, 0.0436554, -0.0127840, 0.0453564, -0.0579519, 0.0564394
9: 0.8642685, 1.0071003, 0.8617136, 1.0074937, -0.1432252, 0.1453868

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818174, upper bound: 0.0757973
time: 1.19 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818174, upper bound: 0.0808305
time: 1.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.81 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0759387, upper bound: 0.0759387
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0759387, upper bound: 0.0798641
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0798641, upper bound: 0.0759387
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0798641, upper bound: 0.0808439
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0783961
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0818174
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0798544, upper bound: 0.0783961
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0798544, upper bound: 0.0829000
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0757888
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0818173, upper bound: 0.0757888
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0818173, upper bound: 0.0808305
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0757973
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0818174, upper bound: 0.0757973
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.81
Output dim: 9, lower bound: -0.0818174, upper bound: 0.0808305

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0158237, 0.0127893, -0.0158237, 0.0127893, -0.0286130, 0.0286130
1: -0.0054148, 0.0461315, -0.0054148, 0.0461315, -0.0515462, 0.0515462
2: 0.0053458, 0.0376629, 0.0053458, 0.0376629, -0.0288620, 0.0288620
3: -0.0125702, 0.0245662, -0.0125702, 0.0245662, -0.0366595, 0.0366595
4: -0.0247824, 0.0048562, -0.0247824, 0.0048562, -0.0263394, 0.0263394
5: -0.0092804, 0.0363421, -0.0092804, 0.0363421, -0.0456225, 0.0456225
6: -0.0104036, 0.0199739, -0.0104036, 0.0199739, -0.0295670, 0.0295670
7: -0.0316892, 0.0117341, -0.0316892, 0.0117341, -0.0434233, 0.0434233
8: -0.0124057, 0.0326010, -0.0124057, 0.0326010, -0.0440003, 0.0440003
9: 0.8859093, 1.0092524, 0.8859093, 1.0092524, -0.1094184, 0.1094184

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0725829
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0751586
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0158237, 0.0127893, -0.0155506, 0.0200678, -0.0358915, 0.0283399
1: -0.0054148, 0.0461315, -0.0071896, 0.0491037, -0.0545185, 0.0533211
2: 0.0053458, 0.0376629, 0.0065629, 0.0408218, -0.0325299, 0.0311000
3: -0.0125702, 0.0245662, -0.0124121, 0.0267453, -0.0393155, 0.0369783
4: -0.0247824, 0.0048562, -0.0333054, 0.0123178, -0.0371002, 0.0350289
5: -0.0092804, 0.0363421, -0.0124739, 0.0386236, -0.0479040, 0.0488160
6: -0.0104036, 0.0199739, -0.0099007, 0.0241867, -0.0337169, 0.0298746
7: -0.0316892, 0.0117341, -0.0340049, 0.0195085, -0.0511977, 0.0457390
8: -0.0124057, 0.0326010, -0.0103662, 0.0406451, -0.0515794, 0.0427393
9: 0.8859093, 1.0092524, 0.8701790, 0.9995501, -0.1048799, 0.1280425

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0776223
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0789311
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0155506, 0.0200678, -0.0158237, 0.0127893, -0.0283399, 0.0358915
1: -0.0071896, 0.0491037, -0.0054148, 0.0461315, -0.0533211, 0.0545185
2: 0.0065629, 0.0408218, 0.0053458, 0.0376629, -0.0311000, 0.0325299
3: -0.0124121, 0.0267453, -0.0125702, 0.0245662, -0.0369783, 0.0393155
4: -0.0333054, 0.0123178, -0.0247824, 0.0048562, -0.0350289, 0.0371002
5: -0.0124739, 0.0386236, -0.0092804, 0.0363421, -0.0488160, 0.0479040
6: -0.0099007, 0.0241867, -0.0104036, 0.0199739, -0.0298746, 0.0337169
7: -0.0340049, 0.0195085, -0.0316892, 0.0117341, -0.0457390, 0.0511977
8: -0.0103662, 0.0406451, -0.0124057, 0.0326010, -0.0427393, 0.0515794
9: 0.8701790, 0.9995501, 0.8859093, 1.0092524, -0.1280425, 0.1048799

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787060, upper bound: 0.0725829
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789311, upper bound: 0.0751586
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0155506, 0.0200678, -0.0155506, 0.0200678, -0.0356184, 0.0356184
1: -0.0071896, 0.0491037, -0.0071896, 0.0491037, -0.0562933, 0.0562933
2: 0.0065629, 0.0408218, 0.0065629, 0.0408218, -0.0342589, 0.0342589
3: -0.0124121, 0.0267453, -0.0124121, 0.0267453, -0.0391574, 0.0391574
4: -0.0333054, 0.0123178, -0.0333054, 0.0123178, -0.0451794, 0.0451794
5: -0.0124739, 0.0386236, -0.0124739, 0.0386236, -0.0510975, 0.0510975
6: -0.0099007, 0.0241867, -0.0099007, 0.0241867, -0.0331370, 0.0331370
7: -0.0340049, 0.0195085, -0.0340049, 0.0195085, -0.0535134, 0.0535134
8: -0.0103662, 0.0406451, -0.0103662, 0.0406451, -0.0479888, 0.0479889
9: 0.8701790, 0.9995501, 0.8701790, 0.9995501, -0.1114374, 0.1114373

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787061, upper bound: 0.0761589
time: 2.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789311, upper bound: 0.0775904
time: 1.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0158237, 0.0127893, -0.0160097, 0.0154769, -0.0313006, 0.0287990
1: -0.0054148, 0.0461315, -0.0083188, 0.0477993, -0.0532141, 0.0544503
2: 0.0053458, 0.0376629, 0.0047449, 0.0397154, -0.0343696, 0.0296031
3: -0.0125702, 0.0245662, -0.0148234, 0.0256052, -0.0381754, 0.0389700
4: -0.0247824, 0.0048562, -0.0270774, 0.0069391, -0.0286148, 0.0286086
5: -0.0092804, 0.0363421, -0.0119602, 0.0376129, -0.0468933, 0.0483023
6: -0.0104036, 0.0199739, -0.0123997, 0.0210322, -0.0314359, 0.0315797
7: -0.0316892, 0.0117341, -0.0329190, 0.0151685, -0.0468577, 0.0446531
8: -0.0124057, 0.0326010, -0.0145032, 0.0343645, -0.0467701, 0.0462320
9: 0.8859093, 1.0092524, 0.8804663, 1.0167246, -0.1179068, 0.1287861

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0749920
time: 2.73 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0776603
time: 1.32 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0158237, 0.0127893, -0.0158550, 0.0220735, -0.0378972, 0.0286443
1: -0.0054148, 0.0461315, -0.0108087, 0.0508285, -0.0562433, 0.0569401
2: 0.0053458, 0.0376629, 0.0057412, 0.0420804, -0.0367346, 0.0319217
3: -0.0125702, 0.0245662, -0.0147648, 0.0277965, -0.0403667, 0.0393310
4: -0.0247824, 0.0048562, -0.0352937, 0.0169752, -0.0417576, 0.0371000
5: -0.0092804, 0.0363421, -0.0161317, 0.0399140, -0.0491944, 0.0524737
6: -0.0104036, 0.0199739, -0.0118801, 0.0259476, -0.0363512, 0.0318540
7: -0.0316892, 0.0117341, -0.0354761, 0.0226024, -0.0542916, 0.0472101
8: -0.0124057, 0.0326010, -0.0125955, 0.0436554, -0.0560611, 0.0448697
9: 0.8859093, 1.0092524, 0.8642685, 1.0071003, -0.1114248, 0.1449839

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0792818
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0809698
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0155506, 0.0200678, -0.0160097, 0.0154769, -0.0310275, 0.0360776
1: -0.0071896, 0.0491037, -0.0083188, 0.0477993, -0.0549889, 0.0574225
2: 0.0065629, 0.0408218, 0.0047449, 0.0397154, -0.0331526, 0.0332711
3: -0.0124121, 0.0267453, -0.0148234, 0.0256052, -0.0380173, 0.0415687
4: -0.0333054, 0.0123178, -0.0270774, 0.0069391, -0.0392160, 0.0393952
5: -0.0124739, 0.0386236, -0.0119602, 0.0376129, -0.0500867, 0.0505838
6: -0.0099007, 0.0241867, -0.0123997, 0.0210322, -0.0309329, 0.0357188
7: -0.0340049, 0.0195085, -0.0329190, 0.0151685, -0.0491734, 0.0524275
8: -0.0103662, 0.0406451, -0.0145032, 0.0343645, -0.0447307, 0.0536883
9: 0.8701790, 0.9995501, 0.8804663, 1.0167246, -0.1365309, 0.1190838

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786935, upper bound: 0.0749920
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789210, upper bound: 0.0776603
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0155506, 0.0200678, -0.0158550, 0.0220735, -0.0376242, 0.0359228
1: -0.0071896, 0.0491037, -0.0108087, 0.0508285, -0.0580181, 0.0599124
2: 0.0065629, 0.0408218, 0.0057412, 0.0420804, -0.0355176, 0.0350806
3: -0.0124121, 0.0267453, -0.0147648, 0.0277965, -0.0402086, 0.0415101
4: -0.0333054, 0.0123178, -0.0352937, 0.0169752, -0.0502806, 0.0473469
5: -0.0124739, 0.0386236, -0.0161317, 0.0399140, -0.0523878, 0.0547553
6: -0.0099007, 0.0241867, -0.0118801, 0.0259476, -0.0358483, 0.0350296
7: -0.0340049, 0.0195085, -0.0354761, 0.0226024, -0.0566073, 0.0549846
8: -0.0103662, 0.0406451, -0.0125955, 0.0436554, -0.0540216, 0.0504169
9: 0.8701790, 0.9995501, 0.8642685, 1.0071003, -0.1203491, 0.1352816

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786935, upper bound: 0.0782554
time: 1.54 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789210, upper bound: 0.0799577
time: 2.38 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0160097, 0.0154769, -0.0158237, 0.0127893, -0.0287990, 0.0313006
1: -0.0083188, 0.0477993, -0.0054148, 0.0461315, -0.0544503, 0.0532141
2: 0.0047449, 0.0397154, 0.0053458, 0.0376629, -0.0296031, 0.0343696
3: -0.0148234, 0.0256052, -0.0125702, 0.0245662, -0.0389700, 0.0381754
4: -0.0270774, 0.0069391, -0.0247824, 0.0048562, -0.0286086, 0.0286148
5: -0.0119602, 0.0376129, -0.0092804, 0.0363421, -0.0483023, 0.0468933
6: -0.0123997, 0.0210322, -0.0104036, 0.0199739, -0.0315797, 0.0314359
7: -0.0329190, 0.0151685, -0.0316892, 0.0117341, -0.0446531, 0.0468577
8: -0.0145032, 0.0343645, -0.0124057, 0.0326010, -0.0462320, 0.0467701
9: 0.8804663, 1.0167246, 0.8859093, 1.0092524, -0.1287861, 0.1179068

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0725649
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0749830
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0160097, 0.0154769, -0.0155506, 0.0200678, -0.0360776, 0.0310275
1: -0.0083188, 0.0477993, -0.0071896, 0.0491037, -0.0574225, 0.0549889
2: 0.0047449, 0.0397154, 0.0065629, 0.0408218, -0.0332711, 0.0331526
3: -0.0148234, 0.0256052, -0.0124121, 0.0267453, -0.0415687, 0.0380173
4: -0.0270774, 0.0069391, -0.0333054, 0.0123178, -0.0393952, 0.0392160
5: -0.0119602, 0.0376129, -0.0124739, 0.0386236, -0.0505838, 0.0500867
6: -0.0123997, 0.0210322, -0.0099007, 0.0241867, -0.0357188, 0.0309329
7: -0.0329190, 0.0151685, -0.0340049, 0.0195085, -0.0524275, 0.0491734
8: -0.0145032, 0.0343645, -0.0103662, 0.0406451, -0.0536883, 0.0447307
9: 0.8804663, 1.0167246, 0.8701790, 0.9995501, -0.1190838, 0.1365309

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0776223
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0789210
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0158550, 0.0220735, -0.0158237, 0.0127893, -0.0286443, 0.0378972
1: -0.0108087, 0.0508285, -0.0054148, 0.0461315, -0.0569401, 0.0562433
2: 0.0057412, 0.0420804, 0.0053458, 0.0376629, -0.0319217, 0.0367346
3: -0.0147648, 0.0277965, -0.0125702, 0.0245662, -0.0393310, 0.0403667
4: -0.0352937, 0.0169752, -0.0247824, 0.0048562, -0.0371000, 0.0417576
5: -0.0161317, 0.0399140, -0.0092804, 0.0363421, -0.0524737, 0.0491944
6: -0.0118801, 0.0259476, -0.0104036, 0.0199739, -0.0318540, 0.0363512
7: -0.0354761, 0.0226024, -0.0316892, 0.0117341, -0.0472101, 0.0542916
8: -0.0125955, 0.0436554, -0.0124057, 0.0326010, -0.0448698, 0.0560611
9: 0.8642685, 1.0071003, 0.8859093, 1.0092524, -0.1449839, 0.1114248

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0725649
time: 2.30 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0749830
time: 1.13 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0158550, 0.0220735, -0.0155506, 0.0200678, -0.0359228, 0.0376242
1: -0.0108087, 0.0508285, -0.0071896, 0.0491037, -0.0599124, 0.0580181
2: 0.0057412, 0.0420804, 0.0065629, 0.0408218, -0.0350806, 0.0355176
3: -0.0147648, 0.0277965, -0.0124121, 0.0267453, -0.0415101, 0.0402086
4: -0.0352937, 0.0169752, -0.0333054, 0.0123178, -0.0473469, 0.0502806
5: -0.0161317, 0.0399140, -0.0124739, 0.0386236, -0.0547553, 0.0523878
6: -0.0118801, 0.0259476, -0.0099007, 0.0241867, -0.0350296, 0.0358483
7: -0.0354761, 0.0226024, -0.0340049, 0.0195085, -0.0549846, 0.0566073
8: -0.0125955, 0.0436554, -0.0103662, 0.0406451, -0.0504169, 0.0540216
9: 0.8642685, 1.0071003, 0.8701790, 0.9995501, -0.1352816, 0.1203492

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0760957
time: 1.54 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0773740
time: 1.96 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0160097, 0.0154769, -0.0160097, 0.0154769, -0.0314866, 0.0314866
1: -0.0083188, 0.0477993, -0.0083188, 0.0477993, -0.0561181, 0.0561181
2: 0.0047449, 0.0397154, 0.0047449, 0.0397154, -0.0349705, 0.0349705
3: -0.0148234, 0.0256052, -0.0148234, 0.0256052, -0.0402409, 0.0402409
4: -0.0270774, 0.0069391, -0.0270774, 0.0069391, -0.0292125, 0.0292124
5: -0.0119602, 0.0376129, -0.0119602, 0.0376129, -0.0495731, 0.0495731
6: -0.0123997, 0.0210322, -0.0123997, 0.0210322, -0.0334319, 0.0334319
7: -0.0329190, 0.0151685, -0.0329190, 0.0151685, -0.0480875, 0.0480875
8: -0.0145032, 0.0343645, -0.0145032, 0.0343645, -0.0488677, 0.0488677
9: 0.8804663, 1.0167246, 0.8804663, 1.0167246, -0.1362582, 0.1362582

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0727799
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0749906
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0160097, 0.0154769, -0.0158550, 0.0220735, -0.0380833, 0.0313319
1: -0.0083188, 0.0477993, -0.0108087, 0.0508285, -0.0591473, 0.0586080
2: 0.0047449, 0.0397154, 0.0057412, 0.0420804, -0.0373355, 0.0339743
3: -0.0148234, 0.0256052, -0.0147648, 0.0277965, -0.0426199, 0.0403700
4: -0.0270774, 0.0069391, -0.0352937, 0.0169752, -0.0437405, 0.0399328
5: -0.0119602, 0.0376129, -0.0161317, 0.0399140, -0.0518742, 0.0537445
6: -0.0123997, 0.0210322, -0.0118801, 0.0259476, -0.0383472, 0.0329123
7: -0.0329190, 0.0151685, -0.0354761, 0.0226024, -0.0555214, 0.0506446
8: -0.0145032, 0.0343645, -0.0125955, 0.0436554, -0.0581586, 0.0469599
9: 0.8804663, 1.0167246, 0.8642685, 1.0071003, -0.1266340, 0.1524560

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0776353
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0789210
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0158550, 0.0220735, -0.0160097, 0.0154769, -0.0313319, 0.0380833
1: -0.0108087, 0.0508285, -0.0083188, 0.0477993, -0.0586080, 0.0591473
2: 0.0057412, 0.0420804, 0.0047449, 0.0397154, -0.0339743, 0.0373355
3: -0.0147648, 0.0277965, -0.0148234, 0.0256052, -0.0403700, 0.0426199
4: -0.0352937, 0.0169752, -0.0270774, 0.0069391, -0.0399328, 0.0437405
5: -0.0161317, 0.0399140, -0.0119602, 0.0376129, -0.0537445, 0.0518742
6: -0.0118801, 0.0259476, -0.0123997, 0.0210322, -0.0329123, 0.0383472
7: -0.0354761, 0.0226024, -0.0329190, 0.0151685, -0.0506446, 0.0555214
8: -0.0125955, 0.0436554, -0.0145032, 0.0343645, -0.0469599, 0.0581586
9: 0.8642685, 1.0071003, 0.8804663, 1.0167246, -0.1524560, 0.1266340

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0727799
time: 2.07 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0749906
time: 1.41 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0158550, 0.0220735, -0.0158550, 0.0220735, -0.0379286, 0.0379286
1: -0.0108087, 0.0508285, -0.0108087, 0.0508285, -0.0616371, 0.0616371
2: 0.0057412, 0.0420804, 0.0057412, 0.0420804, -0.0363393, 0.0363393
3: -0.0147648, 0.0277965, -0.0147648, 0.0277965, -0.0425613, 0.0425613
4: -0.0352937, 0.0169752, -0.0352937, 0.0169752, -0.0522688, 0.0522688
5: -0.0161317, 0.0399140, -0.0161317, 0.0399140, -0.0560456, 0.0560456
6: -0.0118801, 0.0259476, -0.0118801, 0.0259476, -0.0378276, 0.0378276
7: -0.0354761, 0.0226024, -0.0354761, 0.0226024, -0.0580785, 0.0580785
8: -0.0125955, 0.0436554, -0.0125955, 0.0436554, -0.0562509, 0.0562509
9: 0.8642685, 1.0071003, 0.8642685, 1.0071003, -0.1428318, 0.1428318

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0760958
time: 2.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0773740
time: 2.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.96 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0725829
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0751586
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0776223
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0789311
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0787060, upper bound: 0.0725829
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0789311, upper bound: 0.0751586
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0787061, upper bound: 0.0761589
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0789311, upper bound: 0.0775904
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0749920
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0776603
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0792818
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0809698
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0786935, upper bound: 0.0749920
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0789210, upper bound: 0.0776603
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0786935, upper bound: 0.0782554
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0789210, upper bound: 0.0799577
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0725649
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0749830
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0776223
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0789210
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0725649
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0749830
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0760957
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0773740
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0727799
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0749906
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0776353
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0789210
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0727799
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0749906
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0760958
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.96
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0773740

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0158237, 0.0127893, -0.0282749, 0.0282633
1: -0.0048880, 0.0458204, -0.0054148, 0.0461315, -0.0510194, 0.0512352
2: 0.0058141, 0.0375266, 0.0053458, 0.0376629, -0.0272217, 0.0287181
3: -0.0122397, 0.0244096, -0.0125702, 0.0245662, -0.0367212, 0.0363003
4: -0.0245753, 0.0045755, -0.0247824, 0.0048562, -0.0261381, 0.0259122
5: -0.0088945, 0.0361202, -0.0092804, 0.0363421, -0.0452366, 0.0454006
6: -0.0101256, 0.0199167, -0.0104036, 0.0199739, -0.0296113, 0.0292620
7: -0.0314879, 0.0111609, -0.0316892, 0.0117341, -0.0432219, 0.0428501
8: -0.0121138, 0.0322793, -0.0124057, 0.0326010, -0.0440397, 0.0434957
9: 0.8860436, 1.0078429, 0.8859093, 1.0092524, -0.1083079, 0.1094125

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0746708
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0751586
time: 1.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0139436, 0.0123793, -0.0149409, 0.0196335, -0.0335770, 0.0273202
1: -0.0039181, 0.0450282, -0.0062979, 0.0487404, -0.0526585, 0.0513262
2: 0.0087891, 0.0374261, 0.0076341, 0.0406328, -0.0288164, 0.0297920
3: -0.0117863, 0.0240710, -0.0118784, 0.0265774, -0.0379794, 0.0359494
4: -0.0243782, 0.0038566, -0.0329555, 0.0112102, -0.0354764, 0.0337256
5: -0.0083567, 0.0356048, -0.0115860, 0.0383802, -0.0467369, 0.0471908
6: -0.0096787, 0.0197634, -0.0094847, 0.0239234, -0.0324384, 0.0291015
7: -0.0310302, 0.0107769, -0.0337510, 0.0186734, -0.0497036, 0.0445279
8: -0.0118135, 0.0314438, -0.0099292, 0.0399961, -0.0500574, 0.0406410
9: 0.8858225, 1.0042065, 0.8706422, 0.9974203, -0.1022274, 0.1223500

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0776223
time: 2.20 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0776223
time: 1.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0155506, 0.0200678, -0.0355534, 0.0279902
1: -0.0048880, 0.0458204, -0.0071896, 0.0491037, -0.0539917, 0.0530100
2: 0.0058141, 0.0375266, 0.0065629, 0.0408218, -0.0312516, 0.0309637
3: -0.0122397, 0.0244096, -0.0124121, 0.0267453, -0.0389851, 0.0368217
4: -0.0245753, 0.0045755, -0.0333054, 0.0123178, -0.0368931, 0.0345958
5: -0.0088945, 0.0361202, -0.0124739, 0.0386236, -0.0475181, 0.0485941
6: -0.0101256, 0.0199167, -0.0099007, 0.0241867, -0.0336739, 0.0298174
7: -0.0314879, 0.0111609, -0.0340049, 0.0195085, -0.0509964, 0.0451658
8: -0.0121138, 0.0322793, -0.0103662, 0.0406451, -0.0514383, 0.0422346
9: 0.8860436, 1.0078429, 0.8701790, 0.9995501, -0.1037695, 0.1279643

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0787061
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0789311
time: 2.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0138315, 0.0199521, -0.0151920, 0.0124382, -0.0262698, 0.0351441
1: -0.0060836, 0.0481609, -0.0047071, 0.0457318, -0.0518155, 0.0528680
2: 0.0093802, 0.0407030, 0.0065071, 0.0374966, -0.0281165, 0.0313615
3: -0.0115937, 0.0263188, -0.0121456, 0.0243858, -0.0359795, 0.0381350
4: -0.0331513, 0.0123093, -0.0245049, 0.0044634, -0.0347209, 0.0367731
5: -0.0117952, 0.0380069, -0.0087802, 0.0360743, -0.0478695, 0.0467871
6: -0.0091718, 0.0240976, -0.0100419, 0.0198866, -0.0288499, 0.0330206
7: -0.0335392, 0.0189189, -0.0314487, 0.0110592, -0.0445984, 0.0503676
8: -0.0095678, 0.0400491, -0.0120315, 0.0321661, -0.0409416, 0.0504136
9: 0.8698146, 0.9941387, 0.8861243, 1.0072381, -0.1261080, 0.0987768

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0725829
time: 1.96 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0725829
time: 1.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0158237, 0.0127893, -0.0279723, 0.0354768
1: -0.0064677, 0.0488046, -0.0054148, 0.0461315, -0.0525991, 0.0542194
2: 0.0071338, 0.0406640, 0.0053458, 0.0376629, -0.0305149, 0.0323654
3: -0.0119915, 0.0266010, -0.0125702, 0.0245662, -0.0365577, 0.0390242
4: -0.0330374, 0.0113035, -0.0247824, 0.0048562, -0.0347601, 0.0360859
5: -0.0117166, 0.0384160, -0.0092804, 0.0363421, -0.0480587, 0.0476964
6: -0.0095804, 0.0239720, -0.0104036, 0.0199739, -0.0295544, 0.0333560
7: -0.0337955, 0.0187752, -0.0316892, 0.0117341, -0.0455295, 0.0504644
8: -0.0100217, 0.0401521, -0.0124057, 0.0326010, -0.0426227, 0.0510033
9: 0.8705507, 0.9979835, 0.8859093, 1.0092524, -0.1269324, 0.1046924

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0746708
time: 1.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0751586
time: 1.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0138315, 0.0199521, -0.0149409, 0.0196335, -0.0334650, 0.0348930
1: -0.0060836, 0.0481609, -0.0062979, 0.0487404, -0.0548240, 0.0544588
2: 0.0093802, 0.0407030, 0.0076341, 0.0406328, -0.0312527, 0.0330689
3: -0.0115937, 0.0263188, -0.0118784, 0.0265774, -0.0381711, 0.0381972
4: -0.0331513, 0.0123093, -0.0329555, 0.0112102, -0.0437201, 0.0447718
5: -0.0117952, 0.0380069, -0.0115860, 0.0383802, -0.0501754, 0.0495929
6: -0.0091718, 0.0240976, -0.0094847, 0.0239234, -0.0318836, 0.0322593
7: -0.0335392, 0.0189189, -0.0337510, 0.0186734, -0.0522126, 0.0526699
8: -0.0095678, 0.0400491, -0.0099292, 0.0399961, -0.0462422, 0.0465630
9: 0.8698146, 0.9941387, 0.8706422, 0.9974203, -0.1088926, 0.1054013

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0701141, upper bound: 0.0674527
time: 3.33 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792138, upper bound: 0.0755612
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0155506, 0.0200678, -0.0352509, 0.0352037
1: -0.0064677, 0.0488046, -0.0071896, 0.0491037, -0.0555714, 0.0559942
2: 0.0071338, 0.0406640, 0.0065629, 0.0408218, -0.0336389, 0.0341011
3: -0.0119915, 0.0266010, -0.0124121, 0.0267453, -0.0387368, 0.0390131
4: -0.0330374, 0.0113035, -0.0333054, 0.0123178, -0.0449105, 0.0437766
5: -0.0117166, 0.0384160, -0.0124739, 0.0386236, -0.0503402, 0.0508898
6: -0.0095804, 0.0239720, -0.0099007, 0.0241867, -0.0330158, 0.0327709
7: -0.0337955, 0.0187752, -0.0340049, 0.0195085, -0.0533040, 0.0527801
8: -0.0100217, 0.0401521, -0.0103662, 0.0406451, -0.0476979, 0.0474187
9: 0.8705507, 0.9979835, 0.8701790, 0.9995501, -0.1102267, 0.1107234

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0775261
time: 2.04 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0775904
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0160097, 0.0154769, -0.0309624, 0.0284493
1: -0.0048880, 0.0458204, -0.0083188, 0.0477993, -0.0526873, 0.0541392
2: 0.0058141, 0.0375266, 0.0047449, 0.0397154, -0.0339013, 0.0294593
3: -0.0122397, 0.0244096, -0.0148234, 0.0256052, -0.0378449, 0.0386107
4: -0.0245753, 0.0045755, -0.0270774, 0.0069391, -0.0283968, 0.0280288
5: -0.0088945, 0.0361202, -0.0119602, 0.0376129, -0.0465074, 0.0480804
6: -0.0101256, 0.0199167, -0.0123997, 0.0210322, -0.0311579, 0.0312754
7: -0.0314879, 0.0111609, -0.0329190, 0.0151685, -0.0466564, 0.0440799
8: -0.0121138, 0.0322793, -0.0145032, 0.0343645, -0.0464783, 0.0457296
9: 0.8860436, 1.0078429, 0.8804663, 1.0167246, -0.1167963, 0.1273766

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0773646
time: 1.41 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0776603
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0139436, 0.0123793, -0.0152598, 0.0215008, -0.0354444, 0.0276391
1: -0.0039181, 0.0450282, -0.0098583, 0.0504714, -0.0543894, 0.0548866
2: 0.0087891, 0.0374261, 0.0068035, 0.0417080, -0.0329189, 0.0306226
3: -0.0117863, 0.0240710, -0.0141885, 0.0276380, -0.0394243, 0.0382594
4: -0.0243782, 0.0038566, -0.0349385, 0.0157141, -0.0400923, 0.0358041
5: -0.0083567, 0.0356048, -0.0151925, 0.0396773, -0.0480340, 0.0507973
6: -0.0096787, 0.0197634, -0.0114269, 0.0256428, -0.0353215, 0.0307879
7: -0.0310302, 0.0107769, -0.0351832, 0.0217290, -0.0527592, 0.0459601
8: -0.0118135, 0.0314438, -0.0121206, 0.0429541, -0.0547676, 0.0427145
9: 0.8858225, 1.0042065, 0.8648540, 1.0048254, -0.1086554, 0.1393526

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0792818
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0792817
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0158550, 0.0220735, -0.0375591, 0.0282946
1: -0.0048880, 0.0458204, -0.0108087, 0.0508285, -0.0557164, 0.0566290
2: 0.0058141, 0.0375266, 0.0057412, 0.0420804, -0.0362663, 0.0317854
3: -0.0122397, 0.0244096, -0.0147648, 0.0277965, -0.0400362, 0.0391745
4: -0.0245753, 0.0045755, -0.0352937, 0.0169752, -0.0415505, 0.0365492
5: -0.0088945, 0.0361202, -0.0161317, 0.0399140, -0.0488085, 0.0522519
6: -0.0101256, 0.0199167, -0.0118801, 0.0259476, -0.0360732, 0.0315696
7: -0.0314879, 0.0111609, -0.0354761, 0.0226024, -0.0540903, 0.0466370
8: -0.0121138, 0.0322793, -0.0125955, 0.0436554, -0.0557692, 0.0443674
9: 0.8860436, 1.0078429, 0.8642685, 1.0071003, -0.1103143, 0.1435744

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0807830
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0809698
time: 1.77 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0138315, 0.0199521, -0.0153810, 0.0150026, -0.0288342, 0.0353332
1: -0.0060836, 0.0481609, -0.0076001, 0.0474064, -0.0534900, 0.0557610
2: 0.0093802, 0.0407030, 0.0059016, 0.0393924, -0.0300123, 0.0321015
3: -0.0115937, 0.0263188, -0.0143770, 0.0254280, -0.0370217, 0.0403985
4: -0.0331513, 0.0123093, -0.0267911, 0.0064648, -0.0389573, 0.0389501
5: -0.0117952, 0.0380069, -0.0114380, 0.0373461, -0.0491413, 0.0494449
6: -0.0091718, 0.0240976, -0.0120226, 0.0209235, -0.0300954, 0.0349730
7: -0.0335392, 0.0189189, -0.0326497, 0.0144500, -0.0479892, 0.0515686
8: -0.0095678, 0.0400491, -0.0141028, 0.0339358, -0.0435037, 0.0524666
9: 0.8698146, 0.9941387, 0.8807912, 1.0146401, -0.1342411, 0.1133475

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0749920
time: 2.06 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0749920
time: 1.37 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0160097, 0.0154769, -0.0306599, 0.0356628
1: -0.0064677, 0.0488046, -0.0083188, 0.0477993, -0.0542670, 0.0571234
2: 0.0071338, 0.0406640, 0.0047449, 0.0397154, -0.0325816, 0.0331066
3: -0.0119915, 0.0266010, -0.0148234, 0.0256052, -0.0375967, 0.0413347
4: -0.0330374, 0.0113035, -0.0270774, 0.0069391, -0.0389105, 0.0381552
5: -0.0117166, 0.0384160, -0.0119602, 0.0376129, -0.0493295, 0.0503762
6: -0.0095804, 0.0239720, -0.0123997, 0.0210322, -0.0306126, 0.0353586
7: -0.0337955, 0.0187752, -0.0329190, 0.0151685, -0.0489640, 0.0516942
8: -0.0100217, 0.0401521, -0.0145032, 0.0343645, -0.0443862, 0.0531200
9: 0.8705507, 0.9979835, 0.8804663, 1.0167246, -0.1354209, 0.1175172

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0773646
time: 1.16 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0776603
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0138315, 0.0199521, -0.0152598, 0.0215008, -0.0353324, 0.0352120
1: -0.0060836, 0.0481609, -0.0098583, 0.0504714, -0.0565550, 0.0580193
2: 0.0093802, 0.0407030, 0.0068035, 0.0417080, -0.0323278, 0.0338995
3: -0.0115937, 0.0263188, -0.0141885, 0.0276380, -0.0392317, 0.0404992
4: -0.0331513, 0.0123093, -0.0349385, 0.0157141, -0.0488653, 0.0469415
5: -0.0117952, 0.0380069, -0.0151925, 0.0396773, -0.0514725, 0.0531994
6: -0.0091718, 0.0240976, -0.0114269, 0.0256428, -0.0348146, 0.0340902
7: -0.0335392, 0.0189189, -0.0351832, 0.0217290, -0.0552682, 0.0541020
8: -0.0095678, 0.0400491, -0.0121206, 0.0429541, -0.0525220, 0.0489225
9: 0.8698146, 0.9941387, 0.8648540, 1.0048254, -0.1174052, 0.1292847

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0701141, upper bound: 0.0696285
time: 3.66 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0791874, upper bound: 0.0776548
time: 2.49 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0158550, 0.0220735, -0.0372566, 0.0355081
1: -0.0064677, 0.0488046, -0.0108087, 0.0508285, -0.0572961, 0.0596133
2: 0.0071338, 0.0406640, 0.0057412, 0.0420804, -0.0349466, 0.0349228
3: -0.0119915, 0.0266010, -0.0147648, 0.0277965, -0.0397880, 0.0413658
4: -0.0330374, 0.0113035, -0.0352937, 0.0169752, -0.0500126, 0.0458646
5: -0.0117166, 0.0384160, -0.0161317, 0.0399140, -0.0516306, 0.0545476
6: -0.0095804, 0.0239720, -0.0118801, 0.0259476, -0.0355280, 0.0346662
7: -0.0337955, 0.0187752, -0.0354761, 0.0226024, -0.0563979, 0.0542513
8: -0.0100217, 0.0401521, -0.0125955, 0.0436554, -0.0536771, 0.0498501
9: 0.8705507, 0.9979835, 0.8642685, 1.0071003, -0.1191385, 0.1337150

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0799408
time: 2.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0799577
time: 1.18 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0141481, 0.0148794, -0.0151920, 0.0124382, -0.0265864, 0.0300714
1: -0.0067327, 0.0467287, -0.0047071, 0.0457318, -0.0524645, 0.0514358
2: 0.0082062, 0.0392772, 0.0065071, 0.0374966, -0.0259008, 0.0327701
3: -0.0139275, 0.0251292, -0.0121456, 0.0243858, -0.0374012, 0.0372748
4: -0.0265133, 0.0058491, -0.0245049, 0.0044634, -0.0275643, 0.0271830
5: -0.0109288, 0.0368935, -0.0087802, 0.0360743, -0.0470031, 0.0456737
6: -0.0115884, 0.0208009, -0.0100419, 0.0198866, -0.0302452, 0.0308428
7: -0.0322522, 0.0140456, -0.0314487, 0.0110592, -0.0433114, 0.0454942
8: -0.0138035, 0.0332257, -0.0120315, 0.0321661, -0.0446046, 0.0452572
9: 0.8805467, 1.0114024, 0.8861243, 1.0072381, -0.1266915, 0.1115484

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0725649
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0725649
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0156687, 0.0150452, -0.0158237, 0.0127893, -0.0284580, 0.0308689
1: -0.0078102, 0.0474758, -0.0054148, 0.0461315, -0.0539416, 0.0528905
2: 0.0052127, 0.0394592, 0.0053458, 0.0376629, -0.0279410, 0.0341134
3: -0.0145044, 0.0254460, -0.0125702, 0.0245662, -0.0389872, 0.0380162
4: -0.0268684, 0.0066083, -0.0247824, 0.0048562, -0.0284058, 0.0280011
5: -0.0115846, 0.0373834, -0.0092804, 0.0363421, -0.0479267, 0.0466638
6: -0.0121317, 0.0209602, -0.0104036, 0.0199739, -0.0315649, 0.0313638
7: -0.0326990, 0.0146040, -0.0316892, 0.0117341, -0.0444331, 0.0462932
8: -0.0142110, 0.0340559, -0.0124057, 0.0326010, -0.0461946, 0.0464616
9: 0.8806884, 1.0153494, 0.8859093, 1.0092524, -0.1285640, 0.1171161

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0745462
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0749830
time: 1.52 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0141481, 0.0148794, -0.0149409, 0.0196335, -0.0337816, 0.0298202
1: -0.0067327, 0.0467287, -0.0062979, 0.0487404, -0.0554731, 0.0530266
2: 0.0082062, 0.0392772, 0.0076341, 0.0406328, -0.0295483, 0.0316431
3: -0.0139275, 0.0251292, -0.0118784, 0.0265774, -0.0401481, 0.0370077
4: -0.0265133, 0.0058491, -0.0329555, 0.0112102, -0.0375050, 0.0377339
5: -0.0109288, 0.0368935, -0.0115860, 0.0383802, -0.0493090, 0.0484795
6: -0.0115884, 0.0208009, -0.0094847, 0.0239234, -0.0343280, 0.0302855
7: -0.0322522, 0.0140456, -0.0337510, 0.0186734, -0.0509256, 0.0477966
8: -0.0138035, 0.0332257, -0.0099292, 0.0399961, -0.0520562, 0.0431549
9: 0.8805467, 1.0114024, 0.8706422, 0.9974203, -0.1168736, 0.1300463

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0776223
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0776223
time: 1.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0156687, 0.0150452, -0.0155506, 0.0200678, -0.0357365, 0.0305958
1: -0.0078102, 0.0474758, -0.0071896, 0.0491037, -0.0569139, 0.0546654
2: 0.0052127, 0.0394592, 0.0065629, 0.0408218, -0.0319709, 0.0328963
3: -0.0145044, 0.0254460, -0.0124121, 0.0267453, -0.0412497, 0.0378581
4: -0.0268684, 0.0066083, -0.0333054, 0.0123178, -0.0391861, 0.0385792
5: -0.0115846, 0.0373834, -0.0124739, 0.0386236, -0.0502082, 0.0498573
6: -0.0121317, 0.0209602, -0.0099007, 0.0241867, -0.0356161, 0.0308609
7: -0.0326990, 0.0146040, -0.0340049, 0.0195085, -0.0522075, 0.0486089
8: -0.0142110, 0.0340559, -0.0103662, 0.0406451, -0.0534867, 0.0444222
9: 0.8806884, 1.0153494, 0.8701790, 0.9995501, -0.1188617, 0.1356678

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0786935
time: 1.60 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0789210
time: 1.42 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0141690, 0.0217598, -0.0151920, 0.0124382, -0.0266073, 0.0369518
1: -0.0095564, 0.0499182, -0.0047071, 0.0457318, -0.0552882, 0.0546253
2: 0.0080325, 0.0416972, 0.0065071, 0.0374966, -0.0294641, 0.0351901
3: -0.0138287, 0.0274024, -0.0121456, 0.0243858, -0.0382145, 0.0395481
4: -0.0351308, 0.0167084, -0.0245049, 0.0044634, -0.0366142, 0.0412133
5: -0.0152888, 0.0393312, -0.0087802, 0.0360743, -0.0513631, 0.0481114
6: -0.0110571, 0.0257505, -0.0100419, 0.0198866, -0.0304838, 0.0357924
7: -0.0349826, 0.0218801, -0.0314487, 0.0110592, -0.0460418, 0.0533288
8: -0.0116964, 0.0428943, -0.0120315, 0.0321661, -0.0429832, 0.0549259
9: 0.8642952, 1.0012407, 0.8861243, 1.0072381, -0.1429430, 0.1049630

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0725649
time: 1.17 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0725649
time: 3.44 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0158237, 0.0127893, -0.0282730, 0.0373897
1: -0.0100882, 0.0505141, -0.0054148, 0.0461315, -0.0562196, 0.0559289
2: 0.0063352, 0.0417820, 0.0053458, 0.0376629, -0.0313277, 0.0364361
3: -0.0143435, 0.0276428, -0.0125702, 0.0245662, -0.0389097, 0.0402130
4: -0.0350397, 0.0158828, -0.0247824, 0.0048562, -0.0368444, 0.0406652
5: -0.0153792, 0.0396903, -0.0092804, 0.0363421, -0.0517213, 0.0489707
6: -0.0115589, 0.0257092, -0.0104036, 0.0199739, -0.0315329, 0.0361129
7: -0.0352443, 0.0218823, -0.0316892, 0.0117341, -0.0469784, 0.0535715
8: -0.0122451, 0.0431442, -0.0124057, 0.0326010, -0.0447887, 0.0555498
9: 0.8647262, 1.0055966, 0.8859093, 1.0092524, -0.1445262, 0.1111335

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0745462
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0749830
time: 1.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0141690, 0.0217598, -0.0149409, 0.0196335, -0.0338025, 0.0367006
1: -0.0095564, 0.0499182, -0.0062979, 0.0487404, -0.0582968, 0.0562161
2: 0.0080325, 0.0416972, 0.0076341, 0.0406328, -0.0326003, 0.0340631
3: -0.0138287, 0.0274024, -0.0118784, 0.0265774, -0.0404061, 0.0392809
4: -0.0351308, 0.0167084, -0.0329555, 0.0112102, -0.0457415, 0.0496638
5: -0.0152888, 0.0393312, -0.0115860, 0.0383802, -0.0536689, 0.0509172
6: -0.0110571, 0.0257505, -0.0094847, 0.0239234, -0.0336836, 0.0352352
7: -0.0349826, 0.0218801, -0.0337510, 0.0186734, -0.0536560, 0.0556311
8: -0.0116964, 0.0428943, -0.0099292, 0.0399961, -0.0485700, 0.0528235
9: 0.8642952, 1.0012407, 0.8706422, 0.9974203, -0.1331251, 0.1134995

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0723261, upper bound: 0.0674526
time: 1.52 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0814308, upper bound: 0.0754985
time: 1.14 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0155506, 0.0200678, -0.0355516, 0.0371167
1: -0.0100882, 0.0505141, -0.0071896, 0.0491037, -0.0591919, 0.0577037
2: 0.0063352, 0.0417820, 0.0065629, 0.0408218, -0.0344866, 0.0352191
3: -0.0143435, 0.0276428, -0.0124121, 0.0267453, -0.0410888, 0.0400549
4: -0.0350397, 0.0158828, -0.0333054, 0.0123178, -0.0470915, 0.0491881
5: -0.0153792, 0.0396903, -0.0124739, 0.0386236, -0.0540028, 0.0521642
6: -0.0115589, 0.0257092, -0.0099007, 0.0241867, -0.0348398, 0.0356099
7: -0.0352443, 0.0218823, -0.0340049, 0.0195085, -0.0547528, 0.0558872
8: -0.0122451, 0.0431442, -0.0103662, 0.0406451, -0.0500480, 0.0535104
9: 0.8647262, 1.0055966, 0.8701790, 0.9995501, -0.1348239, 0.1188218

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773186
time: 1.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773740
time: 2.49 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0141481, 0.0148794, -0.0153810, 0.0150026, -0.0291508, 0.0302604
1: -0.0067327, 0.0467287, -0.0076001, 0.0474064, -0.0541391, 0.0543288
2: 0.0082062, 0.0392772, 0.0059016, 0.0393924, -0.0311862, 0.0333756
3: -0.0139275, 0.0251292, -0.0143770, 0.0254280, -0.0387096, 0.0388449
4: -0.0265133, 0.0058491, -0.0267911, 0.0064648, -0.0283048, 0.0278957
5: -0.0109288, 0.0368935, -0.0114380, 0.0373461, -0.0482749, 0.0483315
6: -0.0115884, 0.0208009, -0.0120226, 0.0209235, -0.0325119, 0.0328235
7: -0.0322522, 0.0140456, -0.0326497, 0.0144500, -0.0467022, 0.0466952
8: -0.0138035, 0.0332257, -0.0141028, 0.0339358, -0.0477393, 0.0473285
9: 0.8805467, 1.0114024, 0.8807912, 1.0146401, -0.1340934, 0.1306111

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0727799
time: 1.41 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0727799
time: 2.52 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0156687, 0.0150452, -0.0160097, 0.0154769, -0.0311456, 0.0310549
1: -0.0078102, 0.0474758, -0.0083188, 0.0477993, -0.0556095, 0.0557946
2: 0.0052127, 0.0394592, 0.0047449, 0.0397154, -0.0345028, 0.0347143
3: -0.0145044, 0.0254460, -0.0148234, 0.0256052, -0.0401095, 0.0398497
4: -0.0268684, 0.0066083, -0.0270774, 0.0069391, -0.0289842, 0.0286998
5: -0.0115846, 0.0373834, -0.0119602, 0.0376129, -0.0491975, 0.0493436
6: -0.0121317, 0.0209602, -0.0123997, 0.0210322, -0.0331640, 0.0333598
7: -0.0326990, 0.0146040, -0.0329190, 0.0151685, -0.0478675, 0.0475230
8: -0.0142110, 0.0340559, -0.0145032, 0.0343645, -0.0485754, 0.0485592
9: 0.8806884, 1.0153494, 0.8804663, 1.0167246, -0.1360362, 0.1348830

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0746103
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0749906
time: 1.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0141481, 0.0148794, -0.0152598, 0.0215008, -0.0356490, 0.0301392
1: -0.0067327, 0.0467287, -0.0098583, 0.0504714, -0.0572040, 0.0565870
2: 0.0082062, 0.0392772, 0.0068035, 0.0417080, -0.0335018, 0.0324737
3: -0.0139275, 0.0251292, -0.0141885, 0.0276380, -0.0414506, 0.0393177
4: -0.0265133, 0.0058491, -0.0349385, 0.0157141, -0.0419089, 0.0385660
5: -0.0109288, 0.0368935, -0.0151925, 0.0396773, -0.0506061, 0.0520860
6: -0.0115884, 0.0208009, -0.0114269, 0.0256428, -0.0372312, 0.0322278
7: -0.0322522, 0.0140456, -0.0351832, 0.0217290, -0.0539812, 0.0492287
8: -0.0138035, 0.0332257, -0.0121206, 0.0429541, -0.0567577, 0.0453463
9: 0.8805467, 1.0114024, 0.8648540, 1.0048254, -0.1242787, 0.1465484

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0776352
time: 2.12 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0776352
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0156687, 0.0150452, -0.0158550, 0.0220735, -0.0377423, 0.0309002
1: -0.0078102, 0.0474758, -0.0108087, 0.0508285, -0.0586386, 0.0582844
2: 0.0052127, 0.0394592, 0.0057412, 0.0420804, -0.0368678, 0.0337180
3: -0.0145044, 0.0254460, -0.0147648, 0.0277965, -0.0423009, 0.0402108
4: -0.0268684, 0.0066083, -0.0352937, 0.0169752, -0.0435122, 0.0394222
5: -0.0115846, 0.0373834, -0.0161317, 0.0399140, -0.0514986, 0.0535150
6: -0.0121317, 0.0209602, -0.0118801, 0.0259476, -0.0380793, 0.0328402
7: -0.0326990, 0.0146040, -0.0354761, 0.0226024, -0.0553014, 0.0500800
8: -0.0142110, 0.0340559, -0.0125955, 0.0436554, -0.0578664, 0.0466514
9: 0.8806884, 1.0153494, 0.8642685, 1.0071003, -0.1264119, 0.1510808

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0786936
time: 3.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0789210
time: 1.40 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0141690, 0.0217598, -0.0153810, 0.0150026, -0.0291717, 0.0371408
1: -0.0095564, 0.0499182, -0.0076001, 0.0474064, -0.0569628, 0.0575182
2: 0.0080325, 0.0416972, 0.0059016, 0.0393924, -0.0313599, 0.0357956
3: -0.0138287, 0.0274024, -0.0143770, 0.0254280, -0.0392567, 0.0417062
4: -0.0351308, 0.0167084, -0.0267911, 0.0064648, -0.0396573, 0.0430312
5: -0.0152888, 0.0393312, -0.0114380, 0.0373461, -0.0526349, 0.0507692
6: -0.0110571, 0.0257505, -0.0120226, 0.0209235, -0.0319806, 0.0377731
7: -0.0349826, 0.0218801, -0.0326497, 0.0144500, -0.0494326, 0.0545298
8: -0.0116964, 0.0428943, -0.0141028, 0.0339358, -0.0456323, 0.0569971
9: 0.8642952, 1.0012407, 0.8807912, 1.0146401, -0.1503449, 0.1204495

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0727799
time: 1.25 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0727799
time: 1.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0160097, 0.0154769, -0.0309606, 0.0375758
1: -0.0100882, 0.0505141, -0.0083188, 0.0477993, -0.0578875, 0.0588329
2: 0.0063352, 0.0417820, 0.0047449, 0.0397154, -0.0333803, 0.0370371
3: -0.0143435, 0.0276428, -0.0148234, 0.0256052, -0.0399487, 0.0424662
4: -0.0350397, 0.0158828, -0.0270774, 0.0069391, -0.0396415, 0.0424947
5: -0.0153792, 0.0396903, -0.0119602, 0.0376129, -0.0529921, 0.0516506
6: -0.0115589, 0.0257092, -0.0123997, 0.0210322, -0.0325912, 0.0381089
7: -0.0352443, 0.0218823, -0.0329190, 0.0151685, -0.0504128, 0.0548013
8: -0.0122451, 0.0431442, -0.0145032, 0.0343645, -0.0466096, 0.0576474
9: 0.8647262, 1.0055966, 0.8804663, 1.0167246, -0.1519984, 0.1251303

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0746103
time: 1.23 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0749906
time: 6.82 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0141690, 0.0217598, -0.0152598, 0.0215008, -0.0356699, 0.0370196
1: -0.0095564, 0.0499182, -0.0098583, 0.0504714, -0.0600277, 0.0597765
2: 0.0080325, 0.0416972, 0.0068035, 0.0417080, -0.0336755, 0.0348937
3: -0.0138287, 0.0274024, -0.0141885, 0.0276380, -0.0414666, 0.0415909
4: -0.0351308, 0.0167084, -0.0349385, 0.0157141, -0.0508449, 0.0516468
5: -0.0152888, 0.0393312, -0.0151925, 0.0396773, -0.0549661, 0.0545237
6: -0.0110571, 0.0257505, -0.0114269, 0.0256428, -0.0366999, 0.0371774
7: -0.0349826, 0.0218801, -0.0351832, 0.0217290, -0.0567116, 0.0570633
8: -0.0116964, 0.0428943, -0.0121206, 0.0429541, -0.0546506, 0.0550150
9: 0.8642952, 1.0012407, 0.8648540, 1.0048254, -0.1405302, 0.1363868

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0723261, upper bound: 0.0676351
time: 1.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0814308, upper bound: 0.0754988
time: 1.87 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0158550, 0.0220735, -0.0375573, 0.0374210
1: -0.0100882, 0.0505141, -0.0108087, 0.0508285, -0.0609167, 0.0613228
2: 0.0063352, 0.0417820, 0.0057412, 0.0420804, -0.0357453, 0.0360408
3: -0.0143435, 0.0276428, -0.0147648, 0.0277965, -0.0421400, 0.0424076
4: -0.0350397, 0.0158828, -0.0352937, 0.0169752, -0.0520148, 0.0511764
5: -0.0153792, 0.0396903, -0.0161317, 0.0399140, -0.0552932, 0.0558220
6: -0.0115589, 0.0257092, -0.0118801, 0.0259476, -0.0375065, 0.0375893
7: -0.0352443, 0.0218823, -0.0354761, 0.0226024, -0.0578467, 0.0573583
8: -0.0122451, 0.0431442, -0.0125955, 0.0436554, -0.0559005, 0.0557396
9: 0.8647262, 1.0055966, 0.8642685, 1.0071003, -0.1423742, 0.1413281

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773186
time: 1.44 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773740
time: 1.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.79 seconds
NS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0746708
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0751586
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0776223
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0776223
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0787061
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0789311
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0725829
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0725829
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0746708
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0751586
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0701141, upper bound: 0.0674527
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0792138, upper bound: 0.0755612
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0775261
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0775904
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0773646
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0776603
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0792818
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0792817
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0807830
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0809698
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0749920
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0749920
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0773646
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0776603
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0701141, upper bound: 0.0696285
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0791874, upper bound: 0.0776548
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0799408
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0799577
NS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0725649
NS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0725649
NS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0745462
NS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0749830
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0776223
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0776223
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0786935
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0789210
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0725649
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0725649
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0745462
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0749830
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0723261, upper bound: 0.0674526
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0814308, upper bound: 0.0754985
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773186
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773740
NS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0727799
NS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0727799
NS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0746103
NS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0749906
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0776352
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0776352
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0786936
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0789210
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0727799
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0727799
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0746103
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0749906
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0723261, upper bound: 0.0676351
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0814308, upper bound: 0.0754988
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773186
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.79
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773740

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0154856, 0.0124396, -0.0279252, 0.0279252
1: -0.0048880, 0.0458204, -0.0048880, 0.0458204, -0.0507083, 0.0507083
2: 0.0058141, 0.0375266, 0.0058141, 0.0375266, -0.0270761, 0.0270761
3: -0.0122397, 0.0244096, -0.0122397, 0.0244096, -0.0364096, 0.0364096
4: -0.0245753, 0.0045755, -0.0245753, 0.0045755, -0.0257186, 0.0257186
5: -0.0088945, 0.0361202, -0.0088945, 0.0361202, -0.0450147, 0.0450147
6: -0.0101256, 0.0199167, -0.0101256, 0.0199167, -0.0293706, 0.0293706
7: -0.0314879, 0.0111609, -0.0314879, 0.0111609, -0.0426488, 0.0426488
8: -0.0121138, 0.0322793, -0.0121138, 0.0322793, -0.0435847, 0.0435847
9: 0.8860436, 1.0078429, 0.8860436, 1.0078429, -0.1085644, 0.1085644

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0718129, upper bound: 0.0743186
time: 1.43 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0718145, upper bound: 0.0743666
time: 1.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0139436, 0.0123793, -0.0138315, 0.0199521, -0.0338957, 0.0262108
1: -0.0039181, 0.0450282, -0.0060836, 0.0481609, -0.0520790, 0.0511119
2: 0.0087891, 0.0374261, 0.0093802, 0.0407030, -0.0290434, 0.0280459
3: -0.0117863, 0.0240710, -0.0115937, 0.0263188, -0.0377625, 0.0356647
4: -0.0243782, 0.0038566, -0.0331513, 0.0123093, -0.0366356, 0.0342022
5: -0.0083567, 0.0356048, -0.0117952, 0.0380069, -0.0463636, 0.0474000
6: -0.0096787, 0.0197634, -0.0091718, 0.0240976, -0.0326416, 0.0287131
7: -0.0310302, 0.0107769, -0.0335392, 0.0189189, -0.0499491, 0.0443161
8: -0.0118135, 0.0314438, -0.0095678, 0.0400491, -0.0501497, 0.0401717
9: 0.8858225, 1.0042065, 0.8698146, 0.9941387, -0.0994247, 0.1236797

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718129, upper bound: 0.0768203
time: 1.12 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718145, upper bound: 0.0769588
time: 2.12 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0139436, 0.0123793, -0.0151830, 0.0196531, -0.0335966, 0.0275623
1: -0.0039181, 0.0450282, -0.0064677, 0.0488046, -0.0527227, 0.0514959
2: 0.0087891, 0.0374261, 0.0071338, 0.0406640, -0.0288538, 0.0302858
3: -0.0117863, 0.0240710, -0.0119915, 0.0266010, -0.0380780, 0.0360625
4: -0.0243782, 0.0038566, -0.0330374, 0.0113035, -0.0353115, 0.0337984
5: -0.0083567, 0.0356048, -0.0117166, 0.0384160, -0.0467727, 0.0473214
6: -0.0096787, 0.0197634, -0.0095804, 0.0239720, -0.0324810, 0.0290034
7: -0.0310302, 0.0107769, -0.0337955, 0.0187752, -0.0498054, 0.0445724
8: -0.0118135, 0.0314438, -0.0100217, 0.0401521, -0.0502168, 0.0405711
9: 0.8858225, 1.0042065, 0.8705507, 0.9979835, -0.1021972, 0.1223192

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718129, upper bound: 0.0768203
time: 1.92 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718145, upper bound: 0.0769588
time: 1.88 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0138315, 0.0199521, -0.0354377, 0.0262711
1: -0.0048880, 0.0458204, -0.0060836, 0.0481609, -0.0530489, 0.0519040
2: 0.0058141, 0.0375266, 0.0093802, 0.0407030, -0.0320711, 0.0281464
3: -0.0122397, 0.0244096, -0.0115937, 0.0263188, -0.0380564, 0.0360033
4: -0.0245753, 0.0045755, -0.0331513, 0.0123093, -0.0368577, 0.0347304
5: -0.0088945, 0.0361202, -0.0117952, 0.0380069, -0.0469014, 0.0479154
6: -0.0101256, 0.0199167, -0.0091718, 0.0240976, -0.0329322, 0.0288922
7: -0.0314879, 0.0111609, -0.0335392, 0.0189189, -0.0504068, 0.0447001
8: -0.0121138, 0.0322793, -0.0095678, 0.0400491, -0.0503330, 0.0411638
9: 0.8860436, 1.0078429, 0.8698146, 0.9941387, -0.0986193, 0.1261612

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718129, upper bound: 0.0779467
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718145, upper bound: 0.0780380
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0151830, 0.0196531, -0.0351386, 0.0276226
1: -0.0048880, 0.0458204, -0.0064677, 0.0488046, -0.0536926, 0.0522881
2: 0.0058141, 0.0375266, 0.0071338, 0.0406640, -0.0310852, 0.0303692
3: -0.0122397, 0.0244096, -0.0119915, 0.0266010, -0.0388407, 0.0364011
4: -0.0245753, 0.0045755, -0.0330374, 0.0113035, -0.0358788, 0.0343269
5: -0.0088945, 0.0361202, -0.0117166, 0.0384160, -0.0473105, 0.0478368
6: -0.0101256, 0.0199167, -0.0095804, 0.0239720, -0.0333486, 0.0294971
7: -0.0314879, 0.0111609, -0.0337955, 0.0187752, -0.0502631, 0.0449564
8: -0.0121138, 0.0322793, -0.0100217, 0.0401521, -0.0508791, 0.0422815
9: 0.8860436, 1.0078429, 0.8705507, 0.9979835, -0.1038443, 0.1270244

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718129, upper bound: 0.0781574
time: 1.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718145, upper bound: 0.0782515
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0138315, 0.0199521, -0.0139436, 0.0123793, -0.0262108, 0.0338957
1: -0.0060836, 0.0481609, -0.0039181, 0.0450282, -0.0511119, 0.0520790
2: 0.0093802, 0.0407030, 0.0087891, 0.0374261, -0.0280459, 0.0290434
3: -0.0115937, 0.0263188, -0.0117863, 0.0240710, -0.0356647, 0.0377625
4: -0.0331513, 0.0123093, -0.0243782, 0.0038566, -0.0342023, 0.0366356
5: -0.0117952, 0.0380069, -0.0083567, 0.0356048, -0.0474000, 0.0463636
6: -0.0091718, 0.0240976, -0.0096787, 0.0197634, -0.0287131, 0.0326417
7: -0.0335392, 0.0189189, -0.0310302, 0.0107769, -0.0443161, 0.0499491
8: -0.0095678, 0.0400491, -0.0118135, 0.0314438, -0.0401717, 0.0501497
9: 0.8698146, 0.9941387, 0.8858225, 1.0042065, -0.1236797, 0.0994247

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769893, upper bound: 0.0717707
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769893, upper bound: 0.0718145
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0138315, 0.0199521, -0.0154856, 0.0124396, -0.0262711, 0.0354377
1: -0.0060836, 0.0481609, -0.0048880, 0.0458204, -0.0519040, 0.0530489
2: 0.0093802, 0.0407030, 0.0058141, 0.0375266, -0.0281464, 0.0320711
3: -0.0115937, 0.0263188, -0.0122397, 0.0244096, -0.0360033, 0.0380564
4: -0.0331513, 0.0123093, -0.0245753, 0.0045755, -0.0347304, 0.0368577
5: -0.0117952, 0.0380069, -0.0088945, 0.0361202, -0.0479154, 0.0469014
6: -0.0091718, 0.0240976, -0.0101256, 0.0199167, -0.0288922, 0.0329322
7: -0.0335392, 0.0189189, -0.0314879, 0.0111609, -0.0447001, 0.0504068
8: -0.0095678, 0.0400491, -0.0121138, 0.0322793, -0.0411638, 0.0503330
9: 0.8698146, 0.9941387, 0.8860436, 1.0078429, -0.1261612, 0.0986193

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769893, upper bound: 0.0717707
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769893, upper bound: 0.0718145
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0139436, 0.0123793, -0.0275623, 0.0335966
1: -0.0064677, 0.0488046, -0.0039181, 0.0450282, -0.0514959, 0.0527227
2: 0.0071338, 0.0406640, 0.0087891, 0.0374261, -0.0302858, 0.0288538
3: -0.0119915, 0.0266010, -0.0117863, 0.0240710, -0.0360625, 0.0380780
4: -0.0330374, 0.0113035, -0.0243782, 0.0038566, -0.0337984, 0.0353116
5: -0.0117166, 0.0384160, -0.0083567, 0.0356048, -0.0473214, 0.0467727
6: -0.0095804, 0.0239720, -0.0096787, 0.0197634, -0.0290034, 0.0324810
7: -0.0337955, 0.0187752, -0.0310302, 0.0107769, -0.0445724, 0.0498054
8: -0.0100217, 0.0401521, -0.0118135, 0.0314438, -0.0405711, 0.0502168
9: 0.8705507, 0.9979835, 0.8858225, 1.0042065, -0.1223192, 0.1021972

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0738051
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0738261
time: 1.96 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0154856, 0.0124396, -0.0276226, 0.0351386
1: -0.0064677, 0.0488046, -0.0048880, 0.0458204, -0.0522881, 0.0536926
2: 0.0071338, 0.0406640, 0.0058141, 0.0375266, -0.0303692, 0.0310852
3: -0.0119915, 0.0266010, -0.0122397, 0.0244096, -0.0364011, 0.0388407
4: -0.0330374, 0.0113035, -0.0245753, 0.0045755, -0.0343269, 0.0358788
5: -0.0117166, 0.0384160, -0.0088945, 0.0361202, -0.0478368, 0.0473105
6: -0.0095804, 0.0239720, -0.0101256, 0.0199167, -0.0294971, 0.0333486
7: -0.0337955, 0.0187752, -0.0314879, 0.0111609, -0.0449564, 0.0502631
8: -0.0100217, 0.0401521, -0.0121138, 0.0322793, -0.0422815, 0.0508791
9: 0.8705507, 0.9979835, 0.8860436, 1.0078429, -0.1270244, 0.1038443

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0743186
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0743666
time: 6.31 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0138315, 0.0199521, -0.0148821, 0.0186677, -0.0324992, 0.0348343
1: -0.0060836, 0.0481609, -0.0049909, 0.0486864, -0.0547700, 0.0531518
2: 0.0093802, 0.0407030, 0.0077396, 0.0403376, -0.0309574, 0.0320914
3: -0.0115937, 0.0263188, -0.0111455, 0.0265501, -0.0381438, 0.0364908
4: -0.0331513, 0.0123093, -0.0325084, 0.0090023, -0.0409945, 0.0443241
5: -0.0117952, 0.0380069, -0.0100843, 0.0383398, -0.0501350, 0.0480912
6: -0.0091718, 0.0240976, -0.0089372, 0.0235570, -0.0316495, 0.0309035
7: -0.0335392, 0.0189189, -0.0335871, 0.0171782, -0.0507174, 0.0525060
8: -0.0095678, 0.0400491, -0.0093197, 0.0392154, -0.0456369, 0.0447044
9: 0.8698146, 0.9941387, 0.8712049, 0.9958602, -0.1045687, 0.1050497

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0712800, upper bound: 0.0650183
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712800, upper bound: 0.0755612
time: 3.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0138315, 0.0199521, -0.0351352, 0.0334846
1: -0.0064677, 0.0488046, -0.0060836, 0.0481609, -0.0546286, 0.0548882
2: 0.0071338, 0.0406640, 0.0093802, 0.0407030, -0.0335621, 0.0312838
3: -0.0119915, 0.0266010, -0.0115937, 0.0263188, -0.0381520, 0.0381947
4: -0.0330374, 0.0113035, -0.0331513, 0.0123093, -0.0448469, 0.0435475
5: -0.0117166, 0.0384160, -0.0117952, 0.0380069, -0.0497235, 0.0502112
6: -0.0095804, 0.0239720, -0.0091718, 0.0240976, -0.0321446, 0.0318918
7: -0.0337955, 0.0187752, -0.0335392, 0.0189189, -0.0527144, 0.0523144
8: -0.0100217, 0.0401521, -0.0095678, 0.0400491, -0.0465030, 0.0464035
9: 0.8705507, 0.9979835, 0.8698146, 0.9941387, -0.1051885, 0.1088801

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0699300, upper bound: 0.0683378
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0782589, upper bound: 0.0769191
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0151830, 0.0196531, -0.0348361, 0.0348361
1: -0.0064677, 0.0488046, -0.0064677, 0.0488046, -0.0552723, 0.0552723
2: 0.0071338, 0.0406640, 0.0071338, 0.0406640, -0.0334733, 0.0334733
3: -0.0119915, 0.0266010, -0.0119915, 0.0266010, -0.0385925, 0.0385925
4: -0.0330374, 0.0113035, -0.0330374, 0.0113035, -0.0435077, 0.0435077
5: -0.0117166, 0.0384160, -0.0117166, 0.0384160, -0.0501326, 0.0501326
6: -0.0095804, 0.0239720, -0.0095804, 0.0239720, -0.0327019, 0.0327019
7: -0.0337955, 0.0187752, -0.0337955, 0.0187752, -0.0525707, 0.0525707
8: -0.0100217, 0.0401521, -0.0100217, 0.0401521, -0.0471553, 0.0471553
9: 0.8705507, 0.9979835, 0.8705507, 0.9979835, -0.1097984, 0.1097984

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0699300, upper bound: 0.0701337
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0782589, upper bound: 0.0769920
time: 2.44 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0141481, 0.0148794, -0.0303649, 0.0265878
1: -0.0048880, 0.0458204, -0.0067327, 0.0467287, -0.0516166, 0.0525530
2: 0.0058141, 0.0375266, 0.0082062, 0.0392772, -0.0334631, 0.0259384
3: -0.0122397, 0.0244096, -0.0139275, 0.0251292, -0.0373690, 0.0375229
4: -0.0245753, 0.0045755, -0.0265133, 0.0058491, -0.0272503, 0.0275839
5: -0.0088945, 0.0361202, -0.0109288, 0.0368935, -0.0457880, 0.0470490
6: -0.0101256, 0.0199167, -0.0115884, 0.0208009, -0.0309265, 0.0302879
7: -0.0314879, 0.0111609, -0.0322522, 0.0140456, -0.0455334, 0.0434131
8: -0.0121138, 0.0322793, -0.0138035, 0.0332257, -0.0453395, 0.0448281
9: 0.8860436, 1.0078429, 0.8805467, 1.0114024, -0.1113910, 0.1272962

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0764532
time: 3.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0765235
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0156687, 0.0150452, -0.0305308, 0.0281083
1: -0.0048880, 0.0458204, -0.0078102, 0.0474758, -0.0523637, 0.0536306
2: 0.0058141, 0.0375266, 0.0052127, 0.0394592, -0.0336451, 0.0277954
3: -0.0122397, 0.0244096, -0.0145044, 0.0254460, -0.0376857, 0.0386756
4: -0.0245753, 0.0045755, -0.0268684, 0.0066083, -0.0277883, 0.0278292
5: -0.0088945, 0.0361202, -0.0115846, 0.0373834, -0.0462779, 0.0477048
6: -0.0101256, 0.0199167, -0.0121317, 0.0209602, -0.0310858, 0.0313245
7: -0.0314879, 0.0111609, -0.0326990, 0.0146040, -0.0460918, 0.0438599
8: -0.0121138, 0.0322793, -0.0142110, 0.0340559, -0.0461697, 0.0457478
9: 0.8860436, 1.0078429, 0.8806884, 1.0153494, -0.1162680, 0.1271545

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0767883
time: 2.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0768347
time: 1.43 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0139436, 0.0123793, -0.0141690, 0.0217598, -0.0357033, 0.0265484
1: -0.0039181, 0.0450282, -0.0095564, 0.0499182, -0.0538362, 0.0545846
2: 0.0087891, 0.0374261, 0.0080325, 0.0416972, -0.0329081, 0.0293936
3: -0.0117863, 0.0240710, -0.0138287, 0.0274024, -0.0391888, 0.0378997
4: -0.0243782, 0.0038566, -0.0351308, 0.0167084, -0.0410866, 0.0361002
5: -0.0083567, 0.0356048, -0.0152888, 0.0393312, -0.0476879, 0.0508936
6: -0.0096787, 0.0197634, -0.0110571, 0.0257505, -0.0354292, 0.0303469
7: -0.0310302, 0.0107769, -0.0349826, 0.0218801, -0.0529103, 0.0457596
8: -0.0118135, 0.0314438, -0.0116964, 0.0428943, -0.0547078, 0.0422035
9: 0.8858225, 1.0042065, 0.8642952, 1.0012407, -0.1056110, 0.1399114

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0784426
time: 1.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0786287
time: 2.40 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0139436, 0.0123793, -0.0154837, 0.0215660, -0.0355096, 0.0278630
1: -0.0039181, 0.0450282, -0.0100882, 0.0505141, -0.0544322, 0.0551164
2: 0.0087891, 0.0374261, 0.0063352, 0.0417820, -0.0329928, 0.0310909
3: -0.0117863, 0.0240710, -0.0143435, 0.0276428, -0.0394291, 0.0383791
4: -0.0243782, 0.0038566, -0.0350397, 0.0158828, -0.0401608, 0.0358772
5: -0.0083567, 0.0356048, -0.0153792, 0.0396903, -0.0480470, 0.0509841
6: -0.0096787, 0.0197634, -0.0115589, 0.0257092, -0.0353880, 0.0307392
7: -0.0310302, 0.0107769, -0.0352443, 0.0218823, -0.0529125, 0.0460212
8: -0.0118135, 0.0314438, -0.0122451, 0.0431442, -0.0549576, 0.0427132
9: 0.8858225, 1.0042065, 0.8647262, 1.0055966, -0.1088670, 0.1394804

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0784426
time: 1.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0786287
time: 1.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0141690, 0.0217598, -0.0372453, 0.0266087
1: -0.0048880, 0.0458204, -0.0095564, 0.0499182, -0.0548061, 0.0553768
2: 0.0058141, 0.0375266, 0.0080325, 0.0416972, -0.0358831, 0.0294941
3: -0.0122397, 0.0244096, -0.0138287, 0.0274024, -0.0396422, 0.0382383
4: -0.0245753, 0.0045755, -0.0351308, 0.0167084, -0.0412837, 0.0366338
5: -0.0088945, 0.0361202, -0.0152888, 0.0393312, -0.0482257, 0.0514090
6: -0.0101256, 0.0199167, -0.0110571, 0.0257505, -0.0358761, 0.0305265
7: -0.0314879, 0.0111609, -0.0349826, 0.0218801, -0.0533680, 0.0461435
8: -0.0121138, 0.0322793, -0.0116964, 0.0428943, -0.0550081, 0.0432068
9: 0.8860436, 1.0078429, 0.8642952, 1.0012407, -0.1048056, 0.1435477

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0799671
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0800923
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0154837, 0.0215660, -0.0370516, 0.0279233
1: -0.0048880, 0.0458204, -0.0100882, 0.0505141, -0.0554021, 0.0559086
2: 0.0058141, 0.0375266, 0.0063352, 0.0417820, -0.0359678, 0.0311914
3: -0.0122397, 0.0244096, -0.0143435, 0.0276428, -0.0398825, 0.0387531
4: -0.0245753, 0.0045755, -0.0350397, 0.0158828, -0.0404581, 0.0362930
5: -0.0088945, 0.0361202, -0.0153792, 0.0396903, -0.0485848, 0.0514994
6: -0.0101256, 0.0199167, -0.0115589, 0.0257092, -0.0358349, 0.0314756
7: -0.0314879, 0.0111609, -0.0352443, 0.0218823, -0.0533702, 0.0464052
8: -0.0121138, 0.0322793, -0.0122451, 0.0431442, -0.0552580, 0.0443419
9: 0.8860436, 1.0078429, 0.8647262, 1.0055966, -0.1102854, 0.1431167

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0801536
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0802516
time: 3.24 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0138315, 0.0199521, -0.0141481, 0.0148794, -0.0287109, 0.0341003
1: -0.0060836, 0.0481609, -0.0067327, 0.0467287, -0.0528123, 0.0548936
2: 0.0093802, 0.0407030, 0.0082062, 0.0392772, -0.0298970, 0.0297753
3: -0.0115937, 0.0263188, -0.0139275, 0.0251292, -0.0367229, 0.0399312
4: -0.0331513, 0.0123093, -0.0265133, 0.0058491, -0.0383894, 0.0386562
5: -0.0117952, 0.0380069, -0.0109288, 0.0368935, -0.0486887, 0.0489357
6: -0.0091718, 0.0240976, -0.0115884, 0.0208009, -0.0299727, 0.0345309
7: -0.0335392, 0.0189189, -0.0322522, 0.0140456, -0.0475848, 0.0511711
8: -0.0095678, 0.0400491, -0.0138035, 0.0332257, -0.0427935, 0.0521396
9: 0.8698146, 0.9941387, 0.8805467, 1.0114024, -0.1313760, 0.1135920

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769893, upper bound: 0.0740855
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769893, upper bound: 0.0741300
time: 1.52 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0138315, 0.0199521, -0.0156687, 0.0150452, -0.0288768, 0.0356208
1: -0.0060836, 0.0481609, -0.0078102, 0.0474758, -0.0535594, 0.0559711
2: 0.0093802, 0.0407030, 0.0052127, 0.0394592, -0.0300790, 0.0328145
3: -0.0115937, 0.0263188, -0.0145044, 0.0254460, -0.0370397, 0.0403855
4: -0.0331513, 0.0123093, -0.0268684, 0.0066083, -0.0390936, 0.0390445
5: -0.0117952, 0.0380069, -0.0115846, 0.0373834, -0.0491786, 0.0495915
6: -0.0091718, 0.0240976, -0.0121317, 0.0209602, -0.0301320, 0.0349637
7: -0.0335392, 0.0189189, -0.0326990, 0.0146040, -0.0481432, 0.0516179
8: -0.0095678, 0.0400491, -0.0142110, 0.0340559, -0.0436238, 0.0524657
9: 0.8698146, 0.9941387, 0.8806884, 1.0153494, -0.1349044, 0.1134503

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769893, upper bound: 0.0740855
time: 1.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769893, upper bound: 0.0741300
time: 1.83 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0141481, 0.0148794, -0.0300624, 0.0338012
1: -0.0064677, 0.0488046, -0.0067327, 0.0467287, -0.0531964, 0.0555373
2: 0.0071338, 0.0406640, 0.0082062, 0.0392772, -0.0321434, 0.0295857
3: -0.0119915, 0.0266010, -0.0139275, 0.0251292, -0.0371207, 0.0402468
4: -0.0330374, 0.0113035, -0.0265133, 0.0058491, -0.0377641, 0.0373334
5: -0.0117166, 0.0384160, -0.0109288, 0.0368935, -0.0486101, 0.0493448
6: -0.0095804, 0.0239720, -0.0115884, 0.0208009, -0.0303813, 0.0343712
7: -0.0337955, 0.0187752, -0.0322522, 0.0140456, -0.0478410, 0.0510274
8: -0.0100217, 0.0401521, -0.0138035, 0.0332257, -0.0432474, 0.0522184
9: 0.8705507, 0.9979835, 0.8805467, 1.0114024, -0.1300155, 0.1174368

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0764532
time: 1.23 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0765235
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0156687, 0.0150452, -0.0302283, 0.0353218
1: -0.0064677, 0.0488046, -0.0078102, 0.0474758, -0.0539434, 0.0566148
2: 0.0071338, 0.0406640, 0.0052127, 0.0394592, -0.0323254, 0.0318046
3: -0.0119915, 0.0266010, -0.0145044, 0.0254460, -0.0374375, 0.0411054
4: -0.0330374, 0.0113035, -0.0268684, 0.0066083, -0.0382733, 0.0379555
5: -0.0117166, 0.0384160, -0.0115846, 0.0373834, -0.0491000, 0.0500005
6: -0.0095804, 0.0239720, -0.0121317, 0.0209602, -0.0305406, 0.0352916
7: -0.0337955, 0.0187752, -0.0326990, 0.0146040, -0.0483995, 0.0514742
8: -0.0100217, 0.0401521, -0.0142110, 0.0340559, -0.0440776, 0.0529330
9: 0.8705507, 0.9979835, 0.8806884, 1.0153494, -0.1347280, 0.1172951

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0767883
time: 2.39 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0768347
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0138315, 0.0199521, -0.0151952, 0.0203747, -0.0342062, 0.0351474
1: -0.0060836, 0.0481609, -0.0085423, 0.0504056, -0.0564892, 0.0567033
2: 0.0093802, 0.0407030, 0.0071340, 0.0412000, -0.0318199, 0.0335690
3: -0.0115937, 0.0263188, -0.0134584, 0.0276074, -0.0392011, 0.0387090
4: -0.0331513, 0.0123093, -0.0344956, 0.0133142, -0.0458266, 0.0464337
5: -0.0117952, 0.0380069, -0.0137118, 0.0396280, -0.0514232, 0.0517188
6: -0.0091718, 0.0240976, -0.0108728, 0.0252444, -0.0333585, 0.0327431
7: -0.0335392, 0.0189189, -0.0349601, 0.0202487, -0.0537879, 0.0538790
8: -0.0095678, 0.0400491, -0.0115102, 0.0421132, -0.0486315, 0.0471600
9: 0.8698146, 0.9941387, 0.8655274, 1.0033303, -0.1133868, 0.1116522

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0712723, upper bound: 0.0669437
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712723, upper bound: 0.0776548
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0141690, 0.0217598, -0.0369428, 0.0338221
1: -0.0064677, 0.0488046, -0.0095564, 0.0499182, -0.0563858, 0.0583610
2: 0.0071338, 0.0406640, 0.0080325, 0.0416972, -0.0345634, 0.0326315
3: -0.0119915, 0.0266010, -0.0138287, 0.0274024, -0.0393940, 0.0404297
4: -0.0330374, 0.0113035, -0.0351308, 0.0167084, -0.0497458, 0.0455627
5: -0.0117166, 0.0384160, -0.0152888, 0.0393312, -0.0510478, 0.0537047
6: -0.0095804, 0.0239720, -0.0110571, 0.0257505, -0.0353309, 0.0336940
7: -0.0337955, 0.0187752, -0.0349826, 0.0218801, -0.0556756, 0.0537578
8: -0.0100217, 0.0401521, -0.0116964, 0.0428943, -0.0529160, 0.0487289
9: 0.8705507, 0.9979835, 0.8642952, 1.0012407, -0.1132868, 0.1336883

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0699299, upper bound: 0.0708603
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0782589, upper bound: 0.0793314
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0154837, 0.0215660, -0.0367491, 0.0351368
1: -0.0064677, 0.0488046, -0.0100882, 0.0505141, -0.0569818, 0.0588928
2: 0.0071338, 0.0406640, 0.0063352, 0.0417820, -0.0346482, 0.0343288
3: -0.0119915, 0.0266010, -0.0143435, 0.0276428, -0.0396343, 0.0409445
4: -0.0330374, 0.0113035, -0.0350397, 0.0158828, -0.0489202, 0.0456088
5: -0.0117166, 0.0384160, -0.0153792, 0.0396903, -0.0514069, 0.0537952
6: -0.0095804, 0.0239720, -0.0115589, 0.0257092, -0.0352897, 0.0345265
7: -0.0337955, 0.0187752, -0.0352443, 0.0218823, -0.0556778, 0.0540195
8: -0.0100217, 0.0401521, -0.0122451, 0.0431442, -0.0531659, 0.0495095
9: 0.8705507, 0.9979835, 0.8647262, 1.0055966, -0.1178968, 0.1332573

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0699299, upper bound: 0.0725379
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0782589, upper bound: 0.0793530
time: 1.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0141481, 0.0148794, -0.0138315, 0.0199521, -0.0341003, 0.0287109
1: -0.0067327, 0.0467287, -0.0060836, 0.0481609, -0.0548936, 0.0528123
2: 0.0082062, 0.0392772, 0.0093802, 0.0407030, -0.0297753, 0.0298970
3: -0.0139275, 0.0251292, -0.0115937, 0.0263188, -0.0399312, 0.0367229
4: -0.0265133, 0.0058491, -0.0331513, 0.0123093, -0.0386562, 0.0383894
5: -0.0109288, 0.0368935, -0.0117952, 0.0380069, -0.0489357, 0.0486887
6: -0.0115884, 0.0208009, -0.0091718, 0.0240976, -0.0345309, 0.0299727
7: -0.0322522, 0.0140456, -0.0335392, 0.0189189, -0.0511711, 0.0475848
8: -0.0138035, 0.0332257, -0.0095678, 0.0400491, -0.0521396, 0.0427935
9: 0.8805467, 1.0114024, 0.8698146, 0.9941387, -0.1135920, 0.1313760

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741370, upper bound: 0.0768202
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741597, upper bound: 0.0769588
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0141481, 0.0148794, -0.0151830, 0.0196531, -0.0338012, 0.0300624
1: -0.0067327, 0.0467287, -0.0064677, 0.0488046, -0.0555373, 0.0531964
2: 0.0082062, 0.0392772, 0.0071338, 0.0406640, -0.0295857, 0.0321434
3: -0.0139275, 0.0251292, -0.0119915, 0.0266010, -0.0402468, 0.0371207
4: -0.0265133, 0.0058491, -0.0330374, 0.0113035, -0.0373334, 0.0377640
5: -0.0109288, 0.0368935, -0.0117166, 0.0384160, -0.0493448, 0.0486101
6: -0.0115884, 0.0208009, -0.0095804, 0.0239720, -0.0343712, 0.0303813
7: -0.0322522, 0.0140456, -0.0337955, 0.0187752, -0.0510274, 0.0478410
8: -0.0138035, 0.0332257, -0.0100217, 0.0401521, -0.0522184, 0.0432474
9: 0.8805467, 1.0114024, 0.8705507, 0.9979835, -0.1174368, 0.1300156

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741370, upper bound: 0.0768202
time: 1.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741597, upper bound: 0.0769588
time: 2.07 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0156687, 0.0150452, -0.0138315, 0.0199521, -0.0356208, 0.0288768
1: -0.0078102, 0.0474758, -0.0060836, 0.0481609, -0.0559711, 0.0535594
2: 0.0052127, 0.0394592, 0.0093802, 0.0407030, -0.0328146, 0.0300790
3: -0.0145044, 0.0254460, -0.0115937, 0.0263188, -0.0403855, 0.0370397
4: -0.0268684, 0.0066083, -0.0331513, 0.0123093, -0.0390445, 0.0390936
5: -0.0115846, 0.0373834, -0.0117952, 0.0380069, -0.0495915, 0.0491786
6: -0.0121317, 0.0209602, -0.0091718, 0.0240976, -0.0349637, 0.0301320
7: -0.0326990, 0.0146040, -0.0335392, 0.0189189, -0.0516179, 0.0481432
8: -0.0142110, 0.0340559, -0.0095678, 0.0400491, -0.0524657, 0.0436238
9: 0.8806884, 1.0153494, 0.8698146, 0.9941387, -0.1134503, 0.1349045

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740254, upper bound: 0.0777678
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0744122, upper bound: 0.0782117
time: 3.28 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0156687, 0.0150452, -0.0151830, 0.0196531, -0.0353218, 0.0302283
1: -0.0078102, 0.0474758, -0.0064677, 0.0488046, -0.0566148, 0.0539434
2: 0.0052127, 0.0394592, 0.0071338, 0.0406640, -0.0318046, 0.0323254
3: -0.0145044, 0.0254460, -0.0119915, 0.0266010, -0.0411054, 0.0374375
4: -0.0268684, 0.0066083, -0.0330374, 0.0113035, -0.0379556, 0.0382733
5: -0.0115846, 0.0373834, -0.0117166, 0.0384160, -0.0500005, 0.0491000
6: -0.0121317, 0.0209602, -0.0095804, 0.0239720, -0.0352916, 0.0305406
7: -0.0326990, 0.0146040, -0.0337955, 0.0187752, -0.0514742, 0.0483995
8: -0.0142110, 0.0340559, -0.0100217, 0.0401521, -0.0529330, 0.0440776
9: 0.8806884, 1.0153494, 0.8705507, 0.9979835, -0.1172951, 0.1347280

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740254, upper bound: 0.0780300
time: 1.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0744122, upper bound: 0.0784344
time: 1.17 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0141690, 0.0217598, -0.0139436, 0.0123793, -0.0265484, 0.0357033
1: -0.0095564, 0.0499182, -0.0039181, 0.0450282, -0.0545846, 0.0538362
2: 0.0080325, 0.0416972, 0.0087891, 0.0374261, -0.0293936, 0.0329081
3: -0.0138287, 0.0274024, -0.0117863, 0.0240710, -0.0378997, 0.0391888
4: -0.0351308, 0.0167084, -0.0243782, 0.0038566, -0.0361002, 0.0410866
5: -0.0152888, 0.0393312, -0.0083567, 0.0356048, -0.0508936, 0.0476879
6: -0.0110571, 0.0257505, -0.0096787, 0.0197634, -0.0303469, 0.0354292
7: -0.0349826, 0.0218801, -0.0310302, 0.0107769, -0.0457596, 0.0529103
8: -0.0116964, 0.0428943, -0.0118135, 0.0314438, -0.0422035, 0.0547078
9: 0.8642952, 1.0012407, 0.8858225, 1.0042065, -0.1399114, 0.1056110

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787339, upper bound: 0.0717577
time: 2.03 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787339, upper bound: 0.0718058
time: 1.29 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0141690, 0.0217598, -0.0154856, 0.0124396, -0.0266087, 0.0372453
1: -0.0095564, 0.0499182, -0.0048880, 0.0458204, -0.0553768, 0.0548061
2: 0.0080325, 0.0416972, 0.0058141, 0.0375266, -0.0294941, 0.0358831
3: -0.0138287, 0.0274024, -0.0122397, 0.0244096, -0.0382383, 0.0396422
4: -0.0351308, 0.0167084, -0.0245753, 0.0045755, -0.0366338, 0.0412837
5: -0.0152888, 0.0393312, -0.0088945, 0.0361202, -0.0514090, 0.0482257
6: -0.0110571, 0.0257505, -0.0101256, 0.0199167, -0.0305265, 0.0358761
7: -0.0349826, 0.0218801, -0.0314879, 0.0111609, -0.0461435, 0.0533680
8: -0.0116964, 0.0428943, -0.0121138, 0.0322793, -0.0432068, 0.0550081
9: 0.8642952, 1.0012407, 0.8860436, 1.0078429, -0.1435477, 0.1048056

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787339, upper bound: 0.0717577
time: 1.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787339, upper bound: 0.0718058
time: 1.73 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0139436, 0.0123793, -0.0278630, 0.0355096
1: -0.0100882, 0.0505141, -0.0039181, 0.0450282, -0.0551164, 0.0544322
2: 0.0063352, 0.0417820, 0.0087891, 0.0374261, -0.0310909, 0.0329928
3: -0.0143435, 0.0276428, -0.0117863, 0.0240710, -0.0383791, 0.0394291
4: -0.0350397, 0.0158828, -0.0243782, 0.0038566, -0.0358772, 0.0401609
5: -0.0153792, 0.0396903, -0.0083567, 0.0356048, -0.0509841, 0.0480470
6: -0.0115589, 0.0257092, -0.0096787, 0.0197634, -0.0307393, 0.0353880
7: -0.0352443, 0.0218823, -0.0310302, 0.0107769, -0.0460212, 0.0529125
8: -0.0122451, 0.0431442, -0.0118135, 0.0314438, -0.0427132, 0.0549576
9: 0.8647262, 1.0055966, 0.8858225, 1.0042065, -0.1394804, 0.1088670

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786287, upper bound: 0.0736873
time: 2.57 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786287, upper bound: 0.0737159
time: 1.73 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0154856, 0.0124396, -0.0279233, 0.0370516
1: -0.0100882, 0.0505141, -0.0048880, 0.0458204, -0.0559086, 0.0554021
2: 0.0063352, 0.0417820, 0.0058141, 0.0375266, -0.0311914, 0.0359678
3: -0.0143435, 0.0276428, -0.0122397, 0.0244096, -0.0387531, 0.0398825
4: -0.0350397, 0.0158828, -0.0245753, 0.0045755, -0.0362930, 0.0404581
5: -0.0153792, 0.0396903, -0.0088945, 0.0361202, -0.0514994, 0.0485848
6: -0.0115589, 0.0257092, -0.0101256, 0.0199167, -0.0314756, 0.0358349
7: -0.0352443, 0.0218823, -0.0314879, 0.0111609, -0.0464052, 0.0533702
8: -0.0122451, 0.0431442, -0.0121138, 0.0322793, -0.0443419, 0.0552580
9: 0.8647262, 1.0055966, 0.8860436, 1.0078429, -0.1431167, 0.1102854

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786287, upper bound: 0.0741421
time: 1.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786287, upper bound: 0.0741796
time: 1.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0141690, 0.0217598, -0.0148821, 0.0186677, -0.0328367, 0.0366419
1: -0.0095564, 0.0499182, -0.0049909, 0.0486864, -0.0582427, 0.0549090
2: 0.0080325, 0.0416972, 0.0077396, 0.0403376, -0.0323051, 0.0339576
3: -0.0138287, 0.0274024, -0.0111455, 0.0265501, -0.0403787, 0.0385480
4: -0.0351308, 0.0167084, -0.0325084, 0.0090023, -0.0429948, 0.0492167
5: -0.0152888, 0.0393312, -0.0100843, 0.0383398, -0.0536286, 0.0494155
6: -0.0110571, 0.0257505, -0.0089372, 0.0235570, -0.0334513, 0.0346877
7: -0.0349826, 0.0218801, -0.0335871, 0.0171782, -0.0521609, 0.0554672
8: -0.0116964, 0.0428943, -0.0093197, 0.0392154, -0.0479688, 0.0522141
9: 0.8642952, 1.0012407, 0.8712049, 0.9958602, -0.1315650, 0.1131479

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0739165, upper bound: 0.0650183
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0739165, upper bound: 0.0754985
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0138315, 0.0199521, -0.0354358, 0.0353976
1: -0.0100882, 0.0505141, -0.0060836, 0.0481609, -0.0582491, 0.0565977
2: 0.0063352, 0.0417820, 0.0093802, 0.0407030, -0.0343678, 0.0324018
3: -0.0143435, 0.0276428, -0.0115937, 0.0263188, -0.0404616, 0.0392365
4: -0.0350397, 0.0158828, -0.0331513, 0.0123093, -0.0470157, 0.0490340
5: -0.0153792, 0.0396903, -0.0117952, 0.0380069, -0.0533861, 0.0514855
6: -0.0115589, 0.0257092, -0.0091718, 0.0240976, -0.0340693, 0.0348811
7: -0.0352443, 0.0218823, -0.0335392, 0.0189189, -0.0541632, 0.0554215
8: -0.0122451, 0.0431442, -0.0095678, 0.0400491, -0.0489507, 0.0527120
9: 0.8647262, 1.0055966, 0.8698146, 0.9941387, -0.1294125, 0.1180622

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0719107, upper bound: 0.0683034
time: 1.92 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0800270, upper bound: 0.0767191
time: 2.42 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0151830, 0.0196531, -0.0351368, 0.0367491
1: -0.0100882, 0.0505141, -0.0064677, 0.0488046, -0.0588928, 0.0569818
2: 0.0063352, 0.0417820, 0.0071338, 0.0406640, -0.0343288, 0.0346482
3: -0.0143435, 0.0276428, -0.0119915, 0.0266010, -0.0409445, 0.0396343
4: -0.0350397, 0.0158828, -0.0330374, 0.0113035, -0.0456088, 0.0489202
5: -0.0153792, 0.0396903, -0.0117166, 0.0384160, -0.0537952, 0.0514069
6: -0.0115589, 0.0257092, -0.0095804, 0.0239720, -0.0345265, 0.0352897
7: -0.0352443, 0.0218823, -0.0337955, 0.0187752, -0.0540195, 0.0556778
8: -0.0122451, 0.0431442, -0.0100217, 0.0401521, -0.0495095, 0.0531659
9: 0.8647262, 1.0055966, 0.8705507, 0.9979835, -0.1332573, 0.1178969

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0719107, upper bound: 0.0700408
time: 1.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0800270, upper bound: 0.0767832
time: 2.42 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0141481, 0.0148794, -0.0141690, 0.0217598, -0.0359079, 0.0290484
1: -0.0067327, 0.0467287, -0.0095564, 0.0499182, -0.0566508, 0.0562851
2: 0.0082062, 0.0392772, 0.0080325, 0.0416972, -0.0334910, 0.0312447
3: -0.0139275, 0.0251292, -0.0138287, 0.0274024, -0.0412634, 0.0389579
4: -0.0265133, 0.0058491, -0.0351308, 0.0167084, -0.0429270, 0.0391440
5: -0.0109288, 0.0368935, -0.0152888, 0.0393312, -0.0502600, 0.0521823
6: -0.0115884, 0.0208009, -0.0110571, 0.0257505, -0.0373389, 0.0318579
7: -0.0322522, 0.0140456, -0.0349826, 0.0218801, -0.0541323, 0.0490282
8: -0.0138035, 0.0332257, -0.0116964, 0.0428943, -0.0566979, 0.0449221
9: 0.8805467, 1.0114024, 0.8642952, 1.0012407, -0.1206940, 0.1471072

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741370, upper bound: 0.0768336
time: 1.29 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741597, upper bound: 0.0769736
time: 1.98 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0141481, 0.0148794, -0.0154837, 0.0215660, -0.0357142, 0.0303631
1: -0.0067327, 0.0467287, -0.0100882, 0.0505141, -0.0572468, 0.0568169
2: 0.0082062, 0.0392772, 0.0063352, 0.0417820, -0.0335758, 0.0329420
3: -0.0139275, 0.0251292, -0.0143435, 0.0276428, -0.0415399, 0.0394727
4: -0.0265133, 0.0058491, -0.0350397, 0.0158828, -0.0417706, 0.0386126
5: -0.0109288, 0.0368935, -0.0153792, 0.0396903, -0.0506191, 0.0522727
6: -0.0115884, 0.0208009, -0.0115589, 0.0257092, -0.0372976, 0.0323598
7: -0.0322522, 0.0140456, -0.0352443, 0.0218823, -0.0541345, 0.0492899
8: -0.0138035, 0.0332257, -0.0122451, 0.0431442, -0.0569477, 0.0454708
9: 0.8805467, 1.0114024, 0.8647262, 1.0055966, -0.1250499, 0.1466762

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741370, upper bound: 0.0768336
time: 1.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741597, upper bound: 0.0769736
time: 1.81 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0156687, 0.0150452, -0.0141690, 0.0217598, -0.0374285, 0.0292143
1: -0.0078102, 0.0474758, -0.0095564, 0.0499182, -0.0577283, 0.0570321
2: 0.0052127, 0.0394592, 0.0080325, 0.0416972, -0.0364846, 0.0314267
3: -0.0145044, 0.0254460, -0.0138287, 0.0274024, -0.0416305, 0.0392747
4: -0.0268684, 0.0066083, -0.0351308, 0.0167084, -0.0430908, 0.0396773
5: -0.0115846, 0.0373834, -0.0152888, 0.0393312, -0.0509158, 0.0526722
6: -0.0121317, 0.0209602, -0.0110571, 0.0257505, -0.0378822, 0.0320172
7: -0.0326990, 0.0146040, -0.0349826, 0.0218801, -0.0545791, 0.0495866
8: -0.0142110, 0.0340559, -0.0116964, 0.0428943, -0.0571053, 0.0457524
9: 0.8806884, 1.0153494, 0.8642952, 1.0012407, -0.1205523, 0.1510542

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740254, upper bound: 0.0777841
time: 1.98 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0744122, upper bound: 0.0782120
time: 1.80 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.47 + 597.53 = 602.00 seconds
