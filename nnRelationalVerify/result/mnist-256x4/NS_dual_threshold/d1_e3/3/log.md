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
execution time: IAR + RelationalAnalysis = 1.91 + 3.32 = 5.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0882511, upper bound: 0.0882511

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0835640, upper bound: 0.0858858
time: 2.22 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0858858, upper bound: 0.0858858
time: 2.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.68 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.68
Output dim: 9, lower bound: -0.0835640, upper bound: 0.0858858
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.68
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

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798544, upper bound: 0.0783961
time: 2.75 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0808305, upper bound: 0.0829000
time: 1.30 seconds

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

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0818174
time: 2.30 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0829000, upper bound: 0.0829000
time: 1.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.89 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.89
Output dim: 9, lower bound: -0.0798544, upper bound: 0.0783961
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.89
Output dim: 9, lower bound: -0.0808305, upper bound: 0.0829000
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.89
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0818174
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.89
Output dim: 9, lower bound: -0.0829000, upper bound: 0.0829000

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0153956, 0.0182535, -0.0166719, 0.0163575, -0.0317531, 0.0349254
1: -0.0052599, 0.0484950, -0.0088362, 0.0489225, -0.0541824, 0.0573312
2: 0.0068169, 0.0400608, 0.0043608, 0.0404071, -0.0335902, 0.0328959
3: -0.0116096, 0.0263214, -0.0152160, 0.0262915, -0.0379011, 0.0413614
4: -0.0313829, 0.0076923, -0.0279937, 0.0071305, -0.0365793, 0.0356860
5: -0.0100342, 0.0381541, -0.0123947, 0.0384107, -0.0484449, 0.0505487
6: -0.0093593, 0.0228200, -0.0127103, 0.0215172, -0.0308765, 0.0345588
7: -0.0333524, 0.0171090, -0.0335522, 0.0162359, -0.0495883, 0.0506611
8: -0.0100123, 0.0379489, -0.0148000, 0.0354604, -0.0454728, 0.0512286
9: 0.8740575, 0.9988016, 0.8785234, 1.0173712, -0.1315566, 0.1202782

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798544, upper bound: 0.0759387
time: 2.02 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798544, upper bound: 0.0783961
time: 1.99 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0156531, 0.0210691, -0.0162450, 0.0229761, -0.0386292, 0.0373141
1: -0.0082205, 0.0496574, -0.0118083, 0.0516453, -0.0598658, 0.0614657
2: 0.0062132, 0.0412265, 0.0050779, 0.0426678, -0.0364546, 0.0361486
3: -0.0128219, 0.0271098, -0.0152207, 0.0283099, -0.0411317, 0.0423305
4: -0.0343439, 0.0149274, -0.0363260, 0.0192838, -0.0536276, 0.0512533
5: -0.0137598, 0.0390376, -0.0173273, 0.0405147, -0.0542746, 0.0563649
6: -0.0101697, 0.0250146, -0.0121994, 0.0267556, -0.0369253, 0.0369829
7: -0.0345220, 0.0207985, -0.0360643, 0.0238449, -0.0583669, 0.0568627
8: -0.0105467, 0.0421653, -0.0128688, 0.0452179, -0.0557646, 0.0538058
9: 0.8679127, 0.9999187, 0.8618411, 1.0077012, -0.1321408, 0.1380776

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0808305, upper bound: 0.0808439
time: 1.94 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0808305, upper bound: 0.0829000
time: 1.30 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.0160097, 0.0154769, -0.0161503, 0.0215541, -0.0375638, 0.0316271
1: -0.0083188, 0.0477993, -0.0103956, 0.0515605, -0.0598793, 0.0581949
2: 0.0047449, 0.0397154, 0.0058240, 0.0420741, -0.0373292, 0.0338914
3: -0.0148234, 0.0256052, -0.0146734, 0.0282148, -0.0430382, 0.0402786
4: -0.0270774, 0.0069391, -0.0350504, 0.0154262, -0.0421092, 0.0406009
5: -0.0119602, 0.0376129, -0.0155527, 0.0404389, -0.0523992, 0.0531655
6: -0.0123997, 0.0210322, -0.0118379, 0.0258831, -0.0382828, 0.0328701
7: -0.0329190, 0.0151685, -0.0357737, 0.0220866, -0.0550056, 0.0509422
8: -0.0145032, 0.0343645, -0.0126569, 0.0433180, -0.0578212, 0.0470214
9: 0.8804663, 1.0167246, 0.8637484, 1.0072277, -0.1267613, 0.1529762

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
time: 2.35 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
time: 2.25 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0158550, 0.0220735, -0.0164571, 0.0244779, -0.0403329, 0.0385306
1: -0.0108087, 0.0508285, -0.0133277, 0.0527647, -0.0635734, 0.0641562
2: 0.0057412, 0.0420804, 0.0033087, 0.0434461, -0.0377049, 0.0387718
3: -0.0147648, 0.0277965, -0.0158656, 0.0290181, -0.0437829, 0.0436621
4: -0.0352937, 0.0169752, -0.0381023, 0.0231755, -0.0584691, 0.0550775
5: -0.0161317, 0.0399140, -0.0191603, 0.0413453, -0.0574769, 0.0590743
6: -0.0118801, 0.0259476, -0.0126340, 0.0281719, -0.0400519, 0.0385815
7: -0.0354761, 0.0226024, -0.0371033, 0.0257391, -0.0612152, 0.0597057
8: -0.0125955, 0.0436554, -0.0132304, 0.0477483, -0.0603438, 0.0568858
9: 0.8642685, 1.0071003, 0.8575721, 1.0084556, -0.1441871, 0.1495283

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818174, upper bound: 0.0783961
time: 1.41 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818174, upper bound: 0.0829000
time: 3.47 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.75 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 6.75
Output dim: 9, lower bound: -0.0798544, upper bound: 0.0759387
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 6.75
Output dim: 9, lower bound: -0.0798544, upper bound: 0.0783961
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 6.75
Output dim: 9, lower bound: -0.0808305, upper bound: 0.0808439
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 6.75
Output dim: 9, lower bound: -0.0808305, upper bound: 0.0829000
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 6.75
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 6.75
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 6.75
Output dim: 9, lower bound: -0.0818174, upper bound: 0.0783961
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 6.75
Output dim: 9, lower bound: -0.0818174, upper bound: 0.0829000

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0153956, 0.0182535, -0.0159469, 0.0131682, -0.0285638, 0.0342004
1: -0.0052599, 0.0484950, -0.0055100, 0.0462397, -0.0514996, 0.0540050
2: 0.0068169, 0.0400608, 0.0052813, 0.0381136, -0.0312967, 0.0315931
3: -0.0116096, 0.0263214, -0.0126436, 0.0246212, -0.0362308, 0.0386657
4: -0.0313829, 0.0076923, -0.0248543, 0.0050471, -0.0342144, 0.0325466
5: -0.0100342, 0.0381541, -0.0093701, 0.0364139, -0.0464481, 0.0475242
6: -0.0093593, 0.0228200, -0.0104635, 0.0200278, -0.0293871, 0.0322474
7: -0.0333524, 0.0171090, -0.0317995, 0.0118925, -0.0452449, 0.0489085
8: -0.0100123, 0.0379489, -0.0124777, 0.0327108, -0.0427232, 0.0488906
9: 0.8740575, 0.9988016, 0.8857105, 1.0093753, -0.1225193, 0.1130912

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0759387
time: 2.07 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0759387
time: 2.21 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0153956, 0.0182535, -0.0163676, 0.0154843, -0.0308798, 0.0346211
1: -0.0052599, 0.0484950, -0.0083663, 0.0480520, -0.0533119, 0.0568613
2: 0.0068169, 0.0400608, 0.0045636, 0.0398777, -0.0330607, 0.0325194
3: -0.0116096, 0.0263214, -0.0148432, 0.0257480, -0.0373576, 0.0409074
4: -0.0313829, 0.0076923, -0.0271137, 0.0069482, -0.0363840, 0.0348060
5: -0.0100342, 0.0381541, -0.0119910, 0.0377836, -0.0478178, 0.0501450
6: -0.0093593, 0.0228200, -0.0124105, 0.0210896, -0.0304489, 0.0341859
7: -0.0333524, 0.0171090, -0.0330474, 0.0151759, -0.0485283, 0.0501564
8: -0.0100123, 0.0379489, -0.0145305, 0.0345505, -0.0445629, 0.0509446
9: 0.8740575, 0.9988016, 0.8803930, 1.0167825, -0.1309250, 0.1184086

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0783961
time: 1.27 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0783961
time: 1.68 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0156531, 0.0210691, -0.0155506, 0.0200678, -0.0357209, 0.0366197
1: -0.0082205, 0.0496574, -0.0071896, 0.0491037, -0.0573243, 0.0568470
2: 0.0062132, 0.0412265, 0.0065629, 0.0408218, -0.0346085, 0.0346636
3: -0.0128219, 0.0271098, -0.0124121, 0.0267453, -0.0395672, 0.0395219
4: -0.0343439, 0.0149274, -0.0333054, 0.0123178, -0.0466616, 0.0482328
5: -0.0137598, 0.0390376, -0.0124739, 0.0386236, -0.0523835, 0.0515115
6: -0.0101697, 0.0250146, -0.0099007, 0.0241867, -0.0336319, 0.0348822
7: -0.0345220, 0.0207985, -0.0340049, 0.0195085, -0.0540305, 0.0548034
8: -0.0105467, 0.0421653, -0.0103662, 0.0406451, -0.0482888, 0.0515112
9: 0.8679127, 0.9999187, 0.8701790, 0.9995501, -0.1249174, 0.1118884

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0798641
time: 1.35 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0783805
time: 1.76 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0156531, 0.0210691, -0.0158550, 0.0220735, -0.0377266, 0.0369241
1: -0.0082205, 0.0496574, -0.0108087, 0.0508285, -0.0590490, 0.0604661
2: 0.0062132, 0.0412265, 0.0057412, 0.0420804, -0.0358672, 0.0354853
3: -0.0128219, 0.0271098, -0.0147648, 0.0277965, -0.0406184, 0.0418746
4: -0.0343439, 0.0149274, -0.0352937, 0.0169752, -0.0513190, 0.0502211
5: -0.0137598, 0.0390376, -0.0161317, 0.0399140, -0.0536738, 0.0551693
6: -0.0101697, 0.0250146, -0.0118801, 0.0259476, -0.0361173, 0.0365969
7: -0.0345220, 0.0207985, -0.0354761, 0.0226024, -0.0571244, 0.0562746
8: -0.0105467, 0.0421653, -0.0125955, 0.0436554, -0.0542021, 0.0535138
9: 0.8679127, 0.9999187, 0.8642685, 1.0071003, -0.1314623, 0.1356502

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0818173
time: 1.30 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0807065
time: 2.13 seconds

## BFS NS instance: NS_A2_A1_B1

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

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0757888
time: 2.34 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
time: 2.38 seconds

## BFS NS instance: NS_A2_A1_B2

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

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0757973
time: 4.60 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
time: 2.33 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0158550, 0.0220735, -0.0167321, 0.0168462, -0.0327012, 0.0388056
1: -0.0108087, 0.0508285, -0.0090890, 0.0494619, -0.0602706, 0.0599174
2: 0.0057412, 0.0420804, 0.0042905, 0.0407388, -0.0349976, 0.0377899
3: -0.0147648, 0.0277965, -0.0154283, 0.0266160, -0.0413808, 0.0432248
4: -0.0352937, 0.0169752, -0.0286287, 0.0072378, -0.0403047, 0.0456039
5: -0.0161317, 0.0399140, -0.0126154, 0.0388045, -0.0549362, 0.0525294
6: -0.0118801, 0.0259476, -0.0128745, 0.0218298, -0.0337099, 0.0388220
7: -0.0354761, 0.0226024, -0.0339117, 0.0168093, -0.0522854, 0.0565141
8: -0.0125955, 0.0436554, -0.0149618, 0.0359587, -0.0485542, 0.0586172
9: 0.8642685, 1.0071003, 0.8769688, 1.0177102, -0.1534417, 0.1301315

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818173, upper bound: 0.0757888
time: 1.28 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818173, upper bound: 0.0757973
time: 1.55 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0158550, 0.0220735, -0.0163317, 0.0233948, -0.0392498, 0.0384053
1: -0.0108087, 0.0508285, -0.0122635, 0.0521728, -0.0629815, 0.0630919
2: 0.0057412, 0.0420804, 0.0047498, 0.0429479, -0.0372067, 0.0373307
3: -0.0147648, 0.0277965, -0.0154531, 0.0286376, -0.0434024, 0.0432496
4: -0.0352937, 0.0169752, -0.0369140, 0.0202625, -0.0555562, 0.0538892
5: -0.0161317, 0.0399140, -0.0178620, 0.0409051, -0.0570367, 0.0577760
6: -0.0118801, 0.0259476, -0.0123623, 0.0272728, -0.0391529, 0.0383098
7: -0.0354761, 0.0226024, -0.0365085, 0.0243890, -0.0598650, 0.0591109
8: -0.0125955, 0.0436554, -0.0130396, 0.0460678, -0.0586632, 0.0566950
9: 0.8642685, 1.0071003, 0.8600098, 1.0080578, -0.1437893, 0.1470906

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818173, upper bound: 0.0781786
time: 1.86 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0818174, upper bound: 0.0781786
time: 1.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.54 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0759387
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0759387
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0783961
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0783961
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0798641
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0783805
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0818173
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0757888, upper bound: 0.0807065
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0757888
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0757973
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0783961, upper bound: 0.0798544
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0818173, upper bound: 0.0757888
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0818173, upper bound: 0.0757973
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0818173, upper bound: 0.0781786
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 5.54
Output dim: 9, lower bound: -0.0818174, upper bound: 0.0781786

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0158237, 0.0127893, -0.0159469, 0.0131682, -0.0289919, 0.0287362
1: -0.0054148, 0.0461315, -0.0055100, 0.0462397, -0.0516545, 0.0516415
2: 0.0053458, 0.0376629, 0.0052813, 0.0381136, -0.0327678, 0.0289669
3: -0.0125702, 0.0245662, -0.0126436, 0.0246212, -0.0371914, 0.0367372
4: -0.0247824, 0.0048562, -0.0248543, 0.0050471, -0.0264538, 0.0297104
5: -0.0092804, 0.0363421, -0.0093701, 0.0364139, -0.0456943, 0.0457122
6: -0.0104036, 0.0199739, -0.0104635, 0.0200278, -0.0304314, 0.0296466
7: -0.0316892, 0.0117341, -0.0317995, 0.0118925, -0.0435817, 0.0435336
8: -0.0124057, 0.0326010, -0.0124777, 0.0327108, -0.0451165, 0.0441931
9: 0.8859093, 1.0092524, 0.8857105, 1.0093753, -0.1095457, 0.1235420

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0725829
time: 1.24 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0751586
time: 1.32 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0155506, 0.0200678, -0.0159469, 0.0131682, -0.0287188, 0.0360148
1: -0.0071896, 0.0491037, -0.0055100, 0.0462397, -0.0534293, 0.0546137
2: 0.0065629, 0.0408218, 0.0052813, 0.0381136, -0.0315508, 0.0326348
3: -0.0124121, 0.0267453, -0.0126436, 0.0246212, -0.0370333, 0.0393889
4: -0.0333054, 0.0123178, -0.0248543, 0.0050471, -0.0370550, 0.0371720
5: -0.0124739, 0.0386236, -0.0093701, 0.0364139, -0.0488878, 0.0479937
6: -0.0099007, 0.0241867, -0.0104635, 0.0200278, -0.0299285, 0.0337857
7: -0.0340049, 0.0195085, -0.0317995, 0.0118925, -0.0458974, 0.0513081
8: -0.0103662, 0.0406451, -0.0124777, 0.0327108, -0.0430771, 0.0516495
9: 0.8701790, 0.9995501, 0.8857105, 1.0093753, -0.1281698, 0.1138396

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0725829
time: 1.93 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0751586
time: 1.26 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0158237, 0.0127893, -0.0163676, 0.0154843, -0.0313079, 0.0291569
1: -0.0054148, 0.0461315, -0.0083663, 0.0480520, -0.0534668, 0.0544977
2: 0.0053458, 0.0376629, 0.0045636, 0.0398777, -0.0345318, 0.0298932
3: -0.0125702, 0.0245662, -0.0148432, 0.0257480, -0.0383182, 0.0389789
4: -0.0247824, 0.0048562, -0.0271137, 0.0069482, -0.0286234, 0.0319699
5: -0.0092804, 0.0363421, -0.0119910, 0.0377836, -0.0470640, 0.0483330
6: -0.0104036, 0.0199739, -0.0124105, 0.0210896, -0.0314932, 0.0315851
7: -0.0316892, 0.0117341, -0.0330474, 0.0151759, -0.0468651, 0.0447815
8: -0.0124057, 0.0326010, -0.0145305, 0.0345505, -0.0469562, 0.0462471
9: 0.8859093, 1.0092524, 0.8803930, 1.0167825, -0.1179513, 0.1288594

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0749920
time: 1.88 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0776603
time: 1.34 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0155506, 0.0200678, -0.0163676, 0.0154843, -0.0310349, 0.0364354
1: -0.0071896, 0.0491037, -0.0083663, 0.0480520, -0.0552416, 0.0574700
2: 0.0065629, 0.0408218, 0.0045636, 0.0398777, -0.0333148, 0.0335611
3: -0.0124121, 0.0267453, -0.0148432, 0.0257480, -0.0381601, 0.0415885
4: -0.0333054, 0.0123178, -0.0271137, 0.0069482, -0.0392246, 0.0394315
5: -0.0124739, 0.0386236, -0.0119910, 0.0377836, -0.0502574, 0.0506146
6: -0.0099007, 0.0241867, -0.0124105, 0.0210896, -0.0309903, 0.0357243
7: -0.0340049, 0.0195085, -0.0330474, 0.0151759, -0.0491808, 0.0525559
8: -0.0103662, 0.0406451, -0.0145305, 0.0345505, -0.0449168, 0.0537034
9: 0.8701790, 0.9995501, 0.8803930, 1.0167825, -0.1365754, 0.1191571

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0749920
time: 2.04 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0776603
time: 2.29 seconds

## BFS NS instance: NS_A1_B2_B1_A1

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

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0787060
time: 1.04 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0789311
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_B1_A2

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

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0761589
time: 1.41 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0775904
time: 1.67 seconds

## BFS NS instance: NS_A1_B2_B2_A1

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

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0807830
time: 1.05 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0809698
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_B2_A2

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

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0782554
time: 1.67 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0799577
time: 2.49 seconds

## BFS NS instance: NS_A2_A1_B1_B1

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

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A1_B1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0745462
time: 1.07 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0749830
time: 1.11 seconds

## BFS NS instance: NS_A2_A1_B1_B2

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
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A1_B1_B2_B1

### Relational analysis result of NS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0786935
time: 1.33 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2

### Relational analysis result of NS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0789210
time: 1.11 seconds

## BFS NS instance: NS_A2_A1_B2_B1

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

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0727799
time: 1.24 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0749906
time: 1.88 seconds

## BFS NS instance: NS_A2_A1_B2_B2

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

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A1_B2_B2_B1

### Relational analysis result of NS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0786936
time: 1.05 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0789210
time: 1.97 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0158550, 0.0220735, -0.0159469, 0.0131682, -0.0290232, 0.0380205
1: -0.0108087, 0.0508285, -0.0055100, 0.0462397, -0.0570484, 0.0563385
2: 0.0057412, 0.0420804, 0.0052813, 0.0381136, -0.0323725, 0.0367991
3: -0.0147648, 0.0277965, -0.0126436, 0.0246212, -0.0393860, 0.0404401
4: -0.0352937, 0.0169752, -0.0248543, 0.0050471, -0.0392700, 0.0418294
5: -0.0161317, 0.0399140, -0.0093701, 0.0364139, -0.0525456, 0.0492841
6: -0.0118801, 0.0259476, -0.0104635, 0.0200278, -0.0319079, 0.0364111
7: -0.0354761, 0.0226024, -0.0317995, 0.0118925, -0.0473685, 0.0544019
8: -0.0125955, 0.0436554, -0.0124777, 0.0327108, -0.0453063, 0.0561331
9: 0.8642685, 1.0071003, 0.8857105, 1.0093753, -0.1451068, 0.1213899

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0725649
time: 1.13 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0749830
time: 1.12 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0158550, 0.0220735, -0.0163676, 0.0154843, -0.0313393, 0.0384411
1: -0.0108087, 0.0508285, -0.0083663, 0.0480520, -0.0588606, 0.0591947
2: 0.0057412, 0.0420804, 0.0045636, 0.0398777, -0.0341365, 0.0375168
3: -0.0147648, 0.0277965, -0.0148432, 0.0257480, -0.0405128, 0.0426397
4: -0.0352937, 0.0169752, -0.0271137, 0.0069482, -0.0399423, 0.0440889
5: -0.0161317, 0.0399140, -0.0119910, 0.0377836, -0.0539152, 0.0519049
6: -0.0118801, 0.0259476, -0.0124105, 0.0210896, -0.0329697, 0.0383580
7: -0.0354761, 0.0226024, -0.0330474, 0.0151759, -0.0506520, 0.0556498
8: -0.0125955, 0.0436554, -0.0145305, 0.0345505, -0.0471460, 0.0581859
9: 0.8642685, 1.0071003, 0.8803930, 1.0167825, -0.1525140, 0.1267073

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0727799
time: 1.91 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0749906
time: 1.14 seconds

## BFS NS instance: NS_A2_A2_B2_B1

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

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787919, upper bound: 0.0741688
time: 1.81 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0785477, upper bound: 0.0740987
time: 1.16 seconds

## BFS NS instance: NS_A2_A2_B2_B2

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

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0820202, upper bound: 0.0760958
time: 1.26 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0820773, upper bound: 0.0773740
time: 1.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.54 seconds
NS_A1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0725829
NS_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0751586
NS_A1_B1_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0725829
NS_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0751586
NS_A1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0749920
NS_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0776603
NS_A1_B1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0749920
NS_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0776603
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0787060
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0789311
NS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0761589
NS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0751586, upper bound: 0.0775904
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0807830
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0809698
NS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0782554
NS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0749830, upper bound: 0.0799577
NS_A2_A1_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0745462
NS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0749830
NS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0786935
NS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0789210
NS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0727799
NS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0749906
NS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0786936
NS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0776603, upper bound: 0.0789210
NS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0725649
NS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0749830
NS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0807830, upper bound: 0.0727799
NS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0809698, upper bound: 0.0749906
NS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0787919, upper bound: 0.0741688
NS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0785477, upper bound: 0.0740987
NS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0820202, upper bound: 0.0760958
NS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.54
Output dim: 9, lower bound: -0.0820773, upper bound: 0.0773740

## BFS NS instance: NS_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0159469, 0.0131682, -0.0286538, 0.0283865
1: -0.0048880, 0.0458204, -0.0055100, 0.0462397, -0.0511276, 0.0513304
2: 0.0058141, 0.0375266, 0.0052813, 0.0381136, -0.0322995, 0.0288230
3: -0.0122397, 0.0244096, -0.0126436, 0.0246212, -0.0368609, 0.0363779
4: -0.0245753, 0.0045755, -0.0248543, 0.0050471, -0.0262358, 0.0294298
5: -0.0088945, 0.0361202, -0.0093701, 0.0364139, -0.0453084, 0.0454903
6: -0.0101256, 0.0199167, -0.0104635, 0.0200278, -0.0301534, 0.0293423
7: -0.0314879, 0.0111609, -0.0317995, 0.0118925, -0.0433803, 0.0429605
8: -0.0121138, 0.0322793, -0.0124777, 0.0327108, -0.0448246, 0.0436908
9: 0.8860436, 1.0078429, 0.8857105, 1.0093753, -0.1084352, 0.1221324

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0746708
time: 1.08 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0751586
time: 1.60 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0159469, 0.0131682, -0.0283512, 0.0356000
1: -0.0064677, 0.0488046, -0.0055100, 0.0462397, -0.0527074, 0.0543146
2: 0.0071338, 0.0406640, 0.0052813, 0.0381136, -0.0309798, 0.0324703
3: -0.0119915, 0.0266010, -0.0126436, 0.0246212, -0.0366127, 0.0391019
4: -0.0330374, 0.0113035, -0.0248543, 0.0050471, -0.0367495, 0.0361578
5: -0.0117166, 0.0384160, -0.0093701, 0.0364139, -0.0481305, 0.0477861
6: -0.0095804, 0.0239720, -0.0104635, 0.0200278, -0.0296082, 0.0334255
7: -0.0337955, 0.0187752, -0.0317995, 0.0118925, -0.0456879, 0.0505747
8: -0.0100217, 0.0401521, -0.0124777, 0.0327108, -0.0427325, 0.0510811
9: 0.8705507, 0.9979835, 0.8857105, 1.0093753, -0.1270598, 0.1122730

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0746708
time: 1.52 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0751586
time: 1.62 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0163676, 0.0154843, -0.0309698, 0.0288072
1: -0.0048880, 0.0458204, -0.0083663, 0.0480520, -0.0529399, 0.0541866
2: 0.0058141, 0.0375266, 0.0045636, 0.0398777, -0.0340635, 0.0297493
3: -0.0122397, 0.0244096, -0.0148432, 0.0257480, -0.0379877, 0.0386196
4: -0.0245753, 0.0045755, -0.0271137, 0.0069482, -0.0284054, 0.0316892
5: -0.0088945, 0.0361202, -0.0119910, 0.0377836, -0.0466781, 0.0481111
6: -0.0101256, 0.0199167, -0.0124105, 0.0210896, -0.0312152, 0.0312808
7: -0.0314879, 0.0111609, -0.0330474, 0.0151759, -0.0466638, 0.0442083
8: -0.0121138, 0.0322793, -0.0145305, 0.0345505, -0.0466643, 0.0457447
9: 0.8860436, 1.0078429, 0.8803930, 1.0167825, -0.1168409, 0.1274499

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0773646
time: 1.31 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0776603
time: 1.71 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0163676, 0.0154843, -0.0306673, 0.0360207
1: -0.0064677, 0.0488046, -0.0083663, 0.0480520, -0.0545196, 0.0571709
2: 0.0071338, 0.0406640, 0.0045636, 0.0398777, -0.0327439, 0.0333966
3: -0.0119915, 0.0266010, -0.0148432, 0.0257480, -0.0377395, 0.0413435
4: -0.0330374, 0.0113035, -0.0271137, 0.0069482, -0.0389191, 0.0384172
5: -0.0117166, 0.0384160, -0.0119910, 0.0377836, -0.0495002, 0.0504069
6: -0.0095804, 0.0239720, -0.0124105, 0.0210896, -0.0306700, 0.0353640
7: -0.0337955, 0.0187752, -0.0330474, 0.0151759, -0.0489714, 0.0518226
8: -0.0100217, 0.0401521, -0.0145305, 0.0345505, -0.0445722, 0.0531351
9: 0.8705507, 0.9979835, 0.8803930, 1.0167825, -0.1354654, 0.1175905

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0683661, upper bound: 0.0598053
time: 1.75 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0782696, upper bound: 0.0769293
time: 2.39 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0151920, 0.0124382, -0.0138315, 0.0199521, -0.0351441, 0.0262698
1: -0.0047071, 0.0457318, -0.0060836, 0.0481609, -0.0528680, 0.0518155
2: 0.0065071, 0.0374966, 0.0093802, 0.0407030, -0.0313615, 0.0281165
3: -0.0121456, 0.0243858, -0.0115937, 0.0263188, -0.0381350, 0.0359795
4: -0.0245049, 0.0044634, -0.0331513, 0.0123093, -0.0367731, 0.0347209
5: -0.0087802, 0.0360743, -0.0117952, 0.0380069, -0.0467871, 0.0478695
6: -0.0100419, 0.0198866, -0.0091718, 0.0240976, -0.0330206, 0.0288499
7: -0.0314487, 0.0110592, -0.0335392, 0.0189189, -0.0503676, 0.0445984
8: -0.0120315, 0.0321661, -0.0095678, 0.0400491, -0.0504136, 0.0409416
9: 0.8861243, 1.0072381, 0.8698146, 0.9941387, -0.0987768, 0.1261080

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0717707, upper bound: 0.0780380
time: 1.21 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718145, upper bound: 0.0780380
time: 1.80 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0158237, 0.0127893, -0.0151830, 0.0196531, -0.0354768, 0.0279723
1: -0.0054148, 0.0461315, -0.0064677, 0.0488046, -0.0542194, 0.0525991
2: 0.0053458, 0.0376629, 0.0071338, 0.0406640, -0.0323654, 0.0305149
3: -0.0125702, 0.0245662, -0.0119915, 0.0266010, -0.0390242, 0.0365577
4: -0.0247824, 0.0048562, -0.0330374, 0.0113035, -0.0360859, 0.0347601
5: -0.0092804, 0.0363421, -0.0117166, 0.0384160, -0.0476964, 0.0480587
6: -0.0104036, 0.0199739, -0.0095804, 0.0239720, -0.0333560, 0.0295544
7: -0.0316892, 0.0117341, -0.0337955, 0.0187752, -0.0504644, 0.0455295
8: -0.0124057, 0.0326010, -0.0100217, 0.0401521, -0.0510033, 0.0426227
9: 0.8859093, 1.0092524, 0.8705507, 0.9979835, -0.1046925, 0.1269325

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0776223
time: 1.44 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0789311
time: 2.29 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1

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

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0790500, upper bound: 0.0756562
time: 1.36 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0793271, upper bound: 0.0756586
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2

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

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0775261
time: 2.39 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0775904
time: 2.12 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0151920, 0.0124382, -0.0141690, 0.0217598, -0.0369518, 0.0266073
1: -0.0047071, 0.0457318, -0.0095564, 0.0499182, -0.0546253, 0.0552882
2: 0.0065071, 0.0374966, 0.0080325, 0.0416972, -0.0351901, 0.0294641
3: -0.0121456, 0.0243858, -0.0138287, 0.0274024, -0.0395481, 0.0382145
4: -0.0245049, 0.0044634, -0.0351308, 0.0167084, -0.0412133, 0.0366142
5: -0.0087802, 0.0360743, -0.0152888, 0.0393312, -0.0481114, 0.0513631
6: -0.0100419, 0.0198866, -0.0110571, 0.0257505, -0.0357924, 0.0304838
7: -0.0314487, 0.0110592, -0.0349826, 0.0218801, -0.0533288, 0.0460418
8: -0.0120315, 0.0321661, -0.0116964, 0.0428943, -0.0549259, 0.0429832
9: 0.8861243, 1.0072381, 0.8642952, 1.0012407, -0.1049631, 0.1429430

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0799671
time: 1.29 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0800923
time: 2.26 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0158237, 0.0127893, -0.0154837, 0.0215660, -0.0373897, 0.0282730
1: -0.0054148, 0.0461315, -0.0100882, 0.0505141, -0.0559289, 0.0562196
2: 0.0053458, 0.0376629, 0.0063352, 0.0417820, -0.0364361, 0.0313277
3: -0.0125702, 0.0245662, -0.0143435, 0.0276428, -0.0402130, 0.0389097
4: -0.0247824, 0.0048562, -0.0350397, 0.0158828, -0.0406652, 0.0368444
5: -0.0092804, 0.0363421, -0.0153792, 0.0396903, -0.0489707, 0.0517213
6: -0.0104036, 0.0199739, -0.0115589, 0.0257092, -0.0361129, 0.0315329
7: -0.0316892, 0.0117341, -0.0352443, 0.0218823, -0.0535715, 0.0469784
8: -0.0124057, 0.0326010, -0.0122451, 0.0431442, -0.0555498, 0.0447887
9: 0.8859093, 1.0092524, 0.8647262, 1.0055966, -0.1111335, 0.1445262

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0792818
time: 2.11 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0809698
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1

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

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0790215, upper bound: 0.0777550
time: 2.06 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0793002, upper bound: 0.0777619
time: 1.65 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2

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

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0799408
time: 2.89 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0799577
time: 1.47 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0160097, 0.0154769, -0.0154856, 0.0124396, -0.0284493, 0.0309624
1: -0.0083188, 0.0477993, -0.0048880, 0.0458204, -0.0541392, 0.0526873
2: 0.0047449, 0.0397154, 0.0058141, 0.0375266, -0.0294593, 0.0339013
3: -0.0148234, 0.0256052, -0.0122397, 0.0244096, -0.0386107, 0.0378449
4: -0.0270774, 0.0069391, -0.0245753, 0.0045755, -0.0280288, 0.0283968
5: -0.0119602, 0.0376129, -0.0088945, 0.0361202, -0.0480804, 0.0465074
6: -0.0123997, 0.0210322, -0.0101256, 0.0199167, -0.0312754, 0.0311579
7: -0.0329190, 0.0151685, -0.0314879, 0.0111609, -0.0440799, 0.0466564
8: -0.0145032, 0.0343645, -0.0121138, 0.0322793, -0.0457296, 0.0464783
9: 0.8804663, 1.0167246, 0.8860436, 1.0078429, -0.1273766, 0.1167963

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_A1_B1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0725649
time: 1.37 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0749830
time: 1.80 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0153810, 0.0150026, -0.0138315, 0.0199521, -0.0353332, 0.0288342
1: -0.0076001, 0.0474064, -0.0060836, 0.0481609, -0.0557610, 0.0534900
2: 0.0059016, 0.0393924, 0.0093802, 0.0407030, -0.0321015, 0.0300123
3: -0.0143770, 0.0254280, -0.0115937, 0.0263188, -0.0403985, 0.0370217
4: -0.0267911, 0.0064648, -0.0331513, 0.0123093, -0.0389501, 0.0389573
5: -0.0114380, 0.0373461, -0.0117952, 0.0380069, -0.0494449, 0.0491413
6: -0.0120226, 0.0209235, -0.0091718, 0.0240976, -0.0349730, 0.0300954
7: -0.0326497, 0.0144500, -0.0335392, 0.0189189, -0.0515686, 0.0479892
8: -0.0141028, 0.0339358, -0.0095678, 0.0400491, -0.0524666, 0.0435037
9: 0.8807912, 1.0146401, 0.8698146, 0.9941387, -0.1133475, 0.1342411

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B1_B2_B1_B1

### Relational analysis result of NS_A2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740855, upper bound: 0.0780266
time: 1.93 seconds

## Relational analysis of NS_A2_A1_B1_B2_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741300, upper bound: 0.0780266
time: 1.30 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0160097, 0.0154769, -0.0151830, 0.0196531, -0.0356628, 0.0306599
1: -0.0083188, 0.0477993, -0.0064677, 0.0488046, -0.0571234, 0.0542670
2: 0.0047449, 0.0397154, 0.0071338, 0.0406640, -0.0331066, 0.0325816
3: -0.0148234, 0.0256052, -0.0119915, 0.0266010, -0.0413347, 0.0375967
4: -0.0270774, 0.0069391, -0.0330374, 0.0113035, -0.0381552, 0.0389105
5: -0.0119602, 0.0376129, -0.0117166, 0.0384160, -0.0503762, 0.0493295
6: -0.0123997, 0.0210322, -0.0095804, 0.0239720, -0.0353586, 0.0306126
7: -0.0329190, 0.0151685, -0.0337955, 0.0187752, -0.0516942, 0.0489640
8: -0.0145032, 0.0343645, -0.0100217, 0.0401521, -0.0531200, 0.0443862
9: 0.8804663, 1.0167246, 0.8705507, 0.9979835, -0.1175172, 0.1354209

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_A1_B1_B2_B2_B1

### Relational analysis result of NS_A2_A1_B1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0598053, upper bound: 0.0683661
time: 2.17 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2_B2

### Relational analysis result of NS_A2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769293, upper bound: 0.0782696
time: 1.36 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1

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

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0764532, upper bound: 0.0719711
time: 1.58 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0765236, upper bound: 0.0719868
time: 1.28 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2

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

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0746103
time: 1.41 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0749906
time: 2.01 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0153810, 0.0150026, -0.0141690, 0.0217598, -0.0371408, 0.0291717
1: -0.0076001, 0.0474064, -0.0095564, 0.0499182, -0.0575182, 0.0569628
2: 0.0059016, 0.0393924, 0.0080325, 0.0416972, -0.0357956, 0.0313599
3: -0.0143770, 0.0254280, -0.0138287, 0.0274024, -0.0417062, 0.0392567
4: -0.0267911, 0.0064648, -0.0351308, 0.0167084, -0.0430312, 0.0396573
5: -0.0114380, 0.0373461, -0.0152888, 0.0393312, -0.0507692, 0.0526349
6: -0.0120226, 0.0209235, -0.0110571, 0.0257505, -0.0377731, 0.0319806
7: -0.0326497, 0.0144500, -0.0349826, 0.0218801, -0.0545298, 0.0494326
8: -0.0141028, 0.0339358, -0.0116964, 0.0428943, -0.0569971, 0.0456323
9: 0.8807912, 1.0146401, 0.8642952, 1.0012407, -0.1204495, 0.1503449

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B2_B2_B1_B1

### Relational analysis result of NS_A2_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740855, upper bound: 0.0780268
time: 1.22 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_B2

### Relational analysis result of NS_A2_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741300, upper bound: 0.0780268
time: 1.37 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0160097, 0.0154769, -0.0154837, 0.0215660, -0.0375758, 0.0309606
1: -0.0083188, 0.0477993, -0.0100882, 0.0505141, -0.0588329, 0.0578875
2: 0.0047449, 0.0397154, 0.0063352, 0.0417820, -0.0370371, 0.0333803
3: -0.0148234, 0.0256052, -0.0143435, 0.0276428, -0.0424662, 0.0399487
4: -0.0270774, 0.0069391, -0.0350397, 0.0158828, -0.0424947, 0.0396415
5: -0.0119602, 0.0376129, -0.0153792, 0.0396903, -0.0516506, 0.0529921
6: -0.0123997, 0.0210322, -0.0115589, 0.0257092, -0.0381089, 0.0325912
7: -0.0329190, 0.0151685, -0.0352443, 0.0218823, -0.0548013, 0.0504128
8: -0.0145032, 0.0343645, -0.0122451, 0.0431442, -0.0576474, 0.0466096
9: 0.8804663, 1.0167246, 0.8647262, 1.0055966, -0.1251303, 0.1519984

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0776352
time: 1.11 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0789210
time: 1.27 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0141690, 0.0217598, -0.0152436, 0.0127086, -0.0268777, 0.0370034
1: -0.0095564, 0.0499182, -0.0047922, 0.0457769, -0.0553333, 0.0547103
2: 0.0080325, 0.0416972, 0.0064789, 0.0377695, -0.0297370, 0.0352183
3: -0.0138287, 0.0274024, -0.0122156, 0.0244078, -0.0382365, 0.0396180
4: -0.0351308, 0.0167084, -0.0245694, 0.0045863, -0.0389263, 0.0412778
5: -0.0152888, 0.0393312, -0.0088638, 0.0361037, -0.0513925, 0.0481950
6: -0.0110571, 0.0257505, -0.0101000, 0.0199117, -0.0309688, 0.0358505
7: -0.0349826, 0.0218801, -0.0315033, 0.0112134, -0.0461960, 0.0533834
8: -0.0116964, 0.0428943, -0.0120983, 0.0322275, -0.0439239, 0.0549927
9: 0.8642952, 1.0012407, 0.8860177, 1.0073483, -0.1430531, 0.1152231

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0799671, upper bound: 0.0718051
time: 1.22 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0800923, upper bound: 0.0718058
time: 1.96 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0159469, 0.0131682, -0.0286519, 0.0375130
1: -0.0100882, 0.0505141, -0.0055100, 0.0462397, -0.0563279, 0.0560241
2: 0.0063352, 0.0417820, 0.0052813, 0.0381136, -0.0317785, 0.0365006
3: -0.0143435, 0.0276428, -0.0126436, 0.0246212, -0.0389647, 0.0402864
4: -0.0350397, 0.0158828, -0.0248543, 0.0050471, -0.0389822, 0.0407370
5: -0.0153792, 0.0396903, -0.0093701, 0.0364139, -0.0517931, 0.0490605
6: -0.0115589, 0.0257092, -0.0104635, 0.0200278, -0.0315868, 0.0361727
7: -0.0352443, 0.0218823, -0.0317995, 0.0118925, -0.0471368, 0.0536818
8: -0.0122451, 0.0431442, -0.0124777, 0.0327108, -0.0449560, 0.0556219
9: 0.8647262, 1.0055966, 0.8857105, 1.0093753, -0.1446491, 0.1198862

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0745462
time: 1.13 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0749830
time: 1.85 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0141690, 0.0217598, -0.0156544, 0.0150084, -0.0291775, 0.0374141
1: -0.0095564, 0.0499182, -0.0076350, 0.0475764, -0.0571328, 0.0575531
2: 0.0080325, 0.0416972, 0.0057669, 0.0395221, -0.0314896, 0.0359303
3: -0.0138287, 0.0274024, -0.0143915, 0.0255232, -0.0393519, 0.0417097
4: -0.0351308, 0.0167084, -0.0268203, 0.0064705, -0.0396638, 0.0435287
5: -0.0152888, 0.0393312, -0.0114595, 0.0374644, -0.0527532, 0.0507906
6: -0.0110571, 0.0257505, -0.0120303, 0.0209623, -0.0320193, 0.0377808
7: -0.0349826, 0.0218801, -0.0327367, 0.0144559, -0.0494385, 0.0546168
8: -0.0116964, 0.0428943, -0.0141238, 0.0340632, -0.0457596, 0.0570181
9: 0.8642952, 1.0012407, 0.8807395, 1.0146818, -0.1503866, 0.1205013

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0800923, upper bound: 0.0719188
time: 1.21 seconds

## Relational analysis of NS_A2_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0800923, upper bound: 0.0719868
time: 1.33 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0163676, 0.0154843, -0.0309680, 0.0379336
1: -0.0100882, 0.0505141, -0.0083663, 0.0480520, -0.0581402, 0.0588804
2: 0.0063352, 0.0417820, 0.0045636, 0.0398777, -0.0335425, 0.0372183
3: -0.0143435, 0.0276428, -0.0148432, 0.0257480, -0.0400915, 0.0424859
4: -0.0350397, 0.0158828, -0.0271137, 0.0069482, -0.0396510, 0.0429965
5: -0.0153792, 0.0396903, -0.0119910, 0.0377836, -0.0531628, 0.0516813
6: -0.0115589, 0.0257092, -0.0124105, 0.0210896, -0.0326486, 0.0381197
7: -0.0352443, 0.0218823, -0.0330474, 0.0151759, -0.0504202, 0.0549297
8: -0.0122451, 0.0431442, -0.0145305, 0.0345505, -0.0467957, 0.0576746
9: 0.8647262, 1.0055966, 0.8803930, 1.0167825, -0.1520563, 0.1252036

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0746103
time: 1.16 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0749906
time: 5.33 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0157609, 0.0213709, -0.0155506, 0.0200678, -0.0358287, 0.0369215
1: -0.0100257, 0.0503879, -0.0071896, 0.0491037, -0.0591295, 0.0575775
2: 0.0060781, 0.0417080, 0.0065629, 0.0408218, -0.0347437, 0.0351452
3: -0.0144153, 0.0275206, -0.0124121, 0.0267453, -0.0411607, 0.0399327
4: -0.0345242, 0.0150525, -0.0333054, 0.0123178, -0.0465246, 0.0483579
5: -0.0151617, 0.0395811, -0.0124739, 0.0386236, -0.0537853, 0.0520549
6: -0.0116347, 0.0253156, -0.0099007, 0.0241867, -0.0347257, 0.0352163
7: -0.0350917, 0.0216800, -0.0340049, 0.0195085, -0.0546002, 0.0556850
8: -0.0123916, 0.0424625, -0.0103662, 0.0406451, -0.0501988, 0.0528287
9: 0.8661591, 1.0066195, 0.8701790, 0.9995501, -0.1333910, 0.1198421

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0785418, upper bound: 0.0740971
time: 1.44 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0785418, upper bound: 0.0740987
time: 1.24 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0163451, 0.0295548, -0.0154486, 0.0194505, -0.0357956, 0.0450034
1: -0.0215813, 0.0495355, -0.0064527, 0.0484881, -0.0700695, 0.0559883
2: 0.0034087, 0.0473252, 0.0067397, 0.0405275, -0.0371188, 0.0405855
3: -0.0230482, 0.0267738, -0.0120725, 0.0263578, -0.0494059, 0.0388463
4: -0.0351784, 0.0284616, -0.0324811, 0.0106687, -0.0458472, 0.0609427
5: -0.0267715, 0.0389229, -0.0115677, 0.0381610, -0.0649325, 0.0504906
6: -0.0182803, 0.0261947, -0.0096568, 0.0235381, -0.0409980, 0.0358516
7: -0.0363744, 0.0319596, -0.0335809, 0.0186050, -0.0549794, 0.0655405
8: -0.0212106, 0.0446487, -0.0101698, 0.0394330, -0.0584367, 0.0548185
9: 0.8669058, 1.0313890, 0.8723156, 0.9990882, -0.1321824, 0.1445197

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769612, upper bound: 0.0708081
time: 1.78 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0777153, upper bound: 0.0732833
time: 1.55 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1

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

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 188

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0773383, upper bound: 0.0716837
time: 1.60 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769612, upper bound: 0.0708081
time: 1.13 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2

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

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773186
time: 1.52 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773740
time: 1.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.99 seconds
NS_A1_B1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0746708
NS_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0725829, upper bound: 0.0751586
NS_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0746708
NS_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0776223, upper bound: 0.0751586
NS_A1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0773646
NS_A1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0725649, upper bound: 0.0776603
NS_A1_B1_B2_A2_A2_A1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0683661, upper bound: 0.0598053
NS_A1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0782696, upper bound: 0.0769293
NS_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0717707, upper bound: 0.0780380
NS_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0718145, upper bound: 0.0780380
NS_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0776223
NS_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0746708, upper bound: 0.0789311
NS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0790500, upper bound: 0.0756562
NS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0793271, upper bound: 0.0756586
NS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0775261
NS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0775904
NS_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0799671
NS_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0800923
NS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0792818
NS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0745462, upper bound: 0.0809698
NS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0790215, upper bound: 0.0777550
NS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0793002, upper bound: 0.0777619
NS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0799408
NS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0788494, upper bound: 0.0799577
NS_A2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0725649
NS_A2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0749830
NS_A2_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0740855, upper bound: 0.0780266
NS_A2_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0741300, upper bound: 0.0780266
NS_A2_A1_B1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0598053, upper bound: 0.0683661
NS_A2_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0769293, upper bound: 0.0782696
NS_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0764532, upper bound: 0.0719711
NS_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0765236, upper bound: 0.0719868
NS_A2_A1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0746103
NS_A2_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0749920, upper bound: 0.0749906
NS_A2_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0740855, upper bound: 0.0780268
NS_A2_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0741300, upper bound: 0.0780268
NS_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0776352
NS_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0773646, upper bound: 0.0789210
NS_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0799671, upper bound: 0.0718051
NS_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0800923, upper bound: 0.0718058
NS_A2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0745462
NS_A2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0792817, upper bound: 0.0749830
NS_A2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0800923, upper bound: 0.0719188
NS_A2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0800923, upper bound: 0.0719868
NS_A2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0746103
NS_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0792818, upper bound: 0.0749906
NS_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0785418, upper bound: 0.0740971
NS_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0785418, upper bound: 0.0740987
NS_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0769612, upper bound: 0.0708081
NS_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0777153, upper bound: 0.0732833
NS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0773383, upper bound: 0.0716837
NS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0769612, upper bound: 0.0708081
NS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773186
NS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 9, lower bound: -0.0806196, upper bound: 0.0773740

## BFS NS instance: NS_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0155654, 0.0127230, -0.0282086, 0.0280051
1: -0.0048880, 0.0458204, -0.0049765, 0.0458845, -0.0507724, 0.0507969
2: 0.0058141, 0.0375266, 0.0057728, 0.0378293, -0.0320152, 0.0271439
3: -0.0122397, 0.0244096, -0.0123109, 0.0244402, -0.0366799, 0.0364860
4: -0.0245753, 0.0045755, -0.0246377, 0.0047071, -0.0257641, 0.0292132
5: -0.0088945, 0.0361202, -0.0089802, 0.0361610, -0.0450555, 0.0451004
6: -0.0101256, 0.0199167, -0.0101846, 0.0199480, -0.0300736, 0.0294499
7: -0.0314879, 0.0111609, -0.0315558, 0.0113160, -0.0428038, 0.0427167
8: -0.0121138, 0.0322793, -0.0121822, 0.0323555, -0.0444693, 0.0437708
9: 0.8860436, 1.0078429, 0.8859212, 1.0079582, -0.1086865, 0.1219217

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0718129, upper bound: 0.0743186
time: 1.41 seconds

## Relational analysis of NS_A1_B1_B1_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0718145, upper bound: 0.0743666
time: 1.68 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0139452, 0.0127406, -0.0279236, 0.0335983
1: -0.0064677, 0.0488046, -0.0040294, 0.0450448, -0.0515125, 0.0528340
2: 0.0071338, 0.0406640, 0.0087840, 0.0377538, -0.0306200, 0.0288697
3: -0.0119915, 0.0266010, -0.0118819, 0.0240768, -0.0360683, 0.0381802
4: -0.0330374, 0.0113035, -0.0244524, 0.0040243, -0.0357768, 0.0352629
5: -0.0117166, 0.0384160, -0.0084703, 0.0356174, -0.0473340, 0.0468863
6: -0.0095804, 0.0239720, -0.0097584, 0.0197835, -0.0293639, 0.0325716
7: -0.0337955, 0.0187752, -0.0310786, 0.0109855, -0.0447810, 0.0498538
8: -0.0100217, 0.0401521, -0.0119033, 0.0314945, -0.0415162, 0.0503229
9: 0.8705507, 0.9979835, 0.8857085, 1.0043519, -0.1224786, 0.1122751

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0738051
time: 1.30 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0738261
time: 3.95 seconds

## BFS NS instance: NS_A1_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0151830, 0.0196531, -0.0155654, 0.0127230, -0.0279061, 0.0352185
1: -0.0064677, 0.0488046, -0.0049765, 0.0458845, -0.0523522, 0.0537812
2: 0.0071338, 0.0406640, 0.0057728, 0.0378293, -0.0306955, 0.0311531
3: -0.0119915, 0.0266010, -0.0123109, 0.0244402, -0.0364317, 0.0389119
4: -0.0330374, 0.0113035, -0.0246377, 0.0047071, -0.0362491, 0.0359412
5: -0.0117166, 0.0384160, -0.0089802, 0.0361610, -0.0478776, 0.0473962
6: -0.0095804, 0.0239720, -0.0101846, 0.0199480, -0.0295284, 0.0334170
7: -0.0337955, 0.0187752, -0.0315558, 0.0113160, -0.0451115, 0.0503310
8: -0.0100217, 0.0401521, -0.0121822, 0.0323555, -0.0423772, 0.0509560
9: 0.8705507, 0.9979835, 0.8859212, 1.0079582, -0.1271465, 0.1120623

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0743186
time: 1.88 seconds

## Relational analysis of NS_A1_B1_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769588, upper bound: 0.0743666
time: 1.80 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0142899, 0.0148828, -0.0303683, 0.0267295
1: -0.0048880, 0.0458204, -0.0067487, 0.0468021, -0.0516900, 0.0525691
2: 0.0058141, 0.0375266, 0.0081384, 0.0393506, -0.0335365, 0.0260467
3: -0.0122397, 0.0244096, -0.0139338, 0.0251683, -0.0374081, 0.0375262
4: -0.0245753, 0.0045755, -0.0265285, 0.0058516, -0.0272531, 0.0311040
5: -0.0088945, 0.0361202, -0.0109380, 0.0369454, -0.0458399, 0.0470582
6: -0.0101256, 0.0199167, -0.0115919, 0.0208164, -0.0309420, 0.0302898
7: -0.0314879, 0.0111609, -0.0322887, 0.0140482, -0.0455361, 0.0434496
8: -0.0121138, 0.0322793, -0.0138130, 0.0332794, -0.0453932, 0.0448337
9: 0.8860436, 1.0078429, 0.8805223, 1.0114224, -0.1114059, 0.1273206

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0764532
time: 3.45 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0765235
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154856, 0.0124396, -0.0159866, 0.0150518, -0.0305374, 0.0284262
1: -0.0048880, 0.0458204, -0.0078510, 0.0476693, -0.0525573, 0.0536714
2: 0.0058141, 0.0375266, 0.0050576, 0.0396033, -0.0337891, 0.0280401
3: -0.0122397, 0.0244096, -0.0145207, 0.0255524, -0.0377921, 0.0386826
4: -0.0245753, 0.0045755, -0.0268997, 0.0066158, -0.0277955, 0.0314752
5: -0.0088945, 0.0361202, -0.0116096, 0.0375108, -0.0464054, 0.0477298
6: -0.0101256, 0.0199167, -0.0121406, 0.0210017, -0.0311273, 0.0313286
7: -0.0314879, 0.0111609, -0.0327927, 0.0146106, -0.0460985, 0.0439536
8: -0.0121138, 0.0322793, -0.0142346, 0.0341964, -0.0463102, 0.0457595
9: 0.8860436, 1.0078429, 0.8806283, 1.0153996, -0.1163003, 0.1272146

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B1_B2_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0767883
time: 2.14 seconds

## Relational analysis of NS_A1_B1_B2_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0768347
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0151272, 0.0186846, -0.0163676, 0.0154843, -0.0306115, 0.0350522
1: -0.0051587, 0.0487503, -0.0083663, 0.0480520, -0.0532106, 0.0571165
2: 0.0072333, 0.0403685, 0.0045636, 0.0398777, -0.0326443, 0.0329413
3: -0.0112556, 0.0265731, -0.0148432, 0.0257480, -0.0370036, 0.0413142
4: -0.0325846, 0.0090885, -0.0271137, 0.0069482, -0.0382348, 0.0362022
5: -0.0102055, 0.0383755, -0.0119910, 0.0377836, -0.0479891, 0.0503665
6: -0.0090305, 0.0235997, -0.0124105, 0.0210896, -0.0301201, 0.0350616
7: -0.0336267, 0.0172778, -0.0330474, 0.0151759, -0.0488026, 0.0503252
8: -0.0094104, 0.0393607, -0.0145305, 0.0345505, -0.0439609, 0.0524356
9: 0.8711228, 0.9964198, 0.8803930, 1.0167825, -0.1349939, 0.1160268

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_B2_A2_A2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769368, upper bound: 0.0766317
time: 1.73 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769368, upper bound: 0.0769293
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0151504, 0.0122268, -0.0136098, 0.0189897, -0.0341401, 0.0258366
1: -0.0044738, 0.0456737, -0.0046693, 0.0479188, -0.0523926, 0.0503431
2: 0.0066015, 0.0374260, 0.0100419, 0.0403843, -0.0309328, 0.0273841
3: -0.0119519, 0.0243574, -0.0107427, 0.0262029, -0.0377595, 0.0351001
4: -0.0243875, 0.0043623, -0.0325947, 0.0099648, -0.0338626, 0.0339939
5: -0.0085869, 0.0360315, -0.0102172, 0.0378291, -0.0464160, 0.0462487
6: -0.0098947, 0.0198625, -0.0085681, 0.0236568, -0.0324098, 0.0279278
7: -0.0314094, 0.0107381, -0.0332427, 0.0173621, -0.0487715, 0.0439807
8: -0.0118503, 0.0320727, -0.0088119, 0.0391068, -0.0492796, 0.0398651
9: 0.8861941, 1.0066407, 0.8705651, 0.9917552, -0.0957493, 0.1246764

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_B1_A1_B1_B1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705164, upper bound: 0.0774594
time: 1.17 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_B1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0711800, upper bound: 0.0775563
time: 1.69 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0151342, 0.0122081, -0.0136815, 0.0198373, -0.0349715, 0.0258897
1: -0.0044285, 0.0456526, -0.0057936, 0.0480761, -0.0525046, 0.0514462
2: 0.0066416, 0.0374190, 0.0096487, 0.0406199, -0.0311708, 0.0277703
3: -0.0119158, 0.0243467, -0.0112301, 0.0263186, -0.0379493, 0.0355768
4: -0.0243722, 0.0043367, -0.0330853, 0.0124857, -0.0367838, 0.0345936
5: -0.0085568, 0.0360161, -0.0116047, 0.0379518, -0.0465087, 0.0476208
6: -0.0098699, 0.0198538, -0.0089167, 0.0241643, -0.0329196, 0.0283272
7: -0.0313963, 0.0107024, -0.0334851, 0.0186020, -0.0499983, 0.0441876
8: -0.0118141, 0.0320507, -0.0091358, 0.0400857, -0.0503394, 0.0401575
9: 0.8862201, 1.0064726, 0.8695226, 0.9923005, -0.0963260, 0.1261492

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_B1_A1_B1_B2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705391, upper bound: 0.0774594
time: 1.89 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_B2_B2

### Relational analysis result of NS_A1_B2_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712306, upper bound: 0.0775563
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A1

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

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0717707, upper bound: 0.0769588
time: 1.49 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718145, upper bound: 0.0769588
time: 2.33 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2_A2

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

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0717707, upper bound: 0.0782515
time: 1.69 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718145, upper bound: 0.0782515
time: 2.30 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0137891, 0.0195745, -0.0148359, 0.0183386, -0.0321278, 0.0344104
1: -0.0054458, 0.0481179, -0.0043609, 0.0487709, -0.0542167, 0.0524788
2: 0.0096393, 0.0405846, 0.0081500, 0.0402155, -0.0305762, 0.0312968
3: -0.0112205, 0.0262954, -0.0107344, 0.0266217, -0.0378422, 0.0368185
4: -0.0329605, 0.0113767, -0.0323172, 0.0084299, -0.0399605, 0.0429855
5: -0.0111117, 0.0379754, -0.0094935, 0.0384056, -0.0495173, 0.0474689
6: -0.0088819, 0.0239477, -0.0086001, 0.0235148, -0.0312898, 0.0310570
7: -0.0334606, 0.0182607, -0.0336103, 0.0165170, -0.0499776, 0.0518710
8: -0.0092325, 0.0397196, -0.0089423, 0.0390118, -0.0451046, 0.0451758
9: 0.8700617, 0.9930244, 0.8711649, 0.9941540, -0.1052417, 0.1040822

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0782789, upper bound: 0.0749519
time: 2.53 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783809, upper bound: 0.0749519
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0138156, 0.0198104, -0.0148730, 0.0189438, -0.0327594, 0.0346834
1: -0.0058390, 0.0481451, -0.0051766, 0.0486680, -0.0545070, 0.0533217
2: 0.0094718, 0.0406581, 0.0078983, 0.0404141, -0.0309422, 0.0322498
3: -0.0114470, 0.0263105, -0.0112105, 0.0265393, -0.0379863, 0.0374327
4: -0.0330811, 0.0119632, -0.0326382, 0.0096355, -0.0415466, 0.0440107
5: -0.0115352, 0.0379953, -0.0103861, 0.0383271, -0.0498622, 0.0483815
6: -0.0090573, 0.0240434, -0.0089653, 0.0236613, -0.0315579, 0.0315678
7: -0.0335105, 0.0186711, -0.0336166, 0.0174908, -0.0510013, 0.0522877
8: -0.0094309, 0.0399310, -0.0093177, 0.0394251, -0.0456206, 0.0457813
9: 0.8699026, 0.9936959, 0.8710692, 0.9954529, -0.1066284, 0.1046172

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0785819, upper bound: 0.0749532
time: 2.88 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786410, upper bound: 0.0749532
time: 2.50 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B1

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

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783559, upper bound: 0.0767745
time: 2.40 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783560, upper bound: 0.0770377
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B2

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

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783559, upper bound: 0.0768398
time: 3.37 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783560, upper bound: 0.0771019
time: 3.43 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0150407, 0.0116413, -0.0141035, 0.0213760, -0.0364167, 0.0257449
1: -0.0038524, 0.0455192, -0.0090787, 0.0498452, -0.0536977, 0.0545979
2: 0.0068535, 0.0372308, 0.0083666, 0.0414935, -0.0346400, 0.0288641
3: -0.0114399, 0.0242817, -0.0135455, 0.0273680, -0.0388079, 0.0378196
4: -0.0240666, 0.0040998, -0.0349431, 0.0158515, -0.0398027, 0.0360171
5: -0.0080792, 0.0359177, -0.0147629, 0.0392777, -0.0473569, 0.0506806
6: -0.0095077, 0.0197981, -0.0108493, 0.0255999, -0.0351076, 0.0300622
7: -0.0313050, 0.0098572, -0.0348681, 0.0213748, -0.0526798, 0.0447253
8: -0.0113760, 0.0318201, -0.0114410, 0.0425799, -0.0539559, 0.0421914
9: 0.8863806, 1.0050758, 0.8645734, 1.0004811, -0.1037661, 0.1405025

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705326, upper bound: 0.0793951
time: 2.01 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712206, upper bound: 0.0794828
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0150429, 0.0124449, -0.0140865, 0.0213431, -0.0363860, 0.0265314
1: -0.0043320, 0.0456679, -0.0090277, 0.0498233, -0.0541554, 0.0546956
2: 0.0068679, 0.0374400, 0.0084069, 0.0414734, -0.0346056, 0.0290331
3: -0.0118081, 0.0243931, -0.0135082, 0.0273571, -0.0391653, 0.0379012
4: -0.0244909, 0.0042085, -0.0349270, 0.0157769, -0.0401825, 0.0361317
5: -0.0085263, 0.0360335, -0.0147097, 0.0392618, -0.0477881, 0.0507432
6: -0.0098121, 0.0198832, -0.0108240, 0.0255813, -0.0353934, 0.0302177
7: -0.0313937, 0.0108755, -0.0348486, 0.0213272, -0.0527209, 0.0457241
8: -0.0116853, 0.0321020, -0.0113983, 0.0425430, -0.0542283, 0.0426140
9: 0.8859917, 1.0055314, 0.8646165, 1.0003359, -0.1046231, 0.1409149

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0787339
time: 3.12 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0800923
time: 2.11 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A1

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

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0784426
time: 2.13 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0786287
time: 5.47 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2

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

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718051, upper bound: 0.0801536
time: 4.52 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718058, upper bound: 0.0802516
time: 2.27 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0137891, 0.0195745, -0.0151447, 0.0200429, -0.0338320, 0.0347191
1: -0.0054458, 0.0481179, -0.0079673, 0.0504856, -0.0559314, 0.0560852
2: 0.0096393, 0.0405846, 0.0075630, 0.0411037, -0.0314645, 0.0330216
3: -0.0112205, 0.0262954, -0.0131164, 0.0276750, -0.0388955, 0.0391574
4: -0.0329605, 0.0113767, -0.0343124, 0.0126271, -0.0446701, 0.0452974
5: -0.0111117, 0.0379754, -0.0131424, 0.0396918, -0.0508035, 0.0511178
6: -0.0088819, 0.0239477, -0.0105984, 0.0251800, -0.0329296, 0.0329486
7: -0.0334606, 0.0182607, -0.0349656, 0.0196174, -0.0530779, 0.0532262
8: -0.0092325, 0.0397196, -0.0112110, 0.0418711, -0.0479800, 0.0474149
9: 0.8700617, 0.9930244, 0.8654189, 1.0016879, -0.1139480, 0.1105325

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0782609, upper bound: 0.0770375
time: 1.25 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783499, upper bound: 0.0770374
time: 2.56 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0138156, 0.0198104, -0.0151849, 0.0206253, -0.0344409, 0.0349952
1: -0.0058390, 0.0481451, -0.0087611, 0.0503895, -0.0562285, 0.0569062
2: 0.0094718, 0.0406581, 0.0072122, 0.0412725, -0.0318007, 0.0334459
3: -0.0114470, 0.0263105, -0.0135489, 0.0275981, -0.0390451, 0.0396907
4: -0.0330811, 0.0119632, -0.0346031, 0.0139276, -0.0462929, 0.0463433
5: -0.0115352, 0.0379953, -0.0139935, 0.0396168, -0.0511520, 0.0519888
6: -0.0090573, 0.0240434, -0.0109328, 0.0253384, -0.0332718, 0.0333937
7: -0.0335105, 0.0186711, -0.0349888, 0.0205389, -0.0540493, 0.0536599
8: -0.0094309, 0.0399310, -0.0115437, 0.0423076, -0.0486182, 0.0479558
9: 0.8699026, 0.9936959, 0.8654066, 1.0029004, -0.1152231, 0.1112993

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0785553, upper bound: 0.0770443
time: 1.77 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786141, upper bound: 0.0770443
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B1

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

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783559, upper bound: 0.0792330
time: 2.11 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783560, upper bound: 0.0794456
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B2

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

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783559, upper bound: 0.0792510
time: 2.28 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0783560, upper bound: 0.0794638
time: 1.37 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0141481, 0.0148794, -0.0154856, 0.0124396, -0.0265878, 0.0303649
1: -0.0067327, 0.0467287, -0.0048880, 0.0458204, -0.0525530, 0.0516166
2: 0.0082062, 0.0392772, 0.0058141, 0.0375266, -0.0259384, 0.0334631
3: -0.0139275, 0.0251292, -0.0122397, 0.0244096, -0.0375229, 0.0373690
4: -0.0265133, 0.0058491, -0.0245753, 0.0045755, -0.0275839, 0.0272503
5: -0.0109288, 0.0368935, -0.0088945, 0.0361202, -0.0470490, 0.0457880
6: -0.0115884, 0.0208009, -0.0101256, 0.0199167, -0.0302879, 0.0309265
7: -0.0322522, 0.0140456, -0.0314879, 0.0111609, -0.0434131, 0.0455334
8: -0.0138035, 0.0332257, -0.0121138, 0.0322793, -0.0448281, 0.0453395
9: 0.8805467, 1.0114024, 0.8860436, 1.0078429, -0.1272962, 0.1113910

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B1_B1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0740855, upper bound: 0.0718051
time: 1.80 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2_A1_B2

### Relational analysis result of NS_A2_A1_B1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0741300, upper bound: 0.0718058
time: 1.13 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0156687, 0.0150452, -0.0154856, 0.0124396, -0.0281083, 0.0305308
1: -0.0078102, 0.0474758, -0.0048880, 0.0458204, -0.0536306, 0.0523637
2: 0.0052127, 0.0394592, 0.0058141, 0.0375266, -0.0277954, 0.0336451
3: -0.0145044, 0.0254460, -0.0122397, 0.0244096, -0.0386756, 0.0376857
4: -0.0268684, 0.0066083, -0.0245753, 0.0045755, -0.0278292, 0.0277883
5: -0.0115846, 0.0373834, -0.0088945, 0.0361202, -0.0477048, 0.0462779
6: -0.0121317, 0.0209602, -0.0101256, 0.0199167, -0.0313245, 0.0310858
7: -0.0326990, 0.0146040, -0.0314879, 0.0111609, -0.0438599, 0.0460918
8: -0.0142110, 0.0340559, -0.0121138, 0.0322793, -0.0457478, 0.0461697
9: 0.8806884, 1.0153494, 0.8860436, 1.0078429, -0.1271545, 0.1162680

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B1_B1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0740855, upper bound: 0.0741778
time: 1.54 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0741300, upper bound: 0.0741796
time: 1.70 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0153387, 0.0146644, -0.0136098, 0.0189897, -0.0343284, 0.0282741
1: -0.0073136, 0.0473416, -0.0046693, 0.0479188, -0.0552324, 0.0520109
2: 0.0059995, 0.0392191, 0.0100419, 0.0403843, -0.0316638, 0.0291772
3: -0.0141450, 0.0253968, -0.0107427, 0.0262029, -0.0399753, 0.0361395
4: -0.0266337, 0.0063050, -0.0325947, 0.0099648, -0.0360473, 0.0380169
5: -0.0111970, 0.0372986, -0.0102172, 0.0378291, -0.0490261, 0.0475159
6: -0.0118422, 0.0208911, -0.0085681, 0.0236568, -0.0343239, 0.0294592
7: -0.0325924, 0.0140380, -0.0332427, 0.0173621, -0.0499545, 0.0472806
8: -0.0138845, 0.0338186, -0.0088119, 0.0391068, -0.0512843, 0.0426305
9: 0.8808990, 1.0139766, 0.8705651, 0.9917552, -0.1108563, 0.1327268

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A2_A1_B1_B2_B1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0728045, upper bound: 0.0774433
time: 1.09 seconds

## Relational analysis of NS_A2_A1_B1_B2_B1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0735040, upper bound: 0.0775382
time: 1.53 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0153233, 0.0146420, -0.0136815, 0.0198373, -0.0351606, 0.0283235
1: -0.0072735, 0.0473211, -0.0057936, 0.0480761, -0.0553496, 0.0531147
2: 0.0060374, 0.0392047, 0.0096487, 0.0406199, -0.0319068, 0.0295560
3: -0.0141111, 0.0253864, -0.0112301, 0.0263186, -0.0401631, 0.0366165
4: -0.0266201, 0.0062773, -0.0330853, 0.0124857, -0.0389526, 0.0387562
5: -0.0111676, 0.0372836, -0.0116047, 0.0379518, -0.0491195, 0.0488884
6: -0.0118185, 0.0208821, -0.0089167, 0.0241643, -0.0348283, 0.0297988
7: -0.0325783, 0.0140036, -0.0334851, 0.0186020, -0.0511803, 0.0474887
8: -0.0138479, 0.0337974, -0.0091358, 0.0400857, -0.0523111, 0.0429332
9: 0.8809283, 1.0138072, 0.8695226, 0.9923005, -0.1113722, 0.1341513

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A2_A1_B1_B2_B1_B2_B1

### Relational analysis result of NS_A2_A1_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0728409, upper bound: 0.0774436
time: 1.44 seconds

## Relational analysis of NS_A2_A1_B1_B2_B1_B2_B2

### Relational analysis result of NS_A2_A1_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0735496, upper bound: 0.0775382
time: 1.26 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0160097, 0.0154769, -0.0151272, 0.0186846, -0.0346943, 0.0306041
1: -0.0083188, 0.0477993, -0.0051587, 0.0487503, -0.0570691, 0.0529580
2: 0.0047449, 0.0397154, 0.0072333, 0.0403685, -0.0326424, 0.0324821
3: -0.0148234, 0.0256052, -0.0112556, 0.0265731, -0.0413054, 0.0368608
4: -0.0270774, 0.0069391, -0.0325846, 0.0090885, -0.0354970, 0.0382262
5: -0.0119602, 0.0376129, -0.0102055, 0.0383755, -0.0503358, 0.0478184
6: -0.0123997, 0.0210322, -0.0090305, 0.0235997, -0.0350562, 0.0300628
7: -0.0329190, 0.0151685, -0.0336267, 0.0172778, -0.0501968, 0.0487952
8: -0.0145032, 0.0343645, -0.0094104, 0.0393607, -0.0524205, 0.0437748
9: 0.8804663, 1.0167246, 0.8711228, 0.9964198, -0.1159534, 0.1349493

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_A1_B1_B2_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0742697, upper bound: 0.0769368
time: 1.66 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0742697, upper bound: 0.0782696
time: 1.52 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0141069, 0.0145425, -0.0152267, 0.0137827, -0.0278895, 0.0297692
1: -0.0064451, 0.0466642, -0.0065835, 0.0471705, -0.0536156, 0.0532476
2: 0.0083025, 0.0391022, 0.0062601, 0.0387684, -0.0304659, 0.0328421
3: -0.0136936, 0.0250981, -0.0135562, 0.0253145, -0.0382955, 0.0377643
4: -0.0263620, 0.0056874, -0.0262245, 0.0058974, -0.0273887, 0.0271072
5: -0.0106861, 0.0368463, -0.0105911, 0.0371732, -0.0478593, 0.0474374
6: -0.0114061, 0.0207695, -0.0113896, 0.0208049, -0.0322110, 0.0321591
7: -0.0321954, 0.0136291, -0.0324422, 0.0129674, -0.0451628, 0.0460714
8: -0.0135845, 0.0331071, -0.0133334, 0.0335129, -0.0470974, 0.0464405
9: 0.8806475, 1.0107346, 0.8811855, 1.0122691, -0.1316216, 0.1295490

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A1_B2_B1_A1_B1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0741235, upper bound: 0.0719711
time: 1.12 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_B1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0741235, upper bound: 0.0719711
time: 1.70 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0140891, 0.0145218, -0.0152352, 0.0147155, -0.0288046, 0.0297570
1: -0.0064037, 0.0466407, -0.0070770, 0.0473278, -0.0537314, 0.0537177
2: 0.0083438, 0.0390879, 0.0062615, 0.0391042, -0.0307604, 0.0328263
3: -0.0136586, 0.0250865, -0.0139364, 0.0254263, -0.0384377, 0.0382872
4: -0.0263486, 0.0056581, -0.0267017, 0.0060821, -0.0276220, 0.0276088
5: -0.0106558, 0.0368291, -0.0110396, 0.0372909, -0.0479467, 0.0478687
6: -0.0113814, 0.0207595, -0.0117029, 0.0209134, -0.0322948, 0.0324625
7: -0.0321790, 0.0135957, -0.0325555, 0.0140119, -0.0461909, 0.0461513
8: -0.0135473, 0.0330843, -0.0136430, 0.0338325, -0.0473798, 0.0467274
9: 0.8806784, 1.0105554, 0.8806803, 1.0127336, -0.1320552, 0.1298751

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A1_B2_B1_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0741597, upper bound: 0.0719868
time: 1.55 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0741597, upper bound: 0.0719868
time: 1.46 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0153387, 0.0146644, -0.0139375, 0.0206066, -0.0359453, 0.0286019
1: -0.0073136, 0.0473416, -0.0079980, 0.0496618, -0.0569331, 0.0553396
2: 0.0059995, 0.0392191, 0.0090493, 0.0411690, -0.0320910, 0.0301699
3: -0.0141450, 0.0253968, -0.0129070, 0.0272802, -0.0399709, 0.0383038
4: -0.0266337, 0.0063050, -0.0345181, 0.0140430, -0.0394823, 0.0386897
5: -0.0111970, 0.0372986, -0.0135768, 0.0391434, -0.0503404, 0.0508755
6: -0.0118422, 0.0208911, -0.0103860, 0.0252659, -0.0347169, 0.0312771
7: -0.0325924, 0.0140380, -0.0346249, 0.0202139, -0.0528063, 0.0486628
8: -0.0138845, 0.0338186, -0.0108597, 0.0418754, -0.0525724, 0.0446782
9: 0.8808990, 1.0139766, 0.8651679, 0.9986546, -0.1177557, 0.1312310

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_A1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B2_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of NS_A2_A1_B2_B2_B1_B1_B1

### Relational analysis result of NS_A2_A1_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0728045, upper bound: 0.0774456
time: 1.02 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_B1_B2

### Relational analysis result of NS_A2_A1_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0735040, upper bound: 0.0775383
time: 1.17 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0153233, 0.0146420, -0.0140176, 0.0215155, -0.0368388, 0.0286596
1: -0.0072735, 0.0473211, -0.0091599, 0.0498408, -0.0571143, 0.0564810
2: 0.0060374, 0.0392047, 0.0081190, 0.0414444, -0.0354069, 0.0310857
3: -0.0141111, 0.0253864, -0.0134038, 0.0274196, -0.0414427, 0.0387902
4: -0.0266201, 0.0062773, -0.0350671, 0.0166303, -0.0428260, 0.0394975
5: -0.0111676, 0.0372836, -0.0149740, 0.0392733, -0.0504410, 0.0522576
6: -0.0118185, 0.0208821, -0.0107458, 0.0257769, -0.0375954, 0.0316278
7: -0.0325783, 0.0140036, -0.0349012, 0.0215435, -0.0541219, 0.0489048
8: -0.0138479, 0.0337974, -0.0111775, 0.0429059, -0.0567538, 0.0449748
9: 0.8809283, 1.0138072, 0.8641019, 0.9992604, -0.1183321, 0.1497053

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_A1_B2_B2_B1_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741300, upper bound: 0.0770078
time: 1.15 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741300, upper bound: 0.0780268
time: 2.14 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B2_A1

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

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740855, upper bound: 0.0769736
time: 1.77 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741300, upper bound: 0.0769736
time: 1.48 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0156687, 0.0150452, -0.0154837, 0.0215660, -0.0372348, 0.0305289
1: -0.0078102, 0.0474758, -0.0100882, 0.0505141, -0.0583243, 0.0575639
2: 0.0052127, 0.0394592, 0.0063352, 0.0417820, -0.0365693, 0.0331240
3: -0.0145044, 0.0254460, -0.0143435, 0.0276428, -0.0421471, 0.0397895
4: -0.0268684, 0.0066083, -0.0350397, 0.0158828, -0.0422696, 0.0391297
5: -0.0115846, 0.0373834, -0.0153792, 0.0396903, -0.0512749, 0.0527626
6: -0.0121317, 0.0209602, -0.0115589, 0.0257092, -0.0378410, 0.0325191
7: -0.0326990, 0.0146040, -0.0352443, 0.0218823, -0.0545813, 0.0498483
8: -0.0142110, 0.0340559, -0.0122451, 0.0431442, -0.0573552, 0.0463011
9: 0.8806884, 1.0153494, 0.8647262, 1.0055966, -0.1249082, 0.1506232

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740855, upper bound: 0.0782410
time: 1.65 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0741300, upper bound: 0.0782412
time: 1.56 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0141035, 0.0213760, -0.0150407, 0.0116413, -0.0257449, 0.0364167
1: -0.0090787, 0.0498452, -0.0038524, 0.0455192, -0.0545979, 0.0536977
2: 0.0083666, 0.0414935, 0.0068535, 0.0372308, -0.0288641, 0.0346400
3: -0.0135455, 0.0273680, -0.0114399, 0.0242817, -0.0378196, 0.0388079
4: -0.0349431, 0.0158515, -0.0240666, 0.0040998, -0.0360171, 0.0398028
5: -0.0147629, 0.0392777, -0.0080792, 0.0359177, -0.0506806, 0.0473569
6: -0.0108493, 0.0255999, -0.0095077, 0.0197981, -0.0300622, 0.0351076
7: -0.0348681, 0.0213748, -0.0313050, 0.0098572, -0.0447253, 0.0526798
8: -0.0114410, 0.0425799, -0.0113760, 0.0318201, -0.0421914, 0.0539559
9: 0.8645734, 1.0004811, 0.8863806, 1.0050758, -0.1405025, 0.1037661

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 188

## Relational analysis of NS_A2_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0793951, upper bound: 0.0705326
time: 1.23 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0794828, upper bound: 0.0712206
time: 1.18 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0140865, 0.0213431, -0.0150429, 0.0124960, -0.0265825, 0.0363860
1: -0.0090277, 0.0498233, -0.0043320, 0.0456709, -0.0546986, 0.0541554
2: 0.0084069, 0.0414734, 0.0068679, 0.0375077, -0.0291008, 0.0346056
3: -0.0135082, 0.0273571, -0.0118081, 0.0243941, -0.0379022, 0.0391653
4: -0.0349270, 0.0157769, -0.0244909, 0.0042411, -0.0382211, 0.0401825
5: -0.0147097, 0.0392618, -0.0085263, 0.0360358, -0.0507455, 0.0477881
6: -0.0108240, 0.0255813, -0.0098121, 0.0198881, -0.0307120, 0.0353934
7: -0.0348486, 0.0213272, -0.0314056, 0.0108755, -0.0457241, 0.0527328
8: -0.0113983, 0.0425430, -0.0116853, 0.0321049, -0.0435032, 0.0542283
9: 0.8646165, 1.0003359, 0.8859606, 1.0055314, -0.1409149, 0.1143754

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B1_B1_A1_B2_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787339, upper bound: 0.0718058
time: 1.95 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B2_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787339, upper bound: 0.0718058
time: 2.91 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0139452, 0.0127406, -0.0282243, 0.0355112
1: -0.0100882, 0.0505141, -0.0040294, 0.0450448, -0.0551330, 0.0545435
2: 0.0063352, 0.0417820, 0.0087840, 0.0377538, -0.0314186, 0.0329979
3: -0.0143435, 0.0276428, -0.0118819, 0.0240768, -0.0384203, 0.0395247
4: -0.0350397, 0.0158828, -0.0244524, 0.0040243, -0.0380099, 0.0401822
5: -0.0153792, 0.0396903, -0.0084703, 0.0356174, -0.0509967, 0.0481607
6: -0.0115589, 0.0257092, -0.0097584, 0.0197835, -0.0313425, 0.0354676
7: -0.0352443, 0.0218823, -0.0310786, 0.0109855, -0.0462299, 0.0529609
8: -0.0122451, 0.0431442, -0.0119033, 0.0314945, -0.0437397, 0.0550475
9: 0.8647262, 1.0055966, 0.8857085, 1.0043519, -0.1396257, 0.1198882

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B1_B1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0784426, upper bound: 0.0737104
time: 1.18 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786287, upper bound: 0.0737159
time: 1.97 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0155654, 0.0127230, -0.0282067, 0.0368183
1: -0.0100882, 0.0505141, -0.0049765, 0.0458845, -0.0559727, 0.0554907
2: 0.0063352, 0.0417820, 0.0057728, 0.0378293, -0.0314942, 0.0360091
3: -0.0143435, 0.0276428, -0.0123109, 0.0244402, -0.0387837, 0.0399537
4: -0.0350397, 0.0158828, -0.0246377, 0.0047071, -0.0384063, 0.0405205
5: -0.0153792, 0.0396903, -0.0089802, 0.0361610, -0.0515402, 0.0486706
6: -0.0115589, 0.0257092, -0.0101846, 0.0199480, -0.0315069, 0.0358939
7: -0.0352443, 0.0218823, -0.0315558, 0.0113160, -0.0465603, 0.0534380
8: -0.0122451, 0.0431442, -0.0121822, 0.0323555, -0.0446007, 0.0553264
9: 0.8647262, 1.0055966, 0.8859212, 1.0079582, -0.1432320, 0.1196754

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B1_B1_A2_B2_B1

### Relational analysis result of NS_A2_A2_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0784426, upper bound: 0.0741778
time: 1.76 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_B2_B2

### Relational analysis result of NS_A2_A2_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786287, upper bound: 0.0741796
time: 1.71 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0139375, 0.0206066, -0.0155877, 0.0146696, -0.0286071, 0.0361943
1: -0.0079980, 0.0496618, -0.0073452, 0.0474979, -0.0554959, 0.0569806
2: 0.0090493, 0.0411690, 0.0058759, 0.0393394, -0.0302901, 0.0322712
3: -0.0129070, 0.0272802, -0.0141579, 0.0254844, -0.0383914, 0.0399770
4: -0.0345181, 0.0140430, -0.0266599, 0.0063101, -0.0386955, 0.0407029
5: -0.0135768, 0.0391434, -0.0112163, 0.0374073, -0.0509842, 0.0503597
6: -0.0103860, 0.0252659, -0.0118491, 0.0209264, -0.0313124, 0.0347206
7: -0.0346249, 0.0202139, -0.0326715, 0.0140434, -0.0486683, 0.0528854
8: -0.0108597, 0.0418754, -0.0139034, 0.0339365, -0.0447962, 0.0525818
9: 0.8651679, 0.9986546, 0.8808539, 1.0140145, -0.1312604, 0.1178007

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 188

## Relational analysis of NS_A2_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 188

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of NS_A2_A2_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0795133, upper bound: 0.0707122
time: 1.34 seconds

## Relational analysis of NS_A2_A2_B1_B2_A1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0796005, upper bound: 0.0713306
time: 1.18 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0140176, 0.0215155, -0.0155637, 0.0146470, -0.0286646, 0.0368864
1: -0.0091599, 0.0498408, -0.0073038, 0.0474727, -0.0566326, 0.0571446
2: 0.0081190, 0.0414444, 0.0059175, 0.0393214, -0.0312024, 0.0355268
3: -0.0134038, 0.0274196, -0.0141235, 0.0254713, -0.0388751, 0.0414459
4: -0.0350671, 0.0166303, -0.0266456, 0.0062824, -0.0395032, 0.0432759
5: -0.0149740, 0.0392733, -0.0111862, 0.0373890, -0.0523630, 0.0504595
6: -0.0107458, 0.0257769, -0.0118251, 0.0209162, -0.0316620, 0.0376020
7: -0.0349012, 0.0215435, -0.0326550, 0.0140089, -0.0489101, 0.0541985
8: -0.0111775, 0.0429059, -0.0138656, 0.0339118, -0.0450893, 0.0567715
9: 0.8641019, 0.9992604, 0.8808846, 1.0138438, -0.1497419, 0.1183757

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787339, upper bound: 0.0719868
time: 1.15 seconds

## Relational analysis of NS_A2_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0787339, upper bound: 0.0719868
time: 2.84 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0142899, 0.0148828, -0.0303665, 0.0358560
1: -0.0100882, 0.0505141, -0.0067487, 0.0468021, -0.0568902, 0.0572629
2: 0.0063352, 0.0417820, 0.0081384, 0.0393506, -0.0330154, 0.0336436
3: -0.0143435, 0.0276428, -0.0139338, 0.0251683, -0.0395118, 0.0415419
4: -0.0350397, 0.0158828, -0.0265285, 0.0058516, -0.0386157, 0.0424112
5: -0.0153792, 0.0396903, -0.0109380, 0.0369454, -0.0523246, 0.0506283
6: -0.0115589, 0.0257092, -0.0115919, 0.0208164, -0.0323754, 0.0373012
7: -0.0352443, 0.0218823, -0.0322887, 0.0140482, -0.0492925, 0.0541710
8: -0.0122451, 0.0431442, -0.0138130, 0.0332794, -0.0455245, 0.0569571
9: 0.8647262, 1.0055966, 0.8805223, 1.0114224, -0.1466962, 0.1250744

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786287, upper bound: 0.0737551
time: 2.25 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786287, upper bound: 0.0737950
time: 1.20 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0154837, 0.0215660, -0.0159866, 0.0150518, -0.0305355, 0.0373602
1: -0.0100882, 0.0505141, -0.0078510, 0.0476693, -0.0577575, 0.0583651
2: 0.0063352, 0.0417820, 0.0050576, 0.0396033, -0.0332681, 0.0367243
3: -0.0143435, 0.0276428, -0.0145207, 0.0255524, -0.0398959, 0.0421635
4: -0.0350397, 0.0158828, -0.0268997, 0.0066158, -0.0391374, 0.0427824
5: -0.0153792, 0.0396903, -0.0116096, 0.0375108, -0.0528901, 0.0512999
6: -0.0115589, 0.0257092, -0.0121406, 0.0210017, -0.0325606, 0.0378498
7: -0.0352443, 0.0218823, -0.0327927, 0.0146106, -0.0498549, 0.0546749
8: -0.0122451, 0.0431442, -0.0142346, 0.0341964, -0.0464416, 0.0573787
9: 0.8647262, 1.0055966, 0.8806283, 1.0153996, -0.1506734, 0.1249683

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 190

### Candidate
type: B, layer: 1, pos: 190

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786287, upper bound: 0.0741462
time: 1.72 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0786287, upper bound: 0.0741833
time: 3.20 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0157609, 0.0213709, -0.0154621, 0.0194001, -0.0351610, 0.0368330
1: -0.0100257, 0.0503879, -0.0063684, 0.0486618, -0.0586875, 0.0567563
2: 0.0060781, 0.0417080, 0.0067163, 0.0405315, -0.0344534, 0.0349918
3: -0.0144153, 0.0275206, -0.0120535, 0.0264628, -0.0408781, 0.0395741
4: -0.0345242, 0.0150525, -0.0324959, 0.0103638, -0.0441052, 0.0475484
5: -0.0151617, 0.0395811, -0.0114424, 0.0382911, -0.0534529, 0.0510234
6: -0.0116347, 0.0253156, -0.0096482, 0.0235427, -0.0342394, 0.0349639
7: -0.0350917, 0.0216800, -0.0336465, 0.0185448, -0.0536365, 0.0553265
8: -0.0123916, 0.0424625, -0.0101709, 0.0394366, -0.0492053, 0.0526334
9: 0.8661591, 1.0066195, 0.8720667, 0.9990969, -0.1329378, 0.1181868

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B2_B1_A1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0756437, upper bound: 0.0729844
time: 1.14 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_B1_B2

### Relational analysis result of NS_A2_A2_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0779658, upper bound: 0.0733576
time: 1.93 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0157609, 0.0213709, -0.0160570, 0.0246925, -0.0404534, 0.0374279
1: -0.0100257, 0.0503879, -0.0184703, 0.0478260, -0.0578517, 0.0688582
2: 0.0060781, 0.0417080, 0.0046806, 0.0421049, -0.0360268, 0.0370275
3: -0.0144153, 0.0275206, -0.0211633, 0.0258037, -0.0402191, 0.0486840
4: -0.0345242, 0.0150525, -0.0332659, 0.0216219, -0.0561461, 0.0483183
5: -0.0151617, 0.0395811, -0.0234690, 0.0376491, -0.0528108, 0.0630500
6: -0.0116347, 0.0253156, -0.0167009, 0.0240062, -0.0344407, 0.0420165
7: -0.0350917, 0.0216800, -0.0337476, 0.0290604, -0.0641520, 0.0554276
8: -0.0123916, 0.0424625, -0.0195407, 0.0412181, -0.0506900, 0.0620032
9: 0.8661591, 1.0066195, 0.8743130, 1.0253808, -0.1592217, 0.1160521

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B2_B1_A1_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0756437, upper bound: 0.0729844
time: 1.05 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_B2_B2

### Relational analysis result of NS_A2_A2_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0779658, upper bound: 0.0733576
time: 1.10 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0146477, 0.0291491, -0.0148364, 0.0190173, -0.0336650, 0.0439855
1: -0.0204166, 0.0485946, -0.0055746, 0.0481214, -0.0685380, 0.0541692
2: 0.0048122, 0.0469634, 0.0078110, 0.0403394, -0.0355272, 0.0391524
3: -0.0222374, 0.0263655, -0.0115439, 0.0261883, -0.0484257, 0.0379094
4: -0.0349919, 0.0278545, -0.0321327, 0.0095963, -0.0442152, 0.0599872
5: -0.0259184, 0.0383158, -0.0106930, 0.0379140, -0.0638324, 0.0490088
6: -0.0175443, 0.0258010, -0.0092461, 0.0232813, -0.0396540, 0.0350471
7: -0.0356113, 0.0312118, -0.0333241, 0.0177746, -0.0533859, 0.0645359
8: -0.0204656, 0.0436054, -0.0097372, 0.0387947, -0.0566948, 0.0533426
9: 0.8674380, 1.0260792, 0.8727716, 0.9969668, -0.1295288, 0.1380900

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of NS_A2_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769612, upper bound: 0.0708081
time: 1.21 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769612, upper bound: 0.0708081
time: 1.07 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0159701, 0.0290279, -0.0154486, 0.0194505, -0.0354206, 0.0444765
1: -0.0208359, 0.0492325, -0.0064527, 0.0484881, -0.0693241, 0.0556853
2: 0.0040781, 0.0469964, 0.0067397, 0.0405275, -0.0364494, 0.0402567
3: -0.0226216, 0.0266215, -0.0120725, 0.0263578, -0.0489794, 0.0386940
4: -0.0349212, 0.0273010, -0.0324811, 0.0106687, -0.0455900, 0.0597821
5: -0.0259852, 0.0387048, -0.0115677, 0.0381610, -0.0641462, 0.0502725
6: -0.0179553, 0.0259206, -0.0096568, 0.0235381, -0.0408144, 0.0355774
7: -0.0360674, 0.0312233, -0.0335809, 0.0186050, -0.0546724, 0.0648042
8: -0.0208613, 0.0440956, -0.0101698, 0.0394330, -0.0580771, 0.0542655
9: 0.8674562, 1.0298907, 0.8723156, 0.9990882, -0.1316320, 0.1432223

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 188
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 107

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749293, upper bound: 0.0728151
time: 1.02 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0749293, upper bound: 0.0732833
time: 1.57 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0141690, 0.0217598, -0.0151634, 0.0207970, -0.0349660, 0.0369231
1: -0.0095564, 0.0499182, -0.0090861, 0.0500277, -0.0595841, 0.0590043
2: 0.0080325, 0.0416972, 0.0071582, 0.0413365, -0.0333040, 0.0345390
3: -0.0138287, 0.0274024, -0.0138428, 0.0273615, -0.0411902, 0.0412452
4: -0.0351308, 0.0167084, -0.0341698, 0.0138129, -0.0489437, 0.0508782
5: -0.0152888, 0.0393312, -0.0142344, 0.0393411, -0.0546299, 0.0535656
6: -0.0110571, 0.0257505, -0.0111865, 0.0250192, -0.0360762, 0.0369370
7: -0.0349826, 0.0218801, -0.0347979, 0.0208097, -0.0557923, 0.0566780
8: -0.0116964, 0.0428943, -0.0119197, 0.0417722, -0.0534686, 0.0548140
9: 0.8642952, 1.0012407, 0.8667328, 1.0043522, -0.1400570, 0.1345079

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 188

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769612, upper bound: 0.0708081
time: 1.34 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769612, upper bound: 0.0708081
time: 1.81 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0140398, 0.0210739, -0.0157529, 0.0289586, -0.0429984, 0.0368267
1: -0.0087908, 0.0492594, -0.0206214, 0.0491691, -0.0579599, 0.0698808
2: 0.0087643, 0.0413190, 0.0043866, 0.0469254, -0.0381611, 0.0369323
3: -0.0134865, 0.0269893, -0.0224759, 0.0266120, -0.0400986, 0.0494652
4: -0.0342339, 0.0148893, -0.0348378, 0.0271330, -0.0613670, 0.0497271
5: -0.0143688, 0.0388331, -0.0258098, 0.0386817, -0.0530505, 0.0646428
6: -0.0108147, 0.0250469, -0.0178364, 0.0258385, -0.0366532, 0.0428834
7: -0.0344986, 0.0209843, -0.0359628, 0.0310710, -0.0655696, 0.0569470
8: -0.0114957, 0.0415798, -0.0207478, 0.0438832, -0.0553789, 0.0623277
9: 0.8665737, 1.0007604, 0.8676297, 1.0292873, -0.1627136, 0.1331307

Time for backsubstitution: 1.37 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.23 + 595.16 = 600.40 seconds
