## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.006503490000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0054520, 0.0084662, 0.0054520, 0.0084662, -0.0030142, 0.0030142)
1: (0.0001389, 0.0044804, 0.0001389, 0.0044804, -0.0043415, 0.0043415)
2: (-0.0218196, 0.0158076, -0.0218196, 0.0158076, -0.0376272, 0.0376272)
3: (-0.0039244, 0.0045587, -0.0039244, 0.0045587, -0.0084832, 0.0084832)
4: (0.0022956, 0.0174702, 0.0022956, 0.0174702, -0.0151747, 0.0151747)
5: (-0.0031277, 0.0051703, -0.0031277, 0.0051703, -0.0082980, 0.0082980)
6: (0.9918348, 1.0009363, 0.9918348, 1.0009363, -0.0091015, 0.0091015)
7: (-0.0092274, 0.0182413, -0.0092274, 0.0182413, -0.0243378, 0.0243378)
8: (-0.0019025, 0.0067032, -0.0019025, 0.0067032, -0.0086058, 0.0086058)
9: (-0.0242881, -0.0035319, -0.0242881, -0.0035319, -0.0207561, 0.0207561)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.97 + 3.71 = 5.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0069930, upper bound: 0.0069925

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067911, upper bound: 0.0066096
time: 2.92 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0068719, upper bound: 0.0068708
time: 2.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.70 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.70
Output dim: 6, lower bound: -0.0067911, upper bound: 0.0066096
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.70
Output dim: 6, lower bound: -0.0068719, upper bound: 0.0068708

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0063121, 0.0083496, 0.0059328, 0.0084219, -0.0021098, 0.0024169
1: 0.0003081, 0.0042546, 0.0001555, 0.0043946, -0.0040865, 0.0040991
2: -0.0188016, 0.0144430, -0.0206734, 0.0156735, -0.0344751, 0.0351164
3: -0.0038025, 0.0019397, -0.0039124, 0.0035641, -0.0073666, 0.0058522
4: 0.0028869, 0.0166810, 0.0023537, 0.0171705, -0.0142836, 0.0143273
5: -0.0030099, 0.0014823, -0.0030829, 0.0037697, -0.0067795, 0.0045652
6: 0.9919966, 0.9984513, 0.9918507, 0.9999925, -0.0079959, 0.0066006
7: -0.0081571, 0.0168126, -0.0091223, 0.0176987, -0.0227398, 0.0228374
8: -0.0015672, 0.0062556, -0.0018696, 0.0065332, -0.0081004, 0.0081252
9: -0.0217527, -0.0042012, -0.0233251, -0.0035977, -0.0181550, 0.0191239

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065913, upper bound: 0.0063652
time: 2.45 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066734, upper bound: 0.0065008
time: 2.77 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0059957, 0.0084161, 0.0056663, 0.0084465, -0.0024508, 0.0027499
1: 0.0001731, 0.0043834, 0.0001520, 0.0044422, -0.0042690, 0.0042314
2: -0.0205234, 0.0155315, -0.0213087, 0.0157017, -0.0362251, 0.0368402
3: -0.0038998, 0.0034339, -0.0039150, 0.0041154, -0.0080152, 0.0073489
4: 0.0024152, 0.0171312, 0.0023415, 0.0173366, -0.0149214, 0.0147898
5: -0.0030771, 0.0035864, -0.0031077, 0.0045461, -0.0076232, 0.0066941
6: 0.9918676, 0.9998691, 0.9918474, 1.0005156, -0.0086480, 0.0080217
7: -0.0090109, 0.0176277, -0.0091444, 0.0179994, -0.0239107, 0.0227303
8: -0.0018347, 0.0065110, -0.0018765, 0.0066274, -0.0084621, 0.0083875
9: -0.0231991, -0.0036673, -0.0238589, -0.0035839, -0.0196152, 0.0201915

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066532, upper bound: 0.0066899
time: 2.93 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0067472, upper bound: 0.0067467
time: 3.05 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 7.94 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 7.94
Output dim: 6, lower bound: -0.0065913, upper bound: 0.0063652
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 7.94
Output dim: 6, lower bound: -0.0066734, upper bound: 0.0065008
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.94
Output dim: 6, lower bound: -0.0066532, upper bound: 0.0066899
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.94
Output dim: 6, lower bound: -0.0067472, upper bound: 0.0067467

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 0.0063447, 0.0082613, 0.0062518, 0.0083843, -0.0020396, 0.0020095
1: 0.0003712, 0.0040834, 0.0001913, 0.0043217, -0.0039505, 0.0038922
2: -0.0165142, 0.0139338, -0.0196988, 0.0153851, -0.0318994, 0.0336326
3: -0.0037571, -0.0000451, -0.0038867, 0.0027183, -0.0064754, 0.0038416
4: 0.0031076, 0.0160828, 0.0024786, 0.0169156, -0.0138080, 0.0136042
5: -0.0029206, -0.0009836, -0.0030449, 0.0025787, -0.0054993, 0.0020612
6: 0.9920571, 0.9965681, 0.9918849, 0.9991901, -0.0071329, 0.0046831
7: -0.0077576, 0.0157298, -0.0088961, 0.0172373, -0.0216276, 0.0214434
8: -0.0014421, 0.0059164, -0.0017987, 0.0063887, -0.0078307, 0.0077151
9: -0.0198311, -0.0044510, -0.0225064, -0.0037391, -0.0160920, 0.0180554

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064641, upper bound: 0.0061682
time: 3.23 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064641, upper bound: 0.0062426
time: 3.11 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 0.0063283, 0.0083085, 0.0060987, 0.0084067, -0.0020783, 0.0022098
1: 0.0003395, 0.0041748, 0.0001668, 0.0043650, -0.0040256, 0.0040080
2: -0.0177360, 0.0141898, -0.0202778, 0.0155825, -0.0333184, 0.0344676
3: -0.0037799, 0.0010150, -0.0039043, 0.0032208, -0.0070007, 0.0049193
4: 0.0029966, 0.0164023, 0.0023931, 0.0170670, -0.0140704, 0.0140092
5: -0.0029683, 0.0001801, -0.0030675, 0.0032862, -0.0062545, 0.0032476
6: 0.9920267, 0.9975740, 0.9918615, 0.9996668, -0.0076401, 0.0057124
7: -0.0079584, 0.0163082, -0.0090508, 0.0175114, -0.0223700, 0.0220528
8: -0.0015050, 0.0060976, -0.0018472, 0.0064745, -0.0079795, 0.0079448
9: -0.0208574, -0.0043254, -0.0229928, -0.0036423, -0.0172151, 0.0186674

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064956, upper bound: 0.0064048
time: 3.50 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064956, upper bound: 0.0064067
time: 3.29 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0062609, 0.0083793, 0.0062562, 0.0083599, -0.0020990, 0.0021231
1: 0.0002089, 0.0043120, 0.0001998, 0.0042744, -0.0040655, 0.0041122
2: -0.0195697, 0.0152432, -0.0190664, 0.0153161, -0.0348858, 0.0343096
3: -0.0038740, 0.0026063, -0.0038805, 0.0021695, -0.0060436, 0.0064869
4: 0.0025401, 0.0168819, 0.0025086, 0.0167502, -0.0142101, 0.0143733
5: -0.0030398, 0.0024209, -0.0030202, 0.0018059, -0.0048457, 0.0054411
6: 0.9919018, 0.9990837, 0.9918931, 0.9986694, -0.0067676, 0.0071906
7: -0.0087847, 0.0171762, -0.0088419, 0.0169379, -0.0225945, 0.0218217
8: -0.0017638, 0.0063695, -0.0017818, 0.0062949, -0.0080587, 0.0081513
9: -0.0223979, -0.0038087, -0.0219751, -0.0037730, -0.0186250, 0.0181664

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064465, upper bound: 0.0065249
time: 3.35 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064663, upper bound: 0.0065227
time: 3.74 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0061676, 0.0084003, 0.0061372, 0.0084031, -0.0022355, 0.0022631
1: 0.0001845, 0.0043527, 0.0001833, 0.0043582, -0.0041736, 0.0041694
2: -0.0201134, 0.0154394, -0.0201861, 0.0154495, -0.0355629, 0.0356255
3: -0.0038915, 0.0030781, -0.0038924, 0.0031412, -0.0070327, 0.0069706
4: 0.0024551, 0.0170240, 0.0024508, 0.0170430, -0.0145879, 0.0145733
5: -0.0030611, 0.0030854, -0.0030639, 0.0031742, -0.0062353, 0.0061493
6: 0.9918785, 0.9995315, 0.9918773, 0.9995914, -0.0077129, 0.0076542
7: -0.0089386, 0.0174336, -0.0089465, 0.0174680, -0.0231413, 0.0223626
8: -0.0018121, 0.0064502, -0.0018146, 0.0064609, -0.0082730, 0.0082647
9: -0.0228547, -0.0037125, -0.0229158, -0.0037076, -0.0191471, 0.0192032

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065603, upper bound: 0.0066123
time: 3.23 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0066128, upper bound: 0.0066135
time: 3.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 9.06 seconds
NS_A1_A1_A1, status: Status.VERIFIED, split count: 3, time: 9.06
Output dim: 6, lower bound: -0.0064641, upper bound: 0.0061682
NS_A1_A1_A2, status: Status.VERIFIED, split count: 3, time: 9.06
Output dim: 6, lower bound: -0.0064641, upper bound: 0.0062426
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 9.06
Output dim: 6, lower bound: -0.0064956, upper bound: 0.0064048
NS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 9.06
Output dim: 6, lower bound: -0.0064956, upper bound: 0.0064067
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 9.06
Output dim: 6, lower bound: -0.0064465, upper bound: 0.0065249
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 9.06
Output dim: 6, lower bound: -0.0064663, upper bound: 0.0065227
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 9.06
Output dim: 6, lower bound: -0.0065603, upper bound: 0.0066123
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 9.06
Output dim: 6, lower bound: -0.0066128, upper bound: 0.0066135

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0062673, 0.0083628, 0.0062793, 0.0083049, -0.0020376, 0.0020835
1: 0.0002214, 0.0042801, 0.0002446, 0.0041680, -0.0039466, 0.0040356
2: -0.0191434, 0.0151421, -0.0176444, 0.0149551, -0.0340985, 0.0327865
3: -0.0038650, 0.0022364, -0.0038483, 0.0009356, -0.0048005, 0.0060846
4: 0.0025839, 0.0167704, 0.0026650, 0.0163784, -0.0137944, 0.0141054
5: -0.0030232, 0.0019000, -0.0029647, 0.0000682, -0.0030914, 0.0048647
6: 0.9919138, 0.9987328, 0.9919359, 0.9974986, -0.0055848, 0.0067969
7: -0.0087055, 0.0169744, -0.0085588, 0.0162648, -0.0219509, 0.0213318
8: -0.0017390, 0.0063063, -0.0016931, 0.0060840, -0.0078230, 0.0079994
9: -0.0220398, -0.0038583, -0.0207805, -0.0039500, -0.0180898, 0.0169222

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062739, upper bound: 0.0064168
time: 2.95 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063453, upper bound: 0.0064254
time: 3.17 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0062715, 0.0083543, 0.0061700, 0.0083103, -0.0020387, 0.0021843
1: 0.0002295, 0.0042636, 0.0000329, 0.0041783, -0.0039488, 0.0042307
2: -0.0189228, 0.0150769, -0.0177820, 0.0166625, -0.0355853, 0.0328589
3: -0.0038592, 0.0020450, -0.0040008, 0.0010550, -0.0049142, 0.0060458
4: 0.0026122, 0.0167127, 0.0019251, 0.0164144, -0.0138021, 0.0147876
5: -0.0030146, 0.0016305, -0.0029701, 0.0002364, -0.0032510, 0.0046005
6: 0.9919215, 0.9985511, 0.9917335, 0.9976119, -0.0056905, 0.0068176
7: -0.0086542, 0.0168700, -0.0098981, 0.0163300, -0.0219184, 0.0223894
8: -0.0017230, 0.0062736, -0.0021127, 0.0061044, -0.0078274, 0.0083863
9: -0.0218545, -0.0038903, -0.0208962, -0.0031126, -0.0187419, 0.0170058

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062698, upper bound: 0.0064139
time: 3.32 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063688, upper bound: 0.0064235
time: 3.19 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0062548, 0.0083834, 0.0062708, 0.0083445, -0.0020897, 0.0021126
1: 0.0001971, 0.0043200, 0.0002281, 0.0042447, -0.0040476, 0.0040919
2: -0.0196761, 0.0153381, -0.0186691, 0.0150879, -0.0347640, 0.0340072
3: -0.0038825, 0.0026987, -0.0038601, 0.0018248, -0.0057073, 0.0065588
4: 0.0024990, 0.0169097, 0.0026075, 0.0166463, -0.0141473, 0.0143022
5: -0.0030440, 0.0025510, -0.0030047, 0.0013204, -0.0043644, 0.0055557
6: 0.9918905, 0.9991714, 0.9919202, 0.9983423, -0.0064518, 0.0072511
7: -0.0088592, 0.0172266, -0.0086629, 0.0167499, -0.0224378, 0.0218726
8: -0.0017872, 0.0063853, -0.0017257, 0.0062360, -0.0080231, 0.0081110
9: -0.0224874, -0.0037622, -0.0216414, -0.0038849, -0.0186025, 0.0178792

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0063988, upper bound: 0.0065086
time: 3.45 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064562, upper bound: 0.0065117
time: 3.19 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0062586, 0.0083755, 0.0061458, 0.0083571, -0.0020985, 0.0022297
1: 0.0002044, 0.0043047, -0.0000140, 0.0042690, -0.0040645, 0.0043187
2: -0.0194713, 0.0152789, -0.0189938, 0.0170408, -0.0365121, 0.0342727
3: -0.0038772, 0.0025210, -0.0040346, 0.0021065, -0.0059837, 0.0065555
4: 0.0025247, 0.0168561, 0.0017612, 0.0167312, -0.0142066, 0.0150949
5: -0.0030360, 0.0023007, -0.0030174, 0.0017171, -0.0047532, 0.0053181
6: 0.9918976, 0.9990028, 0.9916885, 0.9986097, -0.0067121, 0.0073143
7: -0.0088128, 0.0171296, -0.0101948, 0.0169036, -0.0225021, 0.0232762
8: -0.0017726, 0.0063549, -0.0022056, 0.0062841, -0.0080568, 0.0085606
9: -0.0223153, -0.0037912, -0.0219141, -0.0029270, -0.0193883, 0.0181229

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 124
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_A2_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0064334, upper bound: 0.0065094
time: 3.31 seconds

## Relational analysis of NS_A2_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0065117, upper bound: 0.0065124
time: 3.25 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 9.00 seconds
NS_A2_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 9.00
Output dim: 6, lower bound: -0.0062739, upper bound: 0.0064168
NS_A2_B1_B1_B2, status: Status.VERIFIED, split count: 4, time: 9.00
Output dim: 6, lower bound: -0.0063453, upper bound: 0.0064254
NS_A2_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 9.00
Output dim: 6, lower bound: -0.0062698, upper bound: 0.0064139
NS_A2_B1_B2_B2, status: Status.VERIFIED, split count: 4, time: 9.00
Output dim: 6, lower bound: -0.0063688, upper bound: 0.0064235
NS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 9.00
Output dim: 6, lower bound: -0.0063988, upper bound: 0.0065086
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 9.00
Output dim: 6, lower bound: -0.0064562, upper bound: 0.0065117
NS_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 9.00
Output dim: 6, lower bound: -0.0064334, upper bound: 0.0065094
NS_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 9.00
Output dim: 6, lower bound: -0.0065117, upper bound: 0.0065124

## BFS NS instance: NS_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0062636, 0.0083593, 0.0062959, 0.0082800, -0.0020164, 0.0020634
1: 0.0002141, 0.0042733, 0.0002767, 0.0041197, -0.0039056, 0.0039966
2: -0.0190524, 0.0152009, -0.0169993, 0.0146959, -0.0337483, 0.0322002
3: -0.0038702, 0.0021574, -0.0038251, 0.0003758, -0.0042461, 0.0059825
4: 0.0025585, 0.0167466, 0.0027773, 0.0162097, -0.0136512, 0.0139693
5: -0.0030196, 0.0017888, -0.0029395, -0.0007200, -0.0022996, 0.0047283
6: 0.9919069, 0.9986578, 0.9919667, 0.9969675, -0.0050606, 0.0066911
7: -0.0087515, 0.0169313, -0.0083554, 0.0159595, -0.0215504, 0.0212657
8: -0.0017535, 0.0062928, -0.0016294, 0.0059883, -0.0077418, 0.0079222
9: -0.0219634, -0.0038295, -0.0202387, -0.0040772, -0.0178862, 0.0164092

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of NS_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062708, upper bound: 0.0062556
time: 2.75 seconds

## Relational analysis of NS_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063526, upper bound: 0.0064693
time: 3.22 seconds

## BFS NS instance: NS_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0062632, 0.0083626, 0.0062247, 0.0083001, -0.0020369, 0.0021379
1: 0.0002133, 0.0042796, 0.0001387, 0.0041586, -0.0039453, 0.0041410
2: -0.0191368, 0.0152071, -0.0175190, 0.0158092, -0.0349460, 0.0327261
3: -0.0038708, 0.0022306, -0.0039246, 0.0008268, -0.0046975, 0.0061552
4: 0.0025558, 0.0167686, 0.0022949, 0.0163456, -0.0137898, 0.0144738
5: -0.0030229, 0.0018919, -0.0029598, -0.0000850, -0.0029379, 0.0048517
6: 0.9919060, 0.9987273, 0.9918346, 0.9973953, -0.0054893, 0.0068927
7: -0.0087564, 0.0169713, -0.0092287, 0.0162055, -0.0217958, 0.0221964
8: -0.0017550, 0.0063053, -0.0019030, 0.0060654, -0.0078204, 0.0082083
9: -0.0220342, -0.0038264, -0.0206752, -0.0035311, -0.0185031, 0.0168488

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 89
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of NS_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062709, upper bound: 0.0062691
time: 3.55 seconds

## Relational analysis of NS_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064199, upper bound: 0.0064781
time: 3.26 seconds

## BFS NS instance: NS_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0062675, 0.0083512, 0.0061710, 0.0082904, -0.0020230, 0.0021802
1: 0.0002216, 0.0042577, 0.0000348, 0.0041399, -0.0039183, 0.0042229
2: -0.0188433, 0.0151401, -0.0172694, 0.0166475, -0.0354907, 0.0324095
3: -0.0038648, 0.0019759, -0.0039994, 0.0006102, -0.0044750, 0.0059754
4: 0.0025848, 0.0166919, 0.0019316, 0.0162803, -0.0136955, 0.0147602
5: -0.0030115, 0.0015333, -0.0029500, -0.0003900, -0.0026215, 0.0044833
6: 0.9919140, 0.9984857, 0.9917351, 0.9971898, -0.0052758, 0.0067506
7: -0.0087039, 0.0168323, -0.0098863, 0.0160873, -0.0215859, 0.0226741
8: -0.0017385, 0.0062618, -0.0021089, 0.0060284, -0.0077669, 0.0083707
9: -0.0217877, -0.0038593, -0.0204655, -0.0031200, -0.0186677, 0.0166062

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 119
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of NS_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0062984, upper bound: 0.0062553
time: 3.02 seconds

## Relational analysis of NS_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0063825, upper bound: 0.0064685
time: 3.57 seconds

## BFS NS instance: NS_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0062669, 0.0083551, 0.0060911, 0.0083143, -0.0020474, 0.0022640
1: 0.0002205, 0.0042651, -0.0001201, 0.0041861, -0.0039656, 0.0043852
2: -0.0189421, 0.0151494, -0.0178869, 0.0178963, -0.0368384, 0.0330362
3: -0.0038656, 0.0020617, -0.0041110, 0.0011460, -0.0050116, 0.0061726
4: 0.0025808, 0.0167177, 0.0013904, 0.0164418, -0.0138610, 0.0153273
5: -0.0030153, 0.0016540, -0.0029742, 0.0003645, -0.0033799, 0.0046281
6: 0.9919128, 0.9985672, 0.9915870, 0.9976982, -0.0057853, 0.0069802
7: -0.0087111, 0.0168791, -0.0108659, 0.0163796, -0.0218599, 0.0238031
8: -0.0017408, 0.0062765, -0.0024159, 0.0061200, -0.0078608, 0.0086923
9: -0.0218707, -0.0038548, -0.0209842, -0.0025074, -0.0193633, 0.0171295

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 89
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 119
type: A, layer: 1, pos: 119
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 179
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 179
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of NS_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064065, upper bound: 0.0062695
time: 3.32 seconds

## Relational analysis of NS_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0064773, upper bound: 0.0064776
time: 3.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 9.33 seconds
NS_A2_B2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 9.33
Output dim: 6, lower bound: -0.0062708, upper bound: 0.0062556
NS_A2_B2_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 9.33
Output dim: 6, lower bound: -0.0063526, upper bound: 0.0064693
NS_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 9.33
Output dim: 6, lower bound: -0.0062709, upper bound: 0.0062691
NS_A2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 5, time: 9.33
Output dim: 6, lower bound: -0.0064199, upper bound: 0.0064781
NS_A2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 9.33
Output dim: 6, lower bound: -0.0062984, upper bound: 0.0062553
NS_A2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 9.33
Output dim: 6, lower bound: -0.0063825, upper bound: 0.0064685
NS_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 9.33
Output dim: 6, lower bound: -0.0064065, upper bound: 0.0062695
NS_A2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 9.33
Output dim: 6, lower bound: -0.0064773, upper bound: 0.0064776

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 5.68 + 128.19 = 133.88 seconds
