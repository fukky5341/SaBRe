## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.36130252799999996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.7136621, -7.4737396, -8.7136621, -7.4737396, -0.7520418, 0.7520416)
1: (-15.5116262, -14.1026058, -15.5116262, -14.1026058, -0.8176699, 0.8176696)
2: (-3.9905839, -2.9777384, -3.9905839, -2.9777384, -0.6762986, 0.6762986)
3: (-9.8407764, -8.5059605, -9.8407764, -8.5059605, -1.0481901, 1.0481901)
4: (-5.8449583, -4.6253567, -5.8449583, -4.6253567, -0.8691096, 0.8691096)
5: (1.0087888, 1.6792165, 1.0087888, 1.6792165, -0.5810986, 0.5810986)
6: (6.6702757, 7.7253547, 6.6702757, 7.7253547, -0.7740793, 0.7740796)
7: (-19.4105358, -17.7361488, -19.4105358, -17.7361488, -0.8765984, 0.8765984)
8: (-1.3603432, -0.5810931, -1.3603432, -0.5810931, -0.6948266, 0.6948266)
9: (-6.4883795, -5.5114546, -6.4883795, -5.5114546, -0.6872334, 0.6872334)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.75 + 32.98 = 54.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3763568, upper bound: 0.3763567

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 931

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763550, upper bound: 0.3751876
time: 3.26 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763250, upper bound: 0.3763249
time: 3.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.85 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.85
Output dim: 6, lower bound: -0.3763550, upper bound: 0.3751876
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.85
Output dim: 6, lower bound: -0.3763250, upper bound: 0.3763249

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.7124329, -7.4743147, -8.7135296, -7.4737864, -0.7510247, 0.7513385
1: -15.5113192, -14.1058636, -15.5115995, -14.1029034, -0.8169434, 0.8143845
2: -3.9910736, -2.9788418, -3.9905200, -2.9779568, -0.6764693, 0.6738198
3: -9.8397474, -8.5063648, -9.8406773, -8.5059967, -1.0472174, 1.0481112
4: -5.8465252, -4.6255608, -5.8448949, -4.6253767, -0.8707991, 0.8686304
5: 1.0093722, 1.6777644, 1.0088508, 1.6790770, -0.5804379, 0.5795279
6: 6.6705222, 7.7260580, 6.6703167, 7.7253380, -0.7737203, 0.7748368
7: -19.4067001, -17.7365952, -19.4101868, -17.7361870, -0.8728058, 0.8758137
8: -1.3600461, -0.5788395, -1.3603135, -0.5811365, -0.6938095, 0.6942053
9: -6.4897566, -5.5116129, -6.4883504, -5.5114708, -0.6887321, 0.6868842

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 931

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751878, upper bound: 0.3751876
time: 3.38 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751878, upper bound: 0.3751876
time: 3.92 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.7136602, -7.4737406, -8.7136621, -7.4737396, -0.7514019, 0.7520409
1: -15.5116253, -14.1026077, -15.5116262, -14.1026058, -0.8176684, 0.8161063
2: -3.9905815, -2.9777384, -3.9905839, -2.9777384, -0.6762977, 0.6761322
3: -9.8407516, -8.5059586, -9.8407764, -8.5059605, -1.0487890, 1.0481894
4: -5.8449574, -4.6254039, -5.8449583, -4.6253567, -0.8691096, 0.8717365
5: 1.0087880, 1.6792107, 1.0087888, 1.6792165, -0.5810978, 0.5806193
6: 6.6702967, 7.7253532, 6.6702757, 7.7253547, -0.7752552, 0.7740791
7: -19.4105396, -17.7361546, -19.4105358, -17.7361488, -0.8743706, 0.8765945
8: -1.3602940, -0.5810935, -1.3603432, -0.5810931, -0.6936641, 0.6948197
9: -6.4883766, -5.5114875, -6.4883795, -5.5114546, -0.6872320, 0.6893287

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 931

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751878, upper bound: 0.3763248
time: 3.33 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751878, upper bound: 0.3763247
time: 3.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.36 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.36
Output dim: 6, lower bound: -0.3751878, upper bound: 0.3751876
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.36
Output dim: 6, lower bound: -0.3751878, upper bound: 0.3751876
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.36
Output dim: 6, lower bound: -0.3751878, upper bound: 0.3763248
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.36
Output dim: 6, lower bound: -0.3751878, upper bound: 0.3763247

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -8.7124329, -7.4743147, -8.7124329, -7.4743147, -0.7505088, 0.7505090
1: -15.5113192, -14.1058636, -15.5113192, -14.1058636, -0.8139963, 0.8139963
2: -3.9910736, -2.9788418, -3.9910736, -2.9788418, -0.6743100, 0.6743100
3: -9.8397474, -8.5063648, -9.8397474, -8.5063648, -1.0472646, 1.0472648
4: -5.8465252, -4.6255608, -5.8465252, -4.6255608, -0.8704882, 0.8704882
5: 1.0093722, 1.6777644, 1.0093722, 1.6777644, -0.5790696, 0.5790696
6: 6.6705222, 7.7260580, 6.6705222, 7.7260580, -0.7745843, 0.7745841
7: -19.4067001, -17.7365952, -19.4067001, -17.7365952, -0.8724055, 0.8724055
8: -1.3600461, -0.5788395, -1.3600461, -0.5788395, -0.6935310, 0.6935310
9: -6.4897566, -5.5116129, -6.4897566, -5.5116129, -0.6884983, 0.6884985

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3707756, upper bound: 0.3751854
time: 3.38 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3753497, upper bound: 0.3751873
time: 3.44 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.7124329, -7.4743147, -8.7136602, -7.4737406, -0.7510719, 0.7514772
1: -15.5113192, -14.1058636, -15.5116253, -14.1026077, -0.8172414, 0.8144233
2: -3.9910736, -2.9788418, -3.9905815, -2.9777384, -0.6767144, 0.6738937
3: -9.8397474, -8.5063648, -9.8407516, -8.5059586, -1.0472517, 1.0481677
4: -5.8465252, -4.6255608, -5.8449574, -4.6254039, -0.8707604, 0.8687677
5: 1.0093722, 1.6777644, 1.0087880, 1.6792107, -0.5805755, 0.5795856
6: 6.6705222, 7.7260580, 6.6702967, 7.7253532, -0.7737784, 0.7748504
7: -19.4067001, -17.7365952, -19.4105396, -17.7361546, -0.8728395, 0.8761611
8: -1.3600461, -0.5788395, -1.3602940, -0.5810935, -0.6938400, 0.6945076
9: -6.4897566, -5.5116129, -6.4883766, -5.5114875, -0.6886997, 0.6869767

Time for backsubstitution: 22.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3707756, upper bound: 0.3751855
time: 3.50 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3753497, upper bound: 0.3751873
time: 3.29 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.7136602, -7.4737406, -8.7124329, -7.4743147, -0.7514772, 0.7510719
1: -15.5116253, -14.1026077, -15.5113192, -14.1058636, -0.8144231, 0.8172414
2: -3.9905815, -2.9777384, -3.9910736, -2.9788418, -0.6738939, 0.6767144
3: -9.8407516, -8.5059586, -9.8397474, -8.5063648, -1.0481677, 1.0472519
4: -5.8449574, -4.6254039, -5.8465252, -4.6255608, -0.8687677, 0.8707604
5: 1.0087880, 1.6792107, 1.0093722, 1.6777644, -0.5795856, 0.5805755
6: 6.6702967, 7.7253532, 6.6705222, 7.7260580, -0.7748504, 0.7737782
7: -19.4105396, -17.7361546, -19.4067001, -17.7365952, -0.8761611, 0.8728395
8: -1.3602940, -0.5810935, -1.3600461, -0.5788395, -0.6945076, 0.6938400
9: -6.4883766, -5.5114875, -6.4897566, -5.5116129, -0.6869767, 0.6886992

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3763220
time: 3.58 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751874, upper bound: 0.3763245
time: 3.26 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.7136602, -7.4737406, -8.7136602, -7.4737406, -0.7514014, 0.7514017
1: -15.5116253, -14.1026077, -15.5116253, -14.1026077, -0.8161056, 0.8161054
2: -3.9905815, -2.9777384, -3.9905815, -2.9777384, -0.6761315, 0.6761312
3: -9.8407516, -8.5059586, -9.8407516, -8.5059586, -1.0487881, 1.0487881
4: -5.8449574, -4.6254039, -5.8449574, -4.6254039, -0.8717365, 0.8717360
5: 1.0087880, 1.6792107, 1.0087880, 1.6792107, -0.5806186, 0.5806189
6: 6.6702967, 7.7253532, 6.6702967, 7.7253532, -0.7752547, 0.7752547
7: -19.4105396, -17.7361546, -19.4105396, -17.7361546, -0.8743668, 0.8743670
8: -1.3602940, -0.5810935, -1.3602940, -0.5810935, -0.6936564, 0.6936567
9: -6.4883766, -5.5114875, -6.4883766, -5.5114875, -0.6893263, 0.6893263

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3763220
time: 3.32 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751874, upper bound: 0.3763245
time: 3.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.55 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 6, lower bound: -0.3707756, upper bound: 0.3751854
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 6, lower bound: -0.3753497, upper bound: 0.3751873
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 6, lower bound: -0.3707756, upper bound: 0.3751855
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 6, lower bound: -0.3753497, upper bound: 0.3751873
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3763220
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 6, lower bound: -0.3751874, upper bound: 0.3763245
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3763220
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.55
Output dim: 6, lower bound: -0.3751874, upper bound: 0.3763245

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.7124290, -7.4740314, -8.7124319, -7.4743166, -0.7502794, 0.7506843
1: -15.5124464, -14.1058865, -15.5113211, -14.1058731, -0.8151176, 0.8139722
2: -3.9920998, -2.9788561, -3.9910746, -2.9788480, -0.6736717, 0.6732688
3: -9.8410931, -8.5063763, -9.8397465, -8.5063686, -1.0484247, 1.0472498
4: -5.8472838, -4.6254745, -5.8465238, -4.6255679, -0.8709335, 0.8698444
5: 1.0093751, 1.6782104, 1.0093737, 1.6777637, -0.5789165, 0.5794861
6: 6.6704569, 7.7260246, 6.6705599, 7.7260575, -0.7747440, 0.7747490
7: -19.4063740, -17.7364979, -19.4065666, -17.7365894, -0.8720818, 0.8728096
8: -1.3601673, -0.5780797, -1.3600398, -0.5788398, -0.6929073, 0.6930485
9: -6.4898024, -5.5117421, -6.4897556, -5.5116649, -0.6887426, 0.6884961

Time for backsubstitution: 24.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3707756, upper bound: 0.3707766
time: 3.73 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3707756, upper bound: 0.3753476
time: 3.38 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.7123833, -7.4743156, -8.7124310, -7.4743137, -0.7504616, 0.7505078
1: -15.5113182, -14.1060925, -15.5113182, -14.1058645, -0.8139904, 0.8148398
2: -3.9910736, -2.9790606, -3.9910746, -2.9788470, -0.6743102, 0.6728489
3: -9.8397379, -8.5065422, -9.8397493, -8.5063677, -1.0472450, 1.0479188
4: -5.8465233, -4.6257191, -5.8465247, -4.6255641, -0.8704886, 0.8703151
5: 1.0094049, 1.6777540, 1.0093726, 1.6777642, -0.5790367, 0.5790610
6: 6.6705327, 7.7260585, 6.6705222, 7.7260580, -0.7747736, 0.7745829
7: -19.4066982, -17.7366009, -19.4067001, -17.7365913, -0.8724635, 0.8723958
8: -1.3598883, -0.5788398, -1.3600432, -0.5788395, -0.6924276, 0.6935301
9: -6.4897532, -5.5116138, -6.4897566, -5.5116129, -0.6884961, 0.6886449

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3753479, upper bound: 0.3707755
time: 3.38 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3753479, upper bound: 0.3753507
time: 3.29 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.7124290, -7.4740314, -8.7136583, -7.4737396, -0.7508426, 0.7516530
1: -15.5124464, -14.1058865, -15.5116262, -14.1026154, -0.8183627, 0.8143976
2: -3.9920998, -2.9788561, -3.9905829, -2.9777455, -0.6760755, 0.6728530
3: -9.8410931, -8.5063763, -9.8407526, -8.5059643, -1.0484123, 1.0481524
4: -5.8472838, -4.6254745, -5.8449569, -4.6254110, -0.8712058, 0.8681240
5: 1.0093751, 1.6782104, 1.0087895, 1.6792095, -0.5804219, 0.5800025
6: 6.6704569, 7.7260246, 6.6703358, 7.7253532, -0.7739367, 0.7750151
7: -19.4063740, -17.7364979, -19.4104061, -17.7361526, -0.8725162, 0.8765645
8: -1.3601673, -0.5780797, -1.3602867, -0.5810938, -0.6932163, 0.6940255
9: -6.4898024, -5.5117421, -6.4883776, -5.5115404, -0.6889431, 0.6869733

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3717799, upper bound: 0.3706143
time: 3.33 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3717799, upper bound: 0.3751854
time: 3.25 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.7123833, -7.4743156, -8.7136593, -7.4737396, -0.7510242, 0.7514760
1: -15.5113182, -14.1060925, -15.5116253, -14.1026115, -0.8172352, 0.8152664
2: -3.9910736, -2.9790606, -3.9905825, -2.9777431, -0.6767139, 0.6724331
3: -9.8397379, -8.5065422, -9.8407516, -8.5059633, -1.0472322, 1.0488214
4: -5.8465233, -4.6257191, -5.8449564, -4.6254086, -0.8707600, 0.8685937
5: 1.0094049, 1.6777540, 1.0087887, 1.6792104, -0.5805426, 0.5795772
6: 6.6705327, 7.7260585, 6.6702967, 7.7253523, -0.7739663, 0.7748492
7: -19.4066982, -17.7366009, -19.4105377, -17.7361526, -0.8728981, 0.8761511
8: -1.3598883, -0.5788398, -1.3602903, -0.5810933, -0.6927361, 0.6945071
9: -6.4897532, -5.5116138, -6.4883766, -5.5114880, -0.6886964, 0.6871228

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763523, upper bound: 0.3706133
time: 3.48 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763523, upper bound: 0.3751874
time: 3.79 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.7136574, -7.4734564, -8.7124319, -7.4743166, -0.7512484, 0.7512474
1: -15.5127535, -14.1026306, -15.5113211, -14.1058731, -0.8155441, 0.8172178
2: -3.9916086, -2.9777513, -3.9910746, -2.9788480, -0.6732564, 0.6756732
3: -9.8420954, -8.5059719, -9.8397465, -8.5063686, -1.0493250, 1.0472369
4: -5.8457170, -4.6253181, -5.8465238, -4.6255679, -0.8692131, 0.8701167
5: 1.0087914, 1.6796571, 1.0093737, 1.6777637, -0.5794322, 0.5809929
6: 6.6702323, 7.7253208, 6.6705599, 7.7260575, -0.7750096, 0.7739418
7: -19.4102135, -17.7360649, -19.4065666, -17.7365894, -0.8758364, 0.8732431
8: -1.3604145, -0.5803330, -1.3600398, -0.5788398, -0.6938829, 0.6933575
9: -6.4884214, -5.5116148, -6.4897556, -5.5116649, -0.6872211, 0.6886973

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3717809
time: 3.30 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3763520
time: 3.40 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.7136106, -7.4737406, -8.7124310, -7.4743137, -0.7514296, 0.7510710
1: -15.5116234, -14.1028357, -15.5113182, -14.1058645, -0.8144166, 0.8180852
2: -3.9905829, -2.9779553, -3.9910746, -2.9788470, -0.6738944, 0.6752534
3: -9.8407440, -8.5061388, -9.8397493, -8.5063677, -1.0481482, 1.0479069
4: -5.8449569, -4.6255617, -5.8465247, -4.6255641, -0.8687682, 0.8705878
5: 1.0088203, 1.6792004, 1.0093726, 1.6777642, -0.5795536, 0.5805669
6: 6.6703072, 7.7253532, 6.6705222, 7.7260580, -0.7750406, 0.7737768
7: -19.4105358, -17.7361603, -19.4067001, -17.7365913, -0.8762176, 0.8728299
8: -1.3601364, -0.5810938, -1.3600432, -0.5788395, -0.6934042, 0.6938393
9: -6.4883738, -5.5114880, -6.4897566, -5.5116129, -0.6869736, 0.6888456

Time for backsubstitution: 21.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751857, upper bound: 0.3717803
time: 3.45 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751857, upper bound: 0.3763544
time: 4.03 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.7136574, -7.4734564, -8.7136583, -7.4737396, -0.7511735, 0.7515769
1: -15.5127535, -14.1026306, -15.5116262, -14.1026154, -0.8172264, 0.8160806
2: -3.9916086, -2.9777513, -3.9905829, -2.9777455, -0.6754928, 0.6750906
3: -9.8420954, -8.5059719, -9.8407526, -8.5059643, -1.0499444, 1.0487733
4: -5.8457170, -4.6253181, -5.8449569, -4.6254110, -0.8721814, 0.8710933
5: 1.0087914, 1.6796571, 1.0087895, 1.6792095, -0.5804651, 0.5810361
6: 6.6702323, 7.7253208, 6.6703358, 7.7253532, -0.7754140, 0.7754185
7: -19.4102135, -17.7360649, -19.4104061, -17.7361526, -0.8740420, 0.8747704
8: -1.3604145, -0.5803330, -1.3602867, -0.5810938, -0.6930323, 0.6931753
9: -6.4884214, -5.5116148, -6.4883776, -5.5115404, -0.6895707, 0.6893239

Time for backsubstitution: 21.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3717509
time: 3.61 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3763232
time: 3.39 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.7136106, -7.4737406, -8.7136593, -7.4737396, -0.7513547, 0.7514005
1: -15.5116234, -14.1028357, -15.5116253, -14.1026115, -0.8160989, 0.8169494
2: -3.9905829, -2.9779553, -3.9905825, -2.9777431, -0.6761312, 0.6746702
3: -9.8407440, -8.5061388, -9.8407516, -8.5059633, -1.0487680, 1.0494413
4: -5.8449569, -4.6255617, -5.8449564, -4.6254086, -0.8717360, 0.8715620
5: 1.0088203, 1.6792004, 1.0087887, 1.6792104, -0.5805857, 0.5806100
6: 6.6703072, 7.7253532, 6.6702967, 7.7253523, -0.7754455, 0.7752545
7: -19.4105358, -17.7361603, -19.4105377, -17.7361526, -0.8744242, 0.8743570
8: -1.3601364, -0.5810938, -1.3602903, -0.5810933, -0.6925535, 0.6936569
9: -6.4883738, -5.5114880, -6.4883766, -5.5114880, -0.6893229, 0.6894727

Time for backsubstitution: 21.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751856, upper bound: 0.3717498
time: 3.30 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751856, upper bound: 0.3763244
time: 3.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.82 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3707756, upper bound: 0.3707766
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3707756, upper bound: 0.3753476
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3753479, upper bound: 0.3707755
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3753479, upper bound: 0.3753507
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3717799, upper bound: 0.3706143
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3717799, upper bound: 0.3751854
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3763523, upper bound: 0.3706133
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3763523, upper bound: 0.3751874
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3717809
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3763520
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3751857, upper bound: 0.3717803
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3751857, upper bound: 0.3763544
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3717509
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3706133, upper bound: 0.3763232
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3751856, upper bound: 0.3717498
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.82
Output dim: 6, lower bound: -0.3751856, upper bound: 0.3763244

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.7124290, -7.4740314, -8.7124290, -7.4740314, -0.7505474, 0.7505474
1: -15.5124464, -14.1058865, -15.5124464, -14.1058865, -0.8151035, 0.8151035
2: -3.9920998, -2.9788561, -3.9920998, -2.9788561, -0.6730504, 0.6730506
3: -9.8410931, -8.5063763, -9.8410931, -8.5063763, -1.0484176, 1.0484176
4: -5.8472838, -4.6254745, -5.8472838, -4.6254745, -0.8706040, 0.8706040
5: 1.0093751, 1.6782104, 1.0093751, 1.6782104, -0.5793955, 0.5793953
6: 6.6704569, 7.7260246, 6.6704569, 7.7260246, -0.7749100, 0.7749100
7: -19.4063740, -17.7364979, -19.4063740, -17.7364979, -0.8726168, 0.8726168
8: -1.3601673, -0.5780797, -1.3601673, -0.5780797, -0.6927404, 0.6927407
9: -6.4898024, -5.5117421, -6.4898024, -5.5117421, -0.6887407, 0.6887410

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3707754, upper bound: 0.3739128
time: 3.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3706015, upper bound: 0.3751382
time: 3.35 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.7124290, -7.4740314, -8.7123833, -7.4743156, -0.7502785, 0.7507758
1: -15.5124464, -14.1058865, -15.5113182, -14.1060925, -0.8150659, 0.8139710
2: -3.9920998, -2.9788561, -3.9910736, -2.9790606, -0.6740904, 0.6732690
3: -9.8410931, -8.5063763, -9.8397379, -8.5065422, -1.0482531, 1.0472355
4: -5.8472838, -4.6254745, -5.8465233, -4.6257191, -0.8712411, 0.8698449
5: 1.0093751, 1.6782104, 1.0094049, 1.6777540, -0.5789082, 0.5795474
6: 6.6704569, 7.7260246, 6.6705327, 7.7260585, -0.7747436, 0.7747047
7: -19.4063740, -17.7364979, -19.4066982, -17.7366009, -0.8720732, 0.8729398
8: -1.3601673, -0.5780797, -1.3598883, -0.5788398, -0.6929069, 0.6933568
9: -6.4898024, -5.5117421, -6.4897532, -5.5116138, -0.6887174, 0.6884940

Time for backsubstitution: 21.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3707754, upper bound: 0.3739482
time: 3.38 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3706015, upper bound: 0.3751736
time: 3.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.7123833, -7.4743156, -8.7124290, -7.4740314, -0.7507758, 0.7502785
1: -15.5113182, -14.1060925, -15.5124464, -14.1058865, -0.8139708, 0.8150661
2: -3.9910736, -2.9790606, -3.9920998, -2.9788561, -0.6732690, 0.6740904
3: -9.8397379, -8.5065422, -9.8410931, -8.5063763, -1.0472355, 1.0482531
4: -5.8465233, -4.6257191, -5.8472838, -4.6254745, -0.8698449, 0.8712411
5: 1.0094049, 1.6777540, 1.0093751, 1.6782104, -0.5795472, 0.5789080
6: 6.6705327, 7.7260585, 6.6704569, 7.7260246, -0.7747049, 0.7747436
7: -19.4066982, -17.7366009, -19.4063740, -17.7364979, -0.8729396, 0.8720729
8: -1.3598883, -0.5788398, -1.3601673, -0.5780797, -0.6933565, 0.6929069
9: -6.4897532, -5.5116138, -6.4898024, -5.5117421, -0.6884937, 0.6887171

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3753477, upper bound: 0.3693760
time: 3.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751738, upper bound: 0.3706015
time: 3.40 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.7123833, -7.4743156, -8.7123833, -7.4743156, -0.7504601, 0.7504601
1: -15.5113182, -14.1060925, -15.5113182, -14.1060925, -0.8148391, 0.8148391
2: -3.9910736, -2.9790606, -3.9910736, -2.9790606, -0.6728485, 0.6728487
3: -9.8397379, -8.5065422, -9.8397379, -8.5065422, -1.0479040, 1.0479040
4: -5.8465233, -4.6257191, -5.8465233, -4.6257191, -0.8703146, 0.8703146
5: 1.0094049, 1.6777540, 1.0094049, 1.6777540, -0.5790281, 0.5790281
6: 6.6705327, 7.7260585, 6.6705327, 7.7260585, -0.7747731, 0.7747731
7: -19.4066982, -17.7366009, -19.4066982, -17.7366009, -0.8724546, 0.8724544
8: -1.3598883, -0.5788398, -1.3598883, -0.5788398, -0.6924267, 0.6924269
9: -6.4897532, -5.5116138, -6.4897532, -5.5116138, -0.6886430, 0.6886430

Time for backsubstitution: 22.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3753477, upper bound: 0.3693792
time: 3.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751738, upper bound: 0.3706025
time: 3.25 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.7124290, -7.4740314, -8.7136574, -7.4734564, -0.7511110, 0.7515163
1: -15.5124464, -14.1058865, -15.5127535, -14.1026306, -0.8183491, 0.8155301
2: -3.9920998, -2.9788561, -3.9916086, -2.9777513, -0.6754551, 0.6726353
3: -9.8410931, -8.5063763, -9.8420954, -8.5059719, -1.0484047, 1.0493171
4: -5.8472838, -4.6254745, -5.8457170, -4.6253181, -0.8708763, 0.8688836
5: 1.0093751, 1.6782104, 1.0087914, 1.6796571, -0.5809021, 0.5799110
6: 6.6704569, 7.7260246, 6.6702323, 7.7253208, -0.7741027, 0.7751758
7: -19.4063740, -17.7364979, -19.4102135, -17.7360649, -0.8730502, 0.8763709
8: -1.3601673, -0.5780797, -1.3604145, -0.5803330, -0.6930494, 0.6937165
9: -6.4898024, -5.5117421, -6.4884214, -5.5116148, -0.6889420, 0.6872196

Time for backsubstitution: 22.28 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 54.73 + 554.13 = 608.87 seconds
