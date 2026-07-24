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
execution time: IAR + RelationalAnalysis = 23.02 + 33.76 = 56.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3763568, upper bound: 0.3763567

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763541, upper bound: 0.3717821
time: 3.50 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763564, upper bound: 0.3763563
time: 3.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.19 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 7.19
Output dim: 6, lower bound: -0.3763541, upper bound: 0.3717821
NS_B2, status: Status.UNKNOWN, split count: 1, time: 7.19
Output dim: 6, lower bound: -0.3763564, upper bound: 0.3763563

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -8.7136593, -7.4737396, -8.7136564, -7.4734554, -0.7522168, 0.7518129
1: -15.5116272, -14.1026163, -15.5127535, -14.1026287, -0.8176448, 0.8187914
2: -3.9905825, -2.9777455, -3.9916091, -2.9777517, -0.6752579, 0.6756601
3: -9.8407736, -8.5059652, -9.8421192, -8.5059710, -1.0481758, 1.0493474
4: -5.8449588, -4.6253638, -5.8457174, -4.6252699, -0.8684649, 0.8695555
5: 1.0087898, 1.6792156, 1.0087914, 1.6796631, -0.5815158, 0.5809441
6: 6.6703134, 7.7253551, 6.6702108, 7.7253218, -0.7742429, 0.7742379
7: -19.4104080, -17.7361507, -19.4102135, -17.7360592, -0.8770018, 0.8762746
8: -1.3603361, -0.5810935, -1.3604627, -0.5803323, -0.6943450, 0.6942019
9: -6.4883809, -5.5115061, -6.4884253, -5.5115824, -0.6872311, 0.6874776

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 931

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763522, upper bound: 0.3706132
time: 3.43 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763222, upper bound: 0.3717499
time: 3.47 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -8.7136602, -7.4737396, -8.7136116, -7.4737406, -0.7520404, 0.7519937
1: -15.5116272, -14.1026106, -15.5116234, -14.1028337, -0.8185124, 0.8176632
2: -3.9905834, -2.9777427, -3.9905829, -2.9779563, -0.6748376, 0.6762986
3: -9.8407745, -8.5059624, -9.8407669, -8.5061369, -1.0488439, 1.0481710
4: -5.8449583, -4.6253605, -5.8449578, -4.6255150, -0.8689370, 0.8691101
5: 1.0087893, 1.6792159, 1.0088210, 1.6792063, -0.5810895, 0.5810661
6: 6.6702757, 7.7253547, 6.6702852, 7.7253551, -0.7740784, 0.7742689
7: -19.4105377, -17.7361507, -19.4105377, -17.7361565, -0.8765888, 0.8766561
8: -1.3603392, -0.5810926, -1.3601847, -0.5810928, -0.6948266, 0.6937237
9: -6.4883800, -5.5114541, -6.4883780, -5.5114555, -0.6873796, 0.6872308

Time for backsubstitution: 21.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 931

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763546, upper bound: 0.3751873
time: 3.50 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763246, upper bound: 0.3763246
time: 3.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.63 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 28.63
Output dim: 6, lower bound: -0.3763522, upper bound: 0.3706132
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 28.63
Output dim: 6, lower bound: -0.3763222, upper bound: 0.3717499
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 28.63
Output dim: 6, lower bound: -0.3763546, upper bound: 0.3751873
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 28.63
Output dim: 6, lower bound: -0.3763246, upper bound: 0.3763246

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -8.7124319, -7.4743166, -8.7135258, -7.4735041, -0.7512002, 0.7511094
1: -15.5113211, -14.1058731, -15.5127277, -14.1029282, -0.8169196, 0.8155055
2: -3.9910746, -2.9788480, -3.9915462, -2.9779706, -0.6754277, 0.6731815
3: -9.8397465, -8.5063686, -9.8420210, -8.5060072, -1.0472021, 1.0492692
4: -5.8465238, -4.6255679, -5.8456545, -4.6252928, -0.8701553, 0.8690753
5: 1.0093737, 1.6777637, 1.0088543, 1.6795241, -0.5808556, 0.5793741
6: 6.6705599, 7.7260575, 6.6702518, 7.7253065, -0.7738833, 0.7749968
7: -19.4065666, -17.7365894, -19.4098587, -17.7360935, -0.8732097, 0.8754895
8: -1.3600398, -0.5788398, -1.3604343, -0.5803766, -0.6933279, 0.6935804
9: -6.4897556, -5.5116649, -6.4883962, -5.5115976, -0.6887293, 0.6871285

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763157, upper bound: 0.3706133
time: 3.52 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763157, upper bound: 0.3706132
time: 3.44 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -8.7136583, -7.4737396, -8.7136564, -7.4734554, -0.7515774, 0.7518115
1: -15.5116262, -14.1026154, -15.5127535, -14.1026287, -0.8176429, 0.8172283
2: -3.9905829, -2.9777455, -3.9916091, -2.9777517, -0.6752574, 0.6754932
3: -9.8407526, -8.5059643, -9.8421192, -8.5059710, -1.0487738, 1.0493474
4: -5.8449569, -4.6254110, -5.8457174, -4.6252699, -0.8684654, 0.8721824
5: 1.0087895, 1.6792095, 1.0087914, 1.6796631, -0.5815158, 0.5804646
6: 6.6703358, 7.7253532, 6.6702108, 7.7253218, -0.7754192, 0.7742372
7: -19.4104061, -17.7361526, -19.4102135, -17.7360592, -0.8747737, 0.8762705
8: -1.3602867, -0.5810938, -1.3604627, -0.5803323, -0.6931820, 0.6941948
9: -6.4883776, -5.5115404, -6.4884253, -5.5115824, -0.6872282, 0.6895730

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3762855, upper bound: 0.3717498
time: 3.41 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3762855, upper bound: 0.3717498
time: 3.67 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -8.7124310, -7.4743137, -8.7134809, -7.4737878, -0.7510238, 0.7512908
1: -15.5113182, -14.1058645, -15.5116005, -14.1031322, -0.8177867, 0.8143780
2: -3.9910746, -2.9788470, -3.9905200, -2.9781747, -0.6750081, 0.6738198
3: -9.8397493, -8.5063677, -9.8406696, -8.5061750, -1.0478706, 1.0480931
4: -5.8465247, -4.6255641, -5.8448954, -4.6255350, -0.8706264, 0.8686304
5: 1.0093726, 1.6777642, 1.0088832, 1.6790670, -0.5804293, 0.5794952
6: 6.6705222, 7.7260580, 6.6703262, 7.7253380, -0.7737179, 0.7750278
7: -19.4067001, -17.7365913, -19.4101830, -17.7361946, -0.8727965, 0.8758719
8: -1.3600432, -0.5788395, -1.3601565, -0.5811367, -0.6938100, 0.6931014
9: -6.4897566, -5.5116129, -6.4883480, -5.5114698, -0.6888778, 0.6868813

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3717799, upper bound: 0.3751854
time: 3.47 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3717799, upper bound: 0.3706153
time: 4.01 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -8.7136593, -7.4737396, -8.7136116, -7.4737406, -0.7514009, 0.7519932
1: -15.5116253, -14.1026115, -15.5116234, -14.1028337, -0.8185110, 0.8161004
2: -3.9905825, -2.9777431, -3.9905829, -2.9779563, -0.6748374, 0.6761320
3: -9.8407516, -8.5059633, -9.8407669, -8.5061369, -1.0494423, 1.0481701
4: -5.8449564, -4.6254086, -5.8449578, -4.6255150, -0.8689365, 0.8717356
5: 1.0087887, 1.6792104, 1.0088210, 1.6792063, -0.5810895, 0.5805862
6: 6.6702967, 7.7253523, 6.6702852, 7.7253551, -0.7752542, 0.7742679
7: -19.4105377, -17.7361526, -19.4105377, -17.7361565, -0.8743608, 0.8766522
8: -1.3602903, -0.5810933, -1.3601847, -0.5810928, -0.6936631, 0.6937160
9: -6.4883766, -5.5114880, -6.4883780, -5.5114555, -0.6873772, 0.6893256

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3717498, upper bound: 0.3763221
time: 3.28 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3717498, upper bound: 0.3717538
time: 3.24 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.90 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 6, lower bound: -0.3763157, upper bound: 0.3706133
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 6, lower bound: -0.3763157, upper bound: 0.3706132
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 6, lower bound: -0.3762855, upper bound: 0.3717498
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 6, lower bound: -0.3762855, upper bound: 0.3717498
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 6, lower bound: -0.3717799, upper bound: 0.3751854
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 6, lower bound: -0.3717799, upper bound: 0.3706153
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 6, lower bound: -0.3717498, upper bound: 0.3763221
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 28.90
Output dim: 6, lower bound: -0.3717498, upper bound: 0.3717538

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -8.7124290, -7.4740314, -8.7135258, -7.4735041, -0.7510638, 0.7513769
1: -15.5124464, -14.1058865, -15.5127277, -14.1029282, -0.8180506, 0.8154914
2: -3.9920998, -2.9788561, -3.9915462, -2.9779706, -0.6752093, 0.6725605
3: -9.8410931, -8.5063763, -9.8420210, -8.5060072, -1.0483704, 1.0492616
4: -5.8472838, -4.6254745, -5.8456545, -4.6252928, -0.8709149, 0.8687458
5: 1.0093751, 1.6782104, 1.0088543, 1.6795241, -0.5807645, 0.5798531
6: 6.6704569, 7.7260246, 6.6702518, 7.7253065, -0.7740445, 0.7751627
7: -19.4063740, -17.7364979, -19.4098587, -17.7360935, -0.8730166, 0.8760242
8: -1.3601673, -0.5780797, -1.3604343, -0.5803766, -0.6930203, 0.6934137
9: -6.4898024, -5.5117421, -6.4883962, -5.5115976, -0.6889739, 0.6871269

Time for backsubstitution: 21.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of NS_B1_A1_A1_A1

### Relational analysis result of NS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763155, upper bound: 0.3692163
time: 3.45 seconds

## Relational analysis of NS_B1_A1_A1_A2

### Relational analysis result of NS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3761439, upper bound: 0.3704420
time: 3.35 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -8.7123833, -7.4743156, -8.7135258, -7.4735041, -0.7512922, 0.7511079
1: -15.5113182, -14.1060925, -15.5127277, -14.1029282, -0.8169179, 0.8154538
2: -3.9910736, -2.9790606, -3.9915462, -2.9779706, -0.6754279, 0.6736004
3: -9.8397379, -8.5065422, -9.8420210, -8.5060072, -1.0471883, 1.0490973
4: -5.8465233, -4.6257191, -5.8456545, -4.6252928, -0.8701553, 0.8693829
5: 1.0094049, 1.6777540, 1.0088543, 1.6795241, -0.5809166, 0.5793657
6: 6.6705327, 7.7260585, 6.6702518, 7.7253065, -0.7738395, 0.7749963
7: -19.4066982, -17.7366009, -19.4098587, -17.7360935, -0.8733394, 0.8754804
8: -1.3598883, -0.5788398, -1.3604343, -0.5803766, -0.6936364, 0.6935799
9: -6.4897532, -5.5116138, -6.4883962, -5.5115976, -0.6887274, 0.6871030

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of NS_B1_A1_A2_A1

### Relational analysis result of NS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3763155, upper bound: 0.3692164
time: 3.45 seconds

## Relational analysis of NS_B1_A1_A2_A2

### Relational analysis result of NS_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3761439, upper bound: 0.3704421
time: 3.56 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -8.7136574, -7.4734564, -8.7136564, -7.4734554, -0.7514415, 0.7520800
1: -15.5127535, -14.1026306, -15.5127535, -14.1026287, -0.8187754, 0.8172145
2: -3.9916086, -2.9777513, -3.9916091, -2.9777517, -0.6750398, 0.6748731
3: -9.8420954, -8.5059719, -9.8421192, -8.5059710, -1.0499372, 1.0493393
4: -5.8457170, -4.6253181, -5.8457174, -4.6252699, -0.8692255, 0.8718524
5: 1.0087914, 1.6796571, 1.0087914, 1.6796631, -0.5814242, 0.5809448
6: 6.6702323, 7.7253208, 6.6702108, 7.7253218, -0.7755799, 0.7744031
7: -19.4102135, -17.7360649, -19.4102135, -17.7360592, -0.8745797, 0.8768048
8: -1.3604145, -0.5803330, -1.3604627, -0.5803323, -0.6928730, 0.6940286
9: -6.4884214, -5.5116148, -6.4884253, -5.5115824, -0.6874743, 0.6895721

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3748878, upper bound: 0.3717497
time: 3.44 seconds

## Relational analysis of NS_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3761143, upper bound: 0.3715786
time: 3.46 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -8.7136106, -7.4737406, -8.7136564, -7.4734554, -0.7516694, 0.7518106
1: -15.5116234, -14.1028357, -15.5127535, -14.1026287, -0.8176425, 0.8171880
2: -3.9905829, -2.9779553, -3.9916091, -2.9777517, -0.6752577, 0.6759126
3: -9.8407440, -8.5061388, -9.8421192, -8.5059710, -1.0487590, 1.0491753
4: -5.8449569, -4.6255617, -5.8457174, -4.6252699, -0.8684659, 0.8724890
5: 1.0088203, 1.6792004, 1.0087914, 1.6796631, -0.5815773, 0.5804565
6: 6.6703072, 7.7253532, 6.6702108, 7.7253218, -0.7753758, 0.7742376
7: -19.4105358, -17.7361603, -19.4102135, -17.7360592, -0.8749039, 0.8762619
8: -1.3601364, -0.5810938, -1.3604627, -0.5803323, -0.6934900, 0.6941950
9: -6.4883738, -5.5114880, -6.4884253, -5.5115824, -0.6872263, 0.6895478

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3748878, upper bound: 0.3717496
time: 3.41 seconds

## Relational analysis of NS_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3761143, upper bound: 0.3715784
time: 3.95 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -8.7124290, -7.4740314, -8.7134809, -7.4737878, -0.7507944, 0.7516050
1: -15.5124464, -14.1058865, -15.5116005, -14.1031322, -0.8180232, 0.8143587
2: -3.9920998, -2.9788561, -3.9905200, -2.9781747, -0.6762493, 0.6727786
3: -9.8410931, -8.5063763, -9.8406696, -8.5061750, -1.0482049, 1.0480833
4: -5.8472838, -4.6254745, -5.8448954, -4.6255350, -0.8715534, 0.8679867
5: 1.0093751, 1.6782104, 1.0088832, 1.6790670, -0.5802763, 0.5800059
6: 6.6704569, 7.7260246, 6.6703262, 7.7253380, -0.7738791, 0.7749591
7: -19.4063740, -17.7364979, -19.4101830, -17.7361946, -0.8724737, 0.8763475
8: -1.3601673, -0.5780797, -1.3601565, -0.5811367, -0.6931863, 0.6940308
9: -6.4898024, -5.5117421, -6.4883480, -5.5114698, -0.6889501, 0.6868792

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of NS_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3717797, upper bound: 0.3737886
time: 3.35 seconds

## Relational analysis of NS_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3716082, upper bound: 0.3750146
time: 3.38 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -8.7123833, -7.4743156, -8.7134809, -7.4737878, -0.7509756, 0.7512889
1: -15.5113182, -14.1060925, -15.5116005, -14.1031322, -0.8177857, 0.8152275
2: -3.9910736, -2.9790606, -3.9905200, -2.9781747, -0.6750076, 0.6723585
3: -9.8397379, -8.5065422, -9.8406696, -8.5061750, -1.0478559, 1.0487518
4: -5.8465233, -4.6257191, -5.8448954, -4.6255350, -0.8706260, 0.8684564
5: 1.0094049, 1.6777540, 1.0088832, 1.6790670, -0.5803959, 0.5794864
6: 6.6705327, 7.7260585, 6.6703262, 7.7253380, -0.7739081, 0.7750273
7: -19.4066982, -17.7366009, -19.4101830, -17.7361946, -0.8728554, 0.8758628
8: -1.3598883, -0.5788398, -1.3601565, -0.5811367, -0.6927071, 0.6931009
9: -6.4897532, -5.5116138, -6.4883480, -5.5114698, -0.6888759, 0.6870286

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 148

## Relational analysis of NS_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3717797, upper bound: 0.3692162
time: 3.92 seconds

## Relational analysis of NS_B2_A1_A2_A2

### Relational analysis result of NS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3716082, upper bound: 0.3704459
time: 3.23 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -8.7136574, -7.4734564, -8.7136116, -7.4737406, -0.7511725, 0.7523079
1: -15.5127535, -14.1026306, -15.5116234, -14.1028337, -0.8187492, 0.8160815
2: -3.9916086, -2.9777513, -3.9905829, -2.9779563, -0.6760790, 0.6750910
3: -9.8420954, -8.5059719, -9.8407669, -8.5061369, -1.0497723, 1.0481608
4: -5.8457170, -4.6253181, -5.8449578, -4.6255150, -0.8698635, 0.8710923
5: 1.0087914, 1.6796571, 1.0088210, 1.6792063, -0.5809364, 0.5810983
6: 6.6702323, 7.7253208, 6.6702852, 7.7253551, -0.7754145, 0.7741995
7: -19.4102135, -17.7360649, -19.4105377, -17.7361565, -0.8740370, 0.8771279
8: -1.3604145, -0.5803330, -1.3601847, -0.5810928, -0.6930394, 0.6946454
9: -6.4884214, -5.5116148, -6.4883780, -5.5114555, -0.6874499, 0.6893244

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3703521, upper bound: 0.3763218
time: 3.35 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3715786, upper bound: 0.3761509
time: 3.28 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -8.7136106, -7.4737406, -8.7136116, -7.4737406, -0.7513537, 0.7519917
1: -15.5116234, -14.1028357, -15.5116234, -14.1028337, -0.8185105, 0.8169494
2: -3.9905829, -2.9779553, -3.9905829, -2.9779563, -0.6748371, 0.6746705
3: -9.8407440, -8.5061388, -9.8407669, -8.5061369, -1.0494270, 1.0488298
4: -5.8449569, -4.6255617, -5.8449578, -4.6255150, -0.8689361, 0.8715615
5: 1.0088203, 1.6792004, 1.0088210, 1.6792063, -0.5810571, 0.5805776
6: 6.6703072, 7.7253532, 6.6702852, 7.7253551, -0.7754450, 0.7742684
7: -19.4105358, -17.7361603, -19.4105377, -17.7361565, -0.8744187, 0.8766437
8: -1.3601364, -0.5810938, -1.3601847, -0.5810928, -0.6925597, 0.6937156
9: -6.4883738, -5.5114880, -6.4883780, -5.5114555, -0.6873748, 0.6894729

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3703521, upper bound: 0.3717495
time: 3.93 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3715786, upper bound: 0.3715820
time: 3.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.64 seconds
NS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3763155, upper bound: 0.3692163
NS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3761439, upper bound: 0.3704420
NS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3763155, upper bound: 0.3692164
NS_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3761439, upper bound: 0.3704421
NS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3748878, upper bound: 0.3717497
NS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3761143, upper bound: 0.3715786
NS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3748878, upper bound: 0.3717496
NS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3761143, upper bound: 0.3715784
NS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3717797, upper bound: 0.3737886
NS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3716082, upper bound: 0.3750146
NS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3717797, upper bound: 0.3692162
NS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3716082, upper bound: 0.3704459
NS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3703521, upper bound: 0.3763218
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3715786, upper bound: 0.3761509
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3703521, upper bound: 0.3717495
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.64
Output dim: 6, lower bound: -0.3715786, upper bound: 0.3715820

## BFS NS instance: NS_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -8.7124300, -7.4740295, -8.7135277, -7.4735041, -0.7510653, 0.7513781
1: -15.5124454, -14.1058846, -15.5127258, -14.1029282, -0.8180485, 0.8154905
2: -3.9921017, -2.9788561, -3.9915457, -2.9779706, -0.6752110, 0.6725597
3: -9.8410931, -8.5063705, -9.8420219, -8.5060072, -1.0483689, 1.0492649
4: -5.8472853, -4.6254749, -5.8456535, -4.6252913, -0.8709168, 0.8687449
5: 1.0093746, 1.6782112, 1.0088540, 1.6795235, -0.5807650, 0.5798545
6: 6.6704564, 7.7260246, 6.6702523, 7.7253051, -0.7740445, 0.7751610
7: -19.4063721, -17.7364979, -19.4098587, -17.7360954, -0.8730145, 0.8760223
8: -1.3601670, -0.5780768, -1.3604345, -0.5803764, -0.6930199, 0.6934168
9: -6.4898005, -5.5117426, -6.4883943, -5.5115976, -0.6889715, 0.6871252

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_B1_A1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749175, upper bound: 0.3737522
time: 3.55 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749175, upper bound: 0.3737520
time: 4.33 seconds

## BFS NS instance: NS_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -8.7124290, -7.4740314, -8.7135277, -7.4735041, -0.7510638, 0.7513769
1: -15.5124454, -14.1058874, -15.5127268, -14.1029301, -0.8180492, 0.8154914
2: -3.9921007, -2.9788580, -3.9915476, -2.9779701, -0.6752095, 0.6725605
3: -9.8410921, -8.5063753, -9.8420219, -8.5060091, -1.0483713, 1.0492618
4: -5.8472843, -4.6254749, -5.8456545, -4.6252928, -0.8709140, 0.8687468
5: 1.0093758, 1.6782094, 1.0088537, 1.6795235, -0.5807636, 0.5798533
6: 6.6704564, 7.7260232, 6.6702518, 7.7253051, -0.7740436, 0.7751601
7: -19.4063740, -17.7365036, -19.4098587, -17.7360954, -0.8730135, 0.8760238
8: -1.3601669, -0.5780802, -1.3604341, -0.5803759, -0.6930203, 0.6934144
9: -6.4898038, -5.5117421, -6.4883966, -5.5115981, -0.6889708, 0.6871266

Time for backsubstitution: 22.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_B1_A1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749175, upper bound: 0.3749778
time: 3.64 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749175, upper bound: 0.3749777
time: 4.41 seconds

## BFS NS instance: NS_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -8.7123842, -7.4743137, -8.7135277, -7.4735041, -0.7512927, 0.7511094
1: -15.5113163, -14.1060886, -15.5127258, -14.1029282, -0.8169160, 0.8154528
2: -3.9910769, -2.9790592, -3.9915457, -2.9779706, -0.6754293, 0.6735997
3: -9.8397388, -8.5065393, -9.8420219, -8.5060072, -1.0471883, 1.0491011
4: -5.8465261, -4.6257181, -5.8456535, -4.6252913, -0.8701577, 0.8693829
5: 1.0094037, 1.6777560, 1.0088540, 1.6795235, -0.5809176, 0.5793672
6: 6.6705313, 7.7260566, 6.6702523, 7.7253051, -0.7738400, 0.7749958
7: -19.4066963, -17.7366009, -19.4098587, -17.7360954, -0.8733377, 0.8754785
8: -1.3598891, -0.5788376, -1.3604345, -0.5803764, -0.6936364, 0.6935833
9: -6.4897509, -5.5116153, -6.4883943, -5.5115976, -0.6887245, 0.6871002

Time for backsubstitution: 22.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_B1_A1_A2_A1_B1

### Relational analysis result of NS_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749541, upper bound: 0.3692164
time: 3.68 seconds

## Relational analysis of NS_B1_A1_A2_A1_B2

### Relational analysis result of NS_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749541, upper bound: 0.3692163
time: 4.12 seconds

## BFS NS instance: NS_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -8.7123823, -7.4743156, -8.7135277, -7.4735041, -0.7512922, 0.7511077
1: -15.5113173, -14.1060934, -15.5127268, -14.1029301, -0.8169158, 0.8154540
2: -3.9910736, -2.9790606, -3.9915476, -2.9779701, -0.6754277, 0.6736002
3: -9.8397379, -8.5065413, -9.8420219, -8.5060091, -1.0471892, 1.0490983
4: -5.8465247, -4.6257186, -5.8456545, -4.6252928, -0.8701544, 0.8693848
5: 1.0094059, 1.6777540, 1.0088537, 1.6795235, -0.5809164, 0.5793660
6: 6.6705322, 7.7260580, 6.6702518, 7.7253051, -0.7738390, 0.7749946
7: -19.4066963, -17.7365990, -19.4098587, -17.7360954, -0.8733358, 0.8754804
8: -1.3598883, -0.5788403, -1.3604341, -0.5803759, -0.6936364, 0.6935804
9: -6.4897542, -5.5116143, -6.4883966, -5.5115981, -0.6887228, 0.6871028

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 931

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 148

## Relational analysis of NS_B1_A1_A2_A2_B1

### Relational analysis result of NS_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749541, upper bound: 0.3704421
time: 3.42 seconds

## Relational analysis of NS_B1_A1_A2_A2_B2

### Relational analysis result of NS_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3749541, upper bound: 0.3704423
time: 4.14 seconds

## BFS NS instance: NS_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -8.7136564, -7.4734559, -8.7136593, -7.4734526, -0.7514420, 0.7520807
1: -15.5127525, -14.1026306, -15.5127516, -14.1026278, -0.8187752, 0.8172116
2: -3.9916081, -2.9777517, -3.9916101, -2.9777517, -0.6750391, 0.6748738
3: -9.8420963, -8.5059719, -9.8421192, -8.5059671, -1.0499420, 1.0493398
4: -5.8457160, -4.6253181, -5.8457189, -4.6252708, -0.8692250, 0.8718524
5: 1.0087917, 1.6796577, 1.0087900, 1.6796643, -0.5814264, 0.5809462
6: 6.6702318, 7.7253199, 6.6702094, 7.7253218, -0.7755804, 0.7744036
7: -19.4102116, -17.7360611, -19.4102116, -17.7360611, -0.8745785, 0.8768027
8: -1.3604147, -0.5803335, -1.3604627, -0.5803299, -0.6928763, 0.6940289
9: -6.4884219, -5.5116143, -6.4884243, -5.5115824, -0.6874723, 0.6895685

Time for backsubstitution: 21.88 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.78 + 559.90 = 616.68 seconds
