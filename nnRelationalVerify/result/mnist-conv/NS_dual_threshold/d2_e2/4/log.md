## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.3662410892


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (8.1516037, 8.9561014, 8.1516037, 8.9561014, -0.5508573, 0.5508573)
1: (-14.3047848, -13.0300407, -14.3047848, -13.0300407, -0.8362961, 0.8362958)
2: (-7.3321538, -6.4065275, -7.3321538, -6.4065275, -0.6987977, 0.6987977)
3: (-8.9521961, -7.9962053, -8.9521961, -7.9962053, -0.7168481, 0.7168481)
4: (-12.9534950, -11.8476171, -12.9534950, -11.8476171, -0.7461720, 0.7461717)
5: (-5.7369843, -4.8009620, -5.7369843, -4.8009620, -0.6630130, 0.6630135)
6: (-3.2953463, -2.4089787, -3.2953463, -2.4089787, -0.6843560, 0.6843560)
7: (-8.3850746, -7.5282459, -8.3850746, -7.5282459, -0.6726513, 0.6726513)
8: (-3.7184324, -2.8380423, -3.7184324, -2.8380423, -0.6035771, 0.6035774)
9: (-2.2429171, -1.3550446, -2.2429171, -1.3550446, -0.6827922, 0.6827924)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.55 + 37.24 = 60.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.3684516, upper bound: 0.3684522

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 554
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4559
type: A, layer: 1, pos: 4559
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 554

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3664168, upper bound: 0.3682010
time: 5.70 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684485, upper bound: 0.3684490
time: 5.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 11.32 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 11.32
Output dim: 0, lower bound: -0.3664168, upper bound: 0.3682010
NS_A2, status: Status.UNKNOWN, split count: 1, time: 11.32
Output dim: 0, lower bound: -0.3684485, upper bound: 0.3684490

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 8.1623478, 8.9533863, 8.1550903, 8.9560757, -0.5388300, 0.5431650
1: -14.2891693, -13.0395498, -14.2996893, -13.0300407, -0.8210006, 0.8201270
2: -7.3101597, -6.4226956, -7.3243828, -6.4066448, -0.6766362, 0.6745648
3: -8.9358149, -8.0191727, -8.9515715, -8.0042715, -0.6925859, 0.6933091
4: -12.9372425, -11.8586636, -12.9479094, -11.8476982, -0.7298813, 0.7280266
5: -5.7288494, -4.8133407, -5.7367878, -4.8052545, -0.6498151, 0.6504059
6: -3.2793479, -2.4259696, -3.2933507, -2.4147608, -0.6578717, 0.6650281
7: -8.3770514, -7.5316186, -8.3825684, -7.5282474, -0.6647272, 0.6629837
8: -3.7115664, -2.8498073, -3.7182927, -2.8420095, -0.5908399, 0.5918779
9: -2.2398999, -1.3590379, -2.2424450, -1.3563192, -0.6769028, 0.6781037

Time for backsubstitution: 20.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 4559
type: A, layer: 1, pos: 4559
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6126

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3642947, upper bound: 0.3680086
time: 5.10 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3664147, upper bound: 0.3681999
time: 6.99 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 8.1516075, 8.9561014, 8.1516037, 8.9561014, -0.5467839, 0.5501347
1: -14.3047743, -13.0300407, -14.3047829, -13.0300407, -0.8218241, 0.8352618
2: -7.3321443, -6.4065280, -7.3321495, -6.4065280, -0.6715567, 0.6936285
3: -8.9521952, -7.9962063, -8.9521942, -7.9962049, -0.7135725, 0.6962414
4: -12.9534883, -11.8476171, -12.9534912, -11.8476171, -0.7292633, 0.7410328
5: -5.7369833, -4.8009644, -5.7369823, -4.8009629, -0.6625171, 0.6506007
6: -3.2953453, -2.4089823, -3.2953467, -2.4089799, -0.6813226, 0.6700141
7: -8.3850708, -7.5282464, -8.3850746, -7.5282459, -0.6697860, 0.6701024
8: -3.7184329, -2.8380446, -3.7184319, -2.8380432, -0.6018405, 0.5934880
9: -2.2429178, -1.3550463, -2.2429171, -1.3550451, -0.6852980, 0.6824443

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 4559
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3682556, upper bound: 0.3663255
time: 5.19 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684469, upper bound: 0.3684469
time: 7.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 33.78 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 33.78
Output dim: 0, lower bound: -0.3642947, upper bound: 0.3680086
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 33.78
Output dim: 0, lower bound: -0.3664147, upper bound: 0.3681999
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 33.78
Output dim: 0, lower bound: -0.3682556, upper bound: 0.3663255
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 33.78
Output dim: 0, lower bound: -0.3684469, upper bound: 0.3684469

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 8.1628799, 8.9515991, 8.1601448, 8.9510698, -0.5332370, 0.5360706
1: -14.2884712, -13.0409288, -14.2952337, -13.0338840, -0.8164310, 0.8137980
2: -7.3077374, -6.4228230, -7.3175082, -6.4112039, -0.6693864, 0.6675501
3: -8.9354639, -8.0195723, -8.9492798, -8.0057468, -0.6878784, 0.6883042
4: -12.9361115, -11.8586912, -12.9444838, -11.8492222, -0.7262444, 0.7245920
5: -5.7281637, -4.8135262, -5.7343426, -4.8062196, -0.6467080, 0.6478848
6: -3.2788358, -2.4318542, -3.2810836, -2.4314783, -0.6406543, 0.6465738
7: -8.3744698, -7.5316305, -8.3748674, -7.5326114, -0.6576605, 0.6551208
8: -3.7114725, -2.8528666, -3.7126932, -2.8509865, -0.5818226, 0.5832026
9: -2.2396677, -1.3633940, -2.2329006, -1.3685484, -0.6642933, 0.6626501

Time for backsubstitution: 21.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4559
type: A, layer: 1, pos: 4559
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6126

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 554

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3642947, upper bound: 0.3662257
time: 3.82 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3642947, upper bound: 0.3680086
time: 5.33 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 8.1623487, 8.9533844, 8.1550941, 8.9560719, -0.5337007, 0.5431616
1: -14.2891712, -13.0395498, -14.2996893, -13.0300436, -0.8169718, 0.8196652
2: -7.3101616, -6.4226933, -7.3243809, -6.4066458, -0.6764040, 0.6672201
3: -8.9358158, -8.0191717, -8.9515724, -8.0042725, -0.6949086, 0.6903677
4: -12.9372425, -11.8586636, -12.9479046, -11.8476954, -0.7291193, 0.7261384
5: -5.7288485, -4.8133402, -5.7367859, -4.8052545, -0.6484103, 0.6512859
6: -3.2793474, -2.4259734, -3.2933483, -2.4147787, -0.6406596, 0.6650224
7: -8.3770475, -7.5316186, -8.3825598, -7.5282488, -0.6647263, 0.6570988
8: -3.7115660, -2.8498096, -3.7182918, -2.8420162, -0.5815964, 0.5918765
9: -2.2398992, -1.3590420, -2.2424440, -1.3563327, -0.6643152, 0.6773763

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 4559
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 4559
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6126

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 554

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3664148, upper bound: 0.3664169
time: 4.36 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3664147, upper bound: 0.3681999
time: 6.00 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 8.1566572, 8.9510965, 8.1521378, 8.9543161, -0.5397484, 0.5445268
1: -14.3003178, -13.0338840, -14.3040924, -13.0314226, -0.8154740, 0.8306999
2: -7.3252721, -6.4110923, -7.3297257, -6.4066586, -0.6645379, 0.6849856
3: -8.9498863, -7.9976850, -8.9518280, -7.9966097, -0.7085571, 0.6915212
4: -12.9500637, -11.8491430, -12.9523630, -11.8476467, -0.7258263, 0.7370145
5: -5.7345405, -4.8019290, -5.7362976, -4.8011498, -0.6599946, 0.6475189
6: -3.2830997, -2.4256935, -3.2948546, -2.4148588, -0.6618803, 0.6527650
7: -8.3773794, -7.5326114, -8.3824987, -7.5282593, -0.6619139, 0.6630442
8: -3.7128334, -2.8470163, -3.7183404, -2.8410950, -0.5913959, 0.5844743
9: -2.2333632, -1.3672783, -2.2426724, -1.3594012, -0.6698384, 0.6698275

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 4559
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6126

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 554

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3680077, upper bound: 0.3642948
time: 4.59 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3680081, upper bound: 0.3642948
time: 6.21 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 8.1516085, 8.9560966, 8.1516056, 8.9561014, -0.5467811, 0.5450046
1: -14.3047714, -13.0300436, -14.3047791, -13.0300398, -0.8213658, 0.8312333
2: -7.3321428, -6.4065304, -7.3321505, -6.4065285, -0.6642118, 0.6894119
3: -8.9521942, -7.9962072, -8.9521952, -7.9962049, -0.7097409, 0.6985481
4: -12.9534855, -11.8476181, -12.9534912, -11.8476171, -0.7273755, 0.7388759
5: -5.7369804, -4.8009658, -5.7369833, -4.8009634, -0.6634130, 0.6491928
6: -3.2953453, -2.4090002, -3.2953463, -2.4089837, -0.6736286, 0.6528037
7: -8.3850632, -7.5282454, -8.3850708, -7.5282459, -0.6639023, 0.6701007
8: -3.7184319, -2.8380537, -3.7184324, -2.8380446, -0.5965590, 0.5842447
9: -2.2429171, -1.3550594, -2.2429161, -1.3550487, -0.6845779, 0.6698449

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: A, layer: 1, pos: 4559
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6126

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 554

## Relational analysis of NS_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4559

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3681164, upper bound: 0.3663230
time: 4.78 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684420, upper bound: 0.3684422
time: 8.24 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 44.13 seconds
NS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 44.13
Output dim: 0, lower bound: -0.3642947, upper bound: 0.3662257
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 44.13
Output dim: 0, lower bound: -0.3642947, upper bound: 0.3680086
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 44.13
Output dim: 0, lower bound: -0.3664148, upper bound: 0.3664169
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 44.13
Output dim: 0, lower bound: -0.3664147, upper bound: 0.3681999
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 44.13
Output dim: 0, lower bound: -0.3680077, upper bound: 0.3642948
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 44.13
Output dim: 0, lower bound: -0.3680081, upper bound: 0.3642948
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 44.13
Output dim: 0, lower bound: -0.3681164, upper bound: 0.3663230
NS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 44.13
Output dim: 0, lower bound: -0.3684420, upper bound: 0.3684422

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: 8.1628799, 8.9515991, 8.1566677, 8.9510956, -0.5324705, 0.5403950
1: -14.2884712, -13.0409288, -14.3003006, -13.0338840, -0.8152819, 0.8195217
2: -7.3077374, -6.4228230, -7.3252726, -6.4110932, -0.6629255, 0.6735291
3: -8.9354639, -8.0195723, -8.9498768, -7.9976835, -0.6941619, 0.6853797
4: -12.9361115, -11.8586912, -12.9500608, -11.8492270, -0.7206492, 0.7286119
5: -5.7281637, -4.8135262, -5.7345376, -4.8019285, -0.6512241, 0.6476414
6: -3.2788358, -2.4318542, -3.2830853, -2.4256935, -0.6473670, 0.6448221
7: -8.3744698, -7.5316305, -8.3773775, -7.5326920, -0.6543465, 0.6581521
8: -3.7114725, -2.8528666, -3.7128062, -2.8470154, -0.5860772, 0.5796447
9: -2.2396677, -1.3633940, -2.2333624, -1.3672953, -0.6658912, 0.6617641

Time for backsubstitution: 20.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4559
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4559

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3639657, upper bound: 0.3658872
time: 4.60 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3642905, upper bound: 0.3680032
time: 5.59 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: 8.1623487, 8.9533844, 8.1623497, 8.9533825, -0.5302272, 0.5353546
1: -14.2891712, -13.0395498, -14.2891693, -13.0395546, -0.8063068, 0.8098764
2: -7.3101616, -6.4226933, -7.3101568, -6.4226933, -0.6602082, 0.6530991
3: -8.9358158, -8.0191717, -8.9358149, -8.0191727, -0.6798625, 0.6746008
4: -12.9372425, -11.8586636, -12.9372396, -11.8586636, -0.7169566, 0.7158325
5: -5.7288485, -4.8133402, -5.7288465, -4.8133426, -0.6405525, 0.6428258
6: -3.2793474, -2.4259734, -3.2793465, -2.4259882, -0.6303692, 0.6475768
7: -8.3770475, -7.5316186, -8.3770437, -7.5316186, -0.6581101, 0.6522276
8: -3.7115660, -2.8498096, -3.7115655, -2.8498135, -0.5742698, 0.5835118
9: -2.2398992, -1.3590420, -2.2398987, -1.3590522, -0.6619091, 0.6737654

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4559
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 4559
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6126

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4559

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3642938, upper bound: 0.3660874
time: 4.49 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3664095, upper bound: 0.3664121
time: 4.34 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: 8.1623487, 8.9533844, 8.1516190, 8.9560986, -0.5329344, 0.5474949
1: -14.2891712, -13.0395498, -14.3047581, -13.0300436, -0.8158231, 0.8253958
2: -7.3101616, -6.4226933, -7.3321433, -6.4065332, -0.6673524, 0.6730795
3: -8.9358158, -8.0191717, -8.9521847, -7.9962082, -0.6984260, 0.6865623
4: -12.9372425, -11.8586636, -12.9534864, -11.8477020, -0.7225099, 0.7297506
5: -5.7288485, -4.8133402, -5.7369776, -4.8009653, -0.6529212, 0.6510406
6: -3.2793474, -2.4259734, -3.2953305, -2.4089997, -0.6473680, 0.6565778
7: -8.3770475, -7.5316186, -8.3850632, -7.5283265, -0.6614122, 0.6601274
8: -3.7115660, -2.8498096, -3.7184072, -2.8380527, -0.5858471, 0.5848186
9: -2.2398992, -1.3590420, -2.2429166, -1.3550782, -0.6659126, 0.6765053

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4559
type: A, layer: 1, pos: 4559
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6126

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3642947, upper bound: 0.3660788
time: 5.82 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3642946, upper bound: 0.3660794
time: 3.80 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: 8.1566677, 8.9510956, 8.1628799, 8.9515991, -0.5403950, 0.5324705
1: -14.3003006, -13.0338840, -14.2884712, -13.0409288, -0.8195219, 0.8152816
2: -7.3252726, -6.4110932, -7.3077374, -6.4228230, -0.6735289, 0.6629255
3: -8.9498768, -7.9976835, -8.9354639, -8.0195723, -0.6853800, 0.6941619
4: -12.9500608, -11.8492270, -12.9361115, -11.8586912, -0.7286119, 0.7206490
5: -5.7345376, -4.8019285, -5.7281637, -4.8135262, -0.6476412, 0.6512241
6: -3.2830853, -2.4256935, -3.2788358, -2.4318542, -0.6448221, 0.6473670
7: -8.3773775, -7.5326920, -8.3744698, -7.5316305, -0.6581521, 0.6543465
8: -3.7128062, -2.8470154, -3.7114725, -2.8528666, -0.5796447, 0.5860770
9: -2.2333624, -1.3672953, -2.2396677, -1.3633940, -0.6617637, 0.6658912

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4559
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6126

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 4559

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3658868, upper bound: 0.3639662
time: 4.32 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3680026, upper bound: 0.3642910
time: 4.59 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: 8.1566572, 8.9510965, 8.1521425, 8.9543161, -0.5397484, 0.5411959
1: -14.3003178, -13.0338840, -14.3040848, -13.0314226, -0.8154740, 0.8172500
2: -7.3252721, -6.4110923, -7.3297224, -6.4066577, -0.6645377, 0.6643076
3: -8.9498863, -7.9976850, -8.9518280, -7.9966102, -0.6912262, 0.6915212
4: -12.9500637, -11.8491430, -12.9523602, -11.8476477, -0.7258258, 0.7256250
5: -5.7345405, -4.8019290, -5.7362967, -4.8011522, -0.6480770, 0.6475189
6: -3.2830997, -2.4256935, -3.2948532, -2.4148614, -0.6515346, 0.6527634
7: -8.3773794, -7.5326114, -8.3824968, -7.5282593, -0.6619134, 0.6627157
8: -3.7128334, -2.8470163, -3.7183385, -2.8410969, -0.5848112, 0.5844741
9: -2.2333632, -1.3672783, -2.2426727, -1.3594018, -0.6698365, 0.6726816

Time for backsubstitution: 22.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4559
type: A, layer: 1, pos: 4559
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6126

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660793, upper bound: 0.3647218
time: 5.62 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660793, upper bound: 0.3642952
time: 4.72 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: 8.1553726, 8.9493256, 8.1522722, 8.9539661, -0.5406730, 0.5374939
1: -14.2999620, -13.0343628, -14.3035746, -13.0314121, -0.8145342, 0.8256106
2: -7.3213434, -6.4099207, -7.3287582, -6.4068327, -0.6531806, 0.6793981
3: -8.9507132, -7.9983363, -8.9518147, -7.9968290, -0.7066994, 0.6953645
4: -12.9506855, -11.8549652, -12.9531908, -11.8499432, -0.7223186, 0.7312567
5: -5.7345657, -4.8095150, -5.7367926, -4.8036289, -0.6560676, 0.6404645
6: -3.2896085, -2.4236045, -3.2947812, -2.4135680, -0.6581769, 0.6376405
7: -8.3721867, -7.5309596, -8.3810930, -7.5282555, -0.6507320, 0.6616457
8: -3.7146378, -2.8506050, -3.7182603, -2.8419847, -0.5845423, 0.5714836
9: -2.2359264, -1.3785794, -2.2427425, -1.3624734, -0.6651974, 0.6460571

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 4559

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 554

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of NS_A2_A2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3659908, upper bound: 0.3661257
time: 5.18 seconds

## Relational analysis of NS_A2_A2_A1_B2

### Relational analysis result of NS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3659908, upper bound: 0.3663239
time: 4.67 seconds

## BFS NS instance: NS_A2_A2_A2

### Backsubstitution after applying NS history:
0: 8.1516104, 8.9560928, 8.1516056, 8.9561014, -0.5467801, 0.5405381
1: -14.3047705, -13.0300465, -14.3047791, -13.0300398, -0.8209352, 0.8293309
2: -7.3321366, -6.4065294, -7.3321505, -6.4065285, -0.6570947, 0.6871767
3: -8.9521933, -7.9962082, -8.9521952, -7.9962049, -0.7090991, 0.7020128
4: -12.9534855, -11.8476210, -12.9534912, -11.8476171, -0.7273755, 0.7343152
5: -5.7369804, -4.8009696, -5.7369833, -4.8009634, -0.6620333, 0.6441045
6: -3.2953444, -2.4090061, -3.2953463, -2.4089837, -0.6699078, 0.6403449
7: -8.3850555, -7.5282454, -8.3850708, -7.5282459, -0.6546869, 0.6696095
8: -3.7184310, -2.8380632, -3.7184324, -2.8380446, -0.5933709, 0.5734725
9: -2.2429173, -1.3550704, -2.2429161, -1.3550487, -0.6816833, 0.6496267

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 4559

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 554

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of NS_A2_A2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3663207, upper bound: 0.3682506
time: 6.72 seconds

## Relational analysis of NS_A2_A2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3663207, upper bound: 0.3684426
time: 3.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 32.56 seconds
NS_A1_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3639657, upper bound: 0.3658872
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3642905, upper bound: 0.3680032
NS_A1_B2_B1_B1, status: Status.VERIFIED, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3642938, upper bound: 0.3660874
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3664095, upper bound: 0.3664121
NS_A1_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3642947, upper bound: 0.3660788
NS_A1_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3642946, upper bound: 0.3660794
NS_A2_A1_B1_B1, status: Status.VERIFIED, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3658868, upper bound: 0.3639662
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3680026, upper bound: 0.3642910
NS_A2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3660793, upper bound: 0.3647218
NS_A2_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3660793, upper bound: 0.3642952
NS_A2_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3659908, upper bound: 0.3661257
NS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3659908, upper bound: 0.3663239
NS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3663207, upper bound: 0.3682506
NS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 0, lower bound: -0.3663207, upper bound: 0.3684426

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: 8.1628799, 8.9515953, 8.1566677, 8.9510956, -0.5324695, 0.5359251
1: -14.2884703, -13.0409317, -14.3003006, -13.0338840, -0.8148556, 0.8175967
2: -7.3077307, -6.4228234, -7.3252726, -6.4110932, -0.6558082, 0.6712935
3: -8.9354649, -8.0195732, -8.9498768, -7.9976835, -0.6935225, 0.6880524
4: -12.9361124, -11.8586912, -12.9500608, -11.8492270, -0.7190237, 0.7240515
5: -5.7281632, -4.8135309, -5.7345376, -4.8019285, -0.6512241, 0.6425524
6: -3.2788339, -2.4318619, -3.2830853, -2.4256935, -0.6470451, 0.6322966
7: -8.3744612, -7.5316315, -8.3773775, -7.5326920, -0.6451268, 0.6580462
8: -3.7114716, -2.8528767, -3.7128062, -2.8470154, -0.5832534, 0.5688355
9: -2.2396681, -1.3634032, -2.2333624, -1.3672953, -0.6655955, 0.6415839

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 4559

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of NS_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3642905, upper bound: 0.3660740
time: 5.46 seconds

## Relational analysis of NS_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3642905, upper bound: 0.3680025
time: 6.19 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: 8.1623487, 8.9533844, 8.1623497, 8.9533777, -0.5257599, 0.5353529
1: -14.2891712, -13.0395498, -14.2891665, -13.0395546, -0.8044004, 0.8094471
2: -7.3101616, -6.4226933, -7.3101535, -6.4226956, -0.6602077, 0.6459837
3: -8.9358158, -8.0191717, -8.9358139, -8.0191746, -0.6833384, 0.6739702
4: -12.9372425, -11.8586636, -12.9372406, -11.8586626, -0.7123990, 0.7158318
5: -5.7288485, -4.8133402, -5.7288465, -4.8133454, -0.6354632, 0.6428263
6: -3.2793474, -2.4259734, -3.2793469, -2.4259942, -0.6179116, 0.6457586
7: -8.3770475, -7.5316186, -8.3770351, -7.5316176, -0.6580043, 0.6430120
8: -3.7115660, -2.8498096, -3.7115645, -2.8498235, -0.5634983, 0.5811582
9: -2.2398992, -1.3590420, -2.2398996, -1.3590616, -0.6417046, 0.6730330

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 4558
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4559

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4558

## Relational analysis of NS_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3658926, upper bound: 0.3650837
time: 6.45 seconds

## Relational analysis of NS_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3664097, upper bound: 0.3664104
time: 4.32 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: 8.1566677, 8.9510956, 8.1628799, 8.9515953, -0.5359251, 0.5324695
1: -14.3003006, -13.0338840, -14.2884703, -13.0409317, -0.8175964, 0.8148553
2: -7.3252726, -6.4110932, -7.3077307, -6.4228234, -0.6712937, 0.6558082
3: -8.9498768, -7.9976835, -8.9354649, -8.0195732, -0.6880527, 0.6935225
4: -12.9500608, -11.8492270, -12.9361124, -11.8586912, -0.7240515, 0.7190237
5: -5.7345376, -4.8019285, -5.7281632, -4.8135309, -0.6425524, 0.6512239
6: -3.2830853, -2.4256935, -3.2788339, -2.4318619, -0.6322966, 0.6470451
7: -8.3773775, -7.5326920, -8.3744612, -7.5316315, -0.6580462, 0.6451271
8: -3.7128062, -2.8470154, -3.7114716, -2.8528767, -0.5688355, 0.5832534
9: -2.2333624, -1.3672953, -2.2396681, -1.3634032, -0.6415844, 0.6655958

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4558
type: A, layer: 1, pos: 4558
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4559

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of NS_A2_A1_B1_B2_B1

### Relational analysis result of NS_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660739, upper bound: 0.3642910
time: 5.71 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2

### Relational analysis result of NS_A2_A1_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660739, upper bound: 0.3642904
time: 6.22 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 60.78 + 543.01 = 603.79 seconds
