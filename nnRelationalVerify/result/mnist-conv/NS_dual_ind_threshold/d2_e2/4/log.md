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
execution time: IAR + RelationalAnalysis = 22.75 + 36.54 = 59.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.3684516, upper bound: 0.3684522

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 4559
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 554

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3664168, upper bound: 0.3682010
time: 5.61 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684485, upper bound: 0.3684490
time: 5.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 11.07 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 11.07
Output dim: 0, lower bound: -0.3664168, upper bound: 0.3682010
NS_A2, status: Status.UNKNOWN, split count: 1, time: 11.07
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

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 4558

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3642947, upper bound: 0.3680086
time: 5.15 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3664147, upper bound: 0.3681999
time: 7.04 seconds

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

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 4558

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3663249, upper bound: 0.3682565
time: 6.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3684465, upper bound: 0.3684478
time: 4.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 33.37 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 33.37
Output dim: 0, lower bound: -0.3642947, upper bound: 0.3680086
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 33.37
Output dim: 0, lower bound: -0.3664147, upper bound: 0.3681999
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 33.37
Output dim: 0, lower bound: -0.3663249, upper bound: 0.3682565
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 33.37
Output dim: 0, lower bound: -0.3684465, upper bound: 0.3684478

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

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4559
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4559

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3639657, upper bound: 0.3658872
time: 4.87 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3642905, upper bound: 0.3680031
time: 5.24 seconds

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

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 4559
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4559

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660852, upper bound: 0.3660789
time: 5.22 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3664099, upper bound: 0.3681946
time: 3.90 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 8.1521425, 8.9543161, 8.1566553, 8.9510956, -0.5411961, 0.5430315
1: -14.3040848, -13.0314226, -14.3003235, -13.0338840, -0.8172507, 0.8289301
2: -7.3297224, -6.4066577, -7.3252764, -6.4110913, -0.6643071, 0.6866012
3: -8.9518280, -7.9966102, -8.9498863, -7.9976840, -0.7088592, 0.6912260
4: -12.9523602, -11.8476477, -12.9500656, -11.8491430, -0.7256255, 0.7375982
5: -5.7362967, -4.8011522, -5.7345400, -4.8019261, -0.6594181, 0.6480768
6: -3.2948532, -2.4148614, -3.2830997, -2.4256907, -0.6641116, 0.6515357
7: -8.3824968, -7.5282593, -8.3773785, -7.5326109, -0.6627159, 0.6622405
8: -3.7183385, -2.8410969, -3.7128329, -2.8470130, -0.5928192, 0.5848122
9: -2.2426727, -1.3594018, -2.2333608, -1.3672751, -0.6726828, 0.6669834

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4559
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4559

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3659908, upper bound: 0.3661261
time: 5.05 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3663207, upper bound: 0.3682510
time: 7.00 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 8.1516075, 8.9561024, 8.1516056, 8.9560986, -0.5416536, 0.5501316
1: -14.3047724, -13.0300398, -14.3047800, -13.0300436, -0.8177948, 0.8340883
2: -7.3321438, -6.4065304, -7.3321495, -6.4065299, -0.6713250, 0.6861489
3: -8.9521942, -7.9962068, -8.9521952, -7.9962063, -0.7131128, 0.6933115
4: -12.9534883, -11.8476162, -12.9534874, -11.8476181, -0.7285018, 0.7387354
5: -5.7369828, -4.8009644, -5.7369809, -4.8009639, -0.6611094, 0.6514981
6: -3.2953453, -2.4089861, -3.2953448, -2.4089971, -0.6641109, 0.6700096
7: -8.3850689, -7.5282464, -8.3850651, -7.5282454, -0.6697841, 0.6642172
8: -3.7184324, -2.8380466, -3.7184319, -2.8380508, -0.5924377, 0.5934865
9: -2.2429159, -1.3550496, -2.2429171, -1.3550593, -0.6726985, 0.6817222

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 4559
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3682557, upper bound: 0.3663253
time: 4.94 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3682556, upper bound: 0.3684473
time: 4.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.76 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 31.76
Output dim: 0, lower bound: -0.3639657, upper bound: 0.3658872
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.76
Output dim: 0, lower bound: -0.3642905, upper bound: 0.3680031
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 31.76
Output dim: 0, lower bound: -0.3660852, upper bound: 0.3660789
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.76
Output dim: 0, lower bound: -0.3664099, upper bound: 0.3681946
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 31.76
Output dim: 0, lower bound: -0.3659908, upper bound: 0.3661261
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.76
Output dim: 0, lower bound: -0.3663207, upper bound: 0.3682510
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.76
Output dim: 0, lower bound: -0.3682557, upper bound: 0.3663253
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.76
Output dim: 0, lower bound: -0.3682556, upper bound: 0.3684473

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 8.1628799, 8.9515953, 8.1601448, 8.9510698, -0.5332363, 0.5316005
1: -14.2884703, -13.0409317, -14.2952337, -13.0338840, -0.8160048, 0.8118801
2: -7.3077307, -6.4228234, -7.3175082, -6.4112039, -0.6622796, 0.6675501
3: -8.9354649, -8.0195732, -8.9492798, -8.0057468, -0.6872394, 0.6917787
4: -12.9361124, -11.8586912, -12.9444838, -11.8492222, -0.7262435, 0.7200365
5: -5.7281632, -4.8135309, -5.7343426, -4.8062196, -0.6467075, 0.6427963
6: -3.2788339, -2.4318619, -3.2810836, -2.4314783, -0.6406538, 0.6341178
7: -8.3744612, -7.5316315, -8.3748674, -7.5326114, -0.6484404, 0.6550150
8: -3.7114716, -2.8528767, -3.7126932, -2.8509865, -0.5818226, 0.5724320
9: -2.2396681, -1.3634032, -2.2329006, -1.3685484, -0.6639986, 0.6424785

Time for backsubstitution: 22.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 4558

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4559

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3621720, upper bound: 0.3676786
time: 5.31 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3621720, upper bound: 0.3680040
time: 3.99 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 8.1623497, 8.9533806, 8.1550941, 8.9560719, -0.5336993, 0.5386977
1: -14.2891674, -13.0395527, -14.2996893, -13.0300436, -0.8165417, 0.8177748
2: -7.3101540, -6.4226952, -7.3243809, -6.4066458, -0.6693065, 0.6672201
3: -8.9358139, -8.0191746, -8.9515724, -8.0042725, -0.6942790, 0.6938508
4: -12.9372435, -11.8586636, -12.9479046, -11.8476954, -0.7291188, 0.7215879
5: -5.7288480, -4.8133459, -5.7367859, -4.8052545, -0.6484094, 0.6462047
6: -3.2793469, -2.4259803, -3.2933483, -2.4147787, -0.6406581, 0.6525869
7: -8.3770409, -7.5316191, -8.3825598, -7.5282488, -0.6555185, 0.6569934
8: -3.7115650, -2.8498182, -3.7182918, -2.8420162, -0.5815959, 0.5811219
9: -2.2398992, -1.3590517, -2.2424440, -1.3563327, -0.6640186, 0.6572278

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 4558

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4559

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3642936, upper bound: 0.3678699
time: 5.21 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3642936, upper bound: 0.3678703
time: 4.64 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 8.1521435, 8.9543104, 8.1566553, 8.9510956, -0.5411954, 0.5385621
1: -14.3040829, -13.0314207, -14.3003235, -13.0338840, -0.8168235, 0.8270044
2: -7.3297172, -6.4066591, -7.3252764, -6.4110913, -0.6572006, 0.6843650
3: -8.9518280, -7.9966092, -8.9498863, -7.9976840, -0.7082098, 0.6946874
4: -12.9523582, -11.8476467, -12.9500656, -11.8491430, -0.7256250, 0.7330377
5: -5.7362976, -4.8011570, -5.7345400, -4.8019261, -0.6588182, 0.6429880
6: -3.2948527, -2.4148688, -3.2830997, -2.4256907, -0.6629450, 0.6390798
7: -8.3824883, -7.5282602, -8.3773785, -7.5326109, -0.6534967, 0.6621356
8: -3.7183399, -2.8411083, -3.7128329, -2.8470130, -0.5896297, 0.5740409
9: -2.2426710, -1.3594124, -2.2333608, -1.3672751, -0.6723900, 0.6467979

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 4558

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 554

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660743, upper bound: 0.3662193
time: 4.07 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3660747, upper bound: 0.3666470
time: 6.79 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 8.1566572, 8.9510965, 8.1516056, 8.9560986, -0.5415287, 0.5451396
1: -14.3003178, -13.0338840, -14.3047800, -13.0300436, -0.8168516, 0.8302374
2: -7.3252721, -6.4110923, -7.3321495, -6.4065299, -0.6644394, 0.6851616
3: -8.9498863, -7.9976850, -8.9521952, -7.9962063, -0.7061520, 0.6898601
4: -12.9500637, -11.8491430, -12.9534874, -11.8476181, -0.7250938, 0.7371733
5: -5.7345405, -4.8019290, -5.7369809, -4.8009639, -0.6587801, 0.6482341
6: -3.2830997, -2.4256935, -3.2953448, -2.4089971, -0.6620836, 0.6533332
7: -8.3773794, -7.5326114, -8.3850651, -7.5282454, -0.6619287, 0.6657283
8: -3.7128334, -2.8470163, -3.7184319, -2.8380508, -0.5916805, 0.5845675
9: -2.2333632, -1.3672783, -2.2429171, -1.3550593, -0.6749125, 0.6694937

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 4558

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 554

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660784, upper bound: 0.3642949
time: 5.41 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660788, upper bound: 0.3642952
time: 3.65 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 8.1516085, 8.9560966, 8.1516056, 8.9560986, -0.5416529, 0.5450032
1: -14.3047714, -13.0300436, -14.3047800, -13.0300436, -0.8177943, 0.8312321
2: -7.3321428, -6.4065304, -7.3321495, -6.4065299, -0.6642108, 0.6871715
3: -8.9521942, -7.9962072, -8.9521952, -7.9962063, -0.7156162, 0.6985452
4: -12.9534855, -11.8476181, -12.9534874, -11.8476181, -0.7273741, 0.7406609
5: -5.7369804, -4.8009658, -5.7369809, -4.8009639, -0.6634121, 0.6514959
6: -3.2953453, -2.4090002, -3.2953448, -2.4089971, -0.6641104, 0.6528025
7: -8.3850632, -7.5282454, -8.3850651, -7.5282454, -0.6639013, 0.6642182
8: -3.7184319, -2.8380537, -3.7184319, -2.8380508, -0.5927567, 0.5842440
9: -2.2429171, -1.3550594, -2.2429171, -1.3550593, -0.6726980, 0.6698439

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 4559
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 4558

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 554

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660784, upper bound: 0.3642952
time: 3.75 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3660788, upper bound: 0.3668442
time: 6.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 32.64 seconds
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.64
Output dim: 0, lower bound: -0.3621720, upper bound: 0.3676786
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.64
Output dim: 0, lower bound: -0.3621720, upper bound: 0.3680040
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.64
Output dim: 0, lower bound: -0.3642936, upper bound: 0.3678699
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.64
Output dim: 0, lower bound: -0.3642936, upper bound: 0.3678703
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 32.64
Output dim: 0, lower bound: -0.3660743, upper bound: 0.3662193
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.64
Output dim: 0, lower bound: -0.3660747, upper bound: 0.3666470
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 32.64
Output dim: 0, lower bound: -0.3660784, upper bound: 0.3642949
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 32.64
Output dim: 0, lower bound: -0.3660788, upper bound: 0.3642952
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 32.64
Output dim: 0, lower bound: -0.3660784, upper bound: 0.3642952
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.64
Output dim: 0, lower bound: -0.3660788, upper bound: 0.3668442

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 8.1628799, 8.9515953, 8.1639376, 8.9442978, -0.5264676, 0.5316133
1: -14.2884703, -13.0409317, -14.2903557, -13.0382042, -0.8116822, 0.8083353
2: -7.3077307, -6.4228234, -7.3067198, -6.4146061, -0.6662588, 0.6567202
3: -8.9354649, -8.0195732, -8.9477806, -8.0078506, -0.6844914, 0.6852930
4: -12.9361124, -11.8586912, -12.9416914, -11.8565655, -0.7189002, 0.7207878
5: -5.7281632, -4.8135309, -5.7319331, -4.8147631, -0.6380887, 0.6456645
6: -3.2788339, -2.4318619, -3.2752819, -2.4460855, -0.6261926, 0.6399598
7: -8.3744612, -7.5316315, -8.3620033, -7.5353355, -0.6548271, 0.6418877
8: -3.7114716, -2.8528767, -3.7088766, -2.8635473, -0.5692463, 0.5758576
9: -2.2396681, -1.3634032, -2.2258835, -1.3920757, -0.6404610, 0.6520596

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3621720, upper bound: 0.3657495
time: 6.12 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3621720, upper bound: 0.3676786
time: 8.46 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 8.1628799, 8.9515953, 8.1601439, 8.9510651, -0.5287693, 0.5316002
1: -14.2884703, -13.0409317, -14.2952328, -13.0338869, -0.8145099, 0.8118782
2: -7.3077307, -6.4228234, -7.3175039, -6.4112053, -0.6622796, 0.6604490
3: -8.9354649, -8.0195732, -8.9492779, -8.0057478, -0.6913507, 0.6917772
4: -12.9361124, -11.8586912, -12.9444838, -11.8492231, -0.7216883, 0.7200356
5: -5.7281632, -4.8135309, -5.7343431, -4.8062229, -0.6416230, 0.6427960
6: -3.2788339, -2.4318619, -3.2810836, -2.4314814, -0.6282053, 0.6341176
7: -8.3744612, -7.5316315, -8.3748627, -7.5326128, -0.6484404, 0.6459048
8: -3.7114716, -2.8528767, -3.7126927, -2.8509932, -0.5710573, 0.5724320
9: -2.2396681, -1.3634032, -2.2328999, -1.3685544, -0.6441269, 0.6424768

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3621720, upper bound: 0.3657504
time: 4.51 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3621720, upper bound: 0.3680035
time: 4.25 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 8.1623497, 8.9533806, 8.1588612, 8.9492989, -0.5269327, 0.5369172
1: -14.2891674, -13.0395527, -14.2948761, -13.0343628, -0.8122206, 0.8142581
2: -7.3101540, -6.4226952, -7.3135762, -6.4100351, -0.6731982, 0.6563940
3: -8.9358139, -8.0191746, -8.9500847, -8.0063992, -0.6915607, 0.6873927
4: -12.9372435, -11.8586636, -12.9451075, -11.8550453, -0.7217727, 0.7219279
5: -5.7288480, -4.8133459, -5.7343707, -4.8137984, -0.6397829, 0.6490729
6: -3.2793469, -2.4259803, -3.2875881, -2.4293890, -0.6261871, 0.6517422
7: -8.3770409, -7.5316191, -8.3696747, -7.5309596, -0.6602812, 0.6438606
8: -3.7115650, -2.8498182, -3.7144985, -2.8545747, -0.5690222, 0.5810342
9: -2.2398992, -1.3590517, -2.2354610, -1.3798518, -0.6404829, 0.6604679

Time for backsubstitution: 21.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6127

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3639663, upper bound: 0.3663685
time: 4.22 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4558

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3629469, upper bound: 0.3673442
time: 4.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3642924, upper bound: 0.3678688
time: 4.17 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 8.1623497, 8.9533806, 8.1550970, 8.9560671, -0.5292325, 0.5386965
1: -14.2891674, -13.0395527, -14.2996883, -13.0300465, -0.8150630, 0.8177717
2: -7.3101540, -6.4226952, -7.3243742, -6.4066448, -0.6693065, 0.6601033
3: -8.9358139, -8.0191746, -8.9515724, -8.0042725, -0.6983821, 0.6938498
4: -12.9372435, -11.8586636, -12.9479065, -11.8476973, -0.7245612, 0.7215877
5: -5.7288480, -4.8133459, -5.7367859, -4.8052597, -0.6433215, 0.6462045
6: -3.2793469, -2.4259803, -3.2933486, -2.4147849, -0.6282005, 0.6525855
7: -8.3770409, -7.5316191, -8.3825521, -7.5282483, -0.6555185, 0.6478837
8: -3.7115650, -2.8498182, -3.7182932, -2.8420267, -0.5708237, 0.5811214
9: -2.2398992, -1.3590517, -2.2424440, -1.3563424, -0.6441026, 0.6572270

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6127

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3639662, upper bound: 0.3666960
time: 3.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.3642916, upper bound: 0.3681929
time: 3.90 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 8.1521435, 8.9543104, 8.1566572, 8.9510965, -0.5411949, 0.5352786
1: -14.3040829, -13.0314207, -14.3003178, -13.0338840, -0.8168230, 0.8135509
2: -7.3297172, -6.4066591, -7.3252721, -6.4110923, -0.6572001, 0.6645374
3: -8.9518280, -7.9966092, -8.9498863, -7.9976850, -0.6908715, 0.6946874
4: -12.9523582, -11.8476467, -12.9500637, -11.8491430, -0.7256246, 0.7212696
5: -5.7362976, -4.8011570, -5.7345405, -4.8019290, -0.6475182, 0.6429875
6: -3.2948527, -2.4148688, -3.2830997, -2.4256935, -0.6527624, 0.6390786
7: -8.3824883, -7.5282602, -8.3773794, -7.5326114, -0.6534958, 0.6618078
8: -3.7183399, -2.8411083, -3.7128334, -2.8470163, -0.5844729, 0.5740409
9: -2.2426710, -1.3594124, -2.2333632, -1.3672783, -0.6723886, 0.6496515

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 4558

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660747, upper bound: 0.3647169
time: 5.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.3660747, upper bound: 0.3662192
time: 4.05 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.29 + 549.60 = 608.90 seconds
