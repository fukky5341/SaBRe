## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.0349644734999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.3545036, -2.6462827, -5.3545036, -2.6462827, -2.2198710, 2.2198710)
1: (-6.3247952, -4.2507443, -6.3247952, -4.2507443, -1.7507243, 1.7507243)
2: (-4.6537833, -2.6294413, -4.6537833, -2.6294413, -1.5917411, 1.5917413)
3: (-7.8594894, -5.0927763, -7.8594894, -5.0927763, -2.0713673, 2.0713677)
4: (-11.8233414, -9.0321541, -11.8233414, -9.0321541, -2.3726444, 2.3726444)
5: (-6.3656282, -4.2320991, -6.3656282, -4.2320991, -1.7345924, 1.7345924)
6: (-10.4613619, -7.9367504, -10.4613619, -7.9367504, -2.0136123, 2.0136118)
7: (-2.8968585, -0.7577722, -2.8968585, -0.7577722, -1.8201418, 1.8201420)
8: (1.9637957, 3.6150055, 1.9637957, 3.6150055, -1.3549571, 1.3549573)
9: (-8.0759621, -5.5572004, -8.0759621, -5.5572004, -2.0553026, 2.0553021)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.42 + 33.71 = 57.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.0401645, upper bound: 1.0401642

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 957

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0285205, upper bound: 1.0393477
time: 4.25 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0393471, upper bound: 1.0393484
time: 4.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.03 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.03
Output dim: 7, lower bound: -1.0285205, upper bound: 1.0393477
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.03
Output dim: 7, lower bound: -1.0393471, upper bound: 1.0393484

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5.3448629, -2.6513064, -5.3535991, -2.6487818, -2.2063532, 2.2138548
1: -6.3199863, -4.2584815, -6.3242135, -4.2543473, -1.7423882, 1.7425404
2: -4.6420040, -2.6378279, -4.6528454, -2.6335866, -1.5757818, 1.5829291
3: -7.8586378, -5.0940123, -7.8584652, -5.0932484, -2.0700850, 2.0685639
4: -11.8088398, -9.0452976, -11.8159161, -9.0326414, -2.3578911, 2.3510776
5: -6.3616781, -4.2332859, -6.3642983, -4.2322807, -1.7303658, 1.7315521
6: -10.4562159, -7.9401374, -10.4607811, -7.9384832, -2.0048246, 2.0096078
7: -2.8744102, -0.7741427, -2.8859327, -0.7579317, -1.7970166, 1.7905664
8: 1.9756560, 3.6014438, 1.9642277, 3.6082225, -1.3354609, 1.3401518
9: -8.0701504, -5.5636220, -8.0728464, -5.5576634, -2.0488563, 2.0413256

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0214434, upper bound: 1.0366213
time: 4.41 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0285087, upper bound: 1.0393364
time: 5.60 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5.3544674, -2.6463146, -5.3544798, -2.6463041, -2.2198105, 2.2110400
1: -6.3247867, -4.2507839, -6.3247857, -4.2507639, -1.7506957, 1.7443419
2: -4.6537824, -2.6294842, -4.6537824, -2.6294675, -1.5917215, 1.5871401
3: -7.8594294, -5.0936785, -7.8594561, -5.0931931, -2.0707273, 2.0737429
4: -11.8232851, -9.0321836, -11.8232985, -9.0321684, -2.3500996, 2.3721294
5: -6.3656182, -4.2321053, -6.3656225, -4.2321062, -1.7331562, 1.7344773
6: -10.4613619, -7.9368286, -10.4613628, -7.9367905, -2.0128613, 2.0066915
7: -2.8968077, -0.7577724, -2.8968346, -0.7577713, -1.7883425, 1.8193042
8: 1.9638104, 3.6149974, 1.9638042, 3.6150012, -1.3549359, 1.3355761
9: -8.0758963, -5.5572109, -8.0759287, -5.5572100, -2.0409937, 2.0527639

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5871

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0322701, upper bound: 1.0366220
time: 4.19 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0393353, upper bound: 1.0393373
time: 4.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 36.37 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 36.37
Output dim: 7, lower bound: -1.0214434, upper bound: 1.0366213
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 36.37
Output dim: 7, lower bound: -1.0285087, upper bound: 1.0393364
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 36.37
Output dim: 7, lower bound: -1.0322701, upper bound: 1.0366220
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 36.37
Output dim: 7, lower bound: -1.0393353, upper bound: 1.0393373

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -5.3409643, -2.6554296, -5.3431625, -2.6578724, -2.1891265, 2.1930790
1: -6.3055334, -4.2615047, -6.2938423, -4.2743654, -1.7075920, 1.7090783
2: -4.6407204, -2.6447191, -4.6447034, -2.6486073, -1.5582271, 1.5672739
3: -7.8552513, -5.1094189, -7.8339024, -5.1246109, -2.0354023, 2.0288162
4: -11.7988453, -9.0497608, -11.7948179, -9.0505409, -2.3290730, 2.3250842
5: -6.3602839, -4.2378111, -6.3581147, -4.2421269, -1.7173853, 1.7192791
6: -10.4347658, -7.9434323, -10.4170904, -7.9678245, -1.9518824, 1.9614477
7: -2.8690841, -0.7834506, -2.8658650, -0.7767360, -1.7726154, 1.7599337
8: 1.9768629, 3.5901785, 1.9795847, 3.5855446, -1.3110099, 1.3129172
9: -8.0652695, -5.5664706, -8.0587111, -5.5638528, -2.0348783, 2.0238047

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0157185, upper bound: 1.0365865
time: 4.24 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0214059, upper bound: 1.0365878
time: 4.59 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -5.3448601, -2.6513114, -5.3535914, -2.6487942, -2.2058253, 2.2121854
1: -6.3199663, -4.2584834, -6.3241682, -4.2543530, -1.7423606, 1.7228224
2: -4.6420031, -2.6378341, -4.6528420, -2.6336021, -1.5675325, 1.5827117
3: -7.8586326, -5.0940342, -7.8584566, -5.0932989, -2.0418677, 2.0685349
4: -11.8088255, -9.0452995, -11.8158817, -9.0326500, -2.3568592, 2.3364859
5: -6.3616772, -4.2332926, -6.3642960, -4.2322936, -1.7284441, 1.7315419
6: -10.4561853, -7.9401398, -10.4607182, -7.9384923, -2.0037155, 1.9654098
7: -2.8744054, -0.7741520, -2.8859212, -0.7579522, -1.7803535, 1.7898812
8: 1.9756579, 3.6014318, 1.9642310, 3.6081948, -1.3094327, 1.3401380
9: -8.0701466, -5.5636234, -8.0728340, -5.5576706, -2.0488358, 2.0445638

Time for backsubstitution: 21.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0227830, upper bound: 1.0393022
time: 4.49 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0284712, upper bound: 1.0393030
time: 4.24 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.3505669, -2.6504419, -5.3440361, -2.6553988, -2.2025824, 2.1902571
1: -6.3103333, -4.2538052, -6.2944183, -4.2707782, -1.7158813, 1.7108705
2: -4.6524954, -2.6363764, -4.6456337, -2.6444864, -1.5741673, 1.5714810
3: -7.8560462, -5.1090841, -7.8348961, -5.1245527, -2.0360522, 2.0339956
4: -11.8132906, -9.0366440, -11.8022003, -9.0500679, -2.3212757, 2.3461342
5: -6.3642244, -4.2366276, -6.3594379, -4.2419491, -1.7201767, 1.7222033
6: -10.4399147, -7.9401245, -10.4176731, -7.9661322, -1.9599142, 1.9585238
7: -2.8914824, -0.7670789, -2.8767552, -0.7765782, -1.7639365, 1.7886713
8: 1.9650187, 3.6037321, 1.9791636, 3.5923219, -1.3304846, 1.3083403
9: -8.0710106, -5.5600572, -8.0617857, -5.5633960, -2.0270147, 2.0352340

Time for backsubstitution: 21.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0265368, upper bound: 1.0365878
time: 4.76 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0322347, upper bound: 1.0365864
time: 4.40 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.3544636, -2.6463196, -5.3544717, -2.6463172, -2.2192845, 2.2093654
1: -6.3247662, -4.2507858, -6.3247404, -4.2507687, -1.7497756, 1.7246246
2: -4.6537790, -2.6294901, -4.6537790, -2.6294832, -1.5834723, 1.5869248
3: -7.8594265, -5.0936995, -7.8594494, -5.0932417, -2.0425096, 2.0737138
4: -11.8232718, -9.0321884, -11.8232679, -9.0321760, -2.3490677, 2.3575387
5: -6.3656158, -4.2321100, -6.3656187, -4.2321186, -1.7312341, 1.7344670
6: -10.4613323, -7.9368329, -10.4613008, -7.9367967, -2.0117512, 1.9624910
7: -2.8968019, -0.7577815, -2.8968217, -0.7577932, -1.7716784, 1.8186159
8: 1.9638119, 3.6149869, 1.9638057, 3.6149745, -1.3289082, 1.3355618
9: -8.0758896, -5.5572119, -8.0759144, -5.5572162, -2.0409737, 2.0559931

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 5856
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 5856

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0336009, upper bound: 1.0393030
time: 4.57 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0392998, upper bound: 1.0393031
time: 4.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.01 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 7, lower bound: -1.0157185, upper bound: 1.0365865
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 7, lower bound: -1.0214059, upper bound: 1.0365878
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 7, lower bound: -1.0227830, upper bound: 1.0393022
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 7, lower bound: -1.0284712, upper bound: 1.0393030
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 7, lower bound: -1.0265368, upper bound: 1.0365878
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 7, lower bound: -1.0322347, upper bound: 1.0365864
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 7, lower bound: -1.0336009, upper bound: 1.0393030
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.01
Output dim: 7, lower bound: -1.0392998, upper bound: 1.0393031

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.3387828, -2.6574416, -5.3431625, -2.6578724, -2.1865706, 2.1910815
1: -6.3010492, -4.2627878, -6.2938423, -4.2743654, -1.7027617, 1.7081990
2: -4.6378722, -2.6452858, -4.6447034, -2.6486073, -1.5551472, 1.5664706
3: -7.8535137, -5.1136637, -7.8339024, -5.1246109, -2.0329289, 2.0233960
4: -11.7899714, -9.0534487, -11.7948179, -9.0505409, -2.3200274, 2.3226938
5: -6.3583736, -4.2381845, -6.3581147, -4.2421269, -1.7144461, 1.7176609
6: -10.4325123, -7.9457660, -10.4170904, -7.9678245, -1.9476972, 1.9574661
7: -2.8629193, -0.7858596, -2.8658650, -0.7767360, -1.7659416, 1.7574751
8: 1.9796786, 3.5892158, 1.9795847, 3.5855446, -1.3075492, 1.3109288
9: -8.0583153, -5.5700006, -8.0587111, -5.5638528, -2.0265374, 2.0202780

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0157121, upper bound: 1.0308854
time: 5.98 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0157121, upper bound: 1.0365865
time: 4.19 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.3790836, -2.6510015, -5.3431573, -2.6578822, -2.2270012, 2.1987677
1: -6.3151951, -4.2285852, -6.2938237, -4.2743702, -1.7136316, 1.7271869
2: -4.6557717, -2.6162767, -4.6446939, -2.6486096, -1.5758328, 1.5818988
3: -7.8990459, -5.1061382, -7.8338966, -5.1246300, -2.0584488, 2.0305090
4: -11.8020935, -8.9689493, -11.7947884, -9.0505543, -2.3341951, 2.3717327
5: -6.3786817, -4.2339058, -6.3581085, -4.2421274, -1.7487311, 1.7219260
6: -10.4587727, -7.9397502, -10.4170837, -7.9678326, -1.9790630, 1.9633901
7: -2.8743601, -0.7241757, -2.8658392, -0.7767456, -1.7760248, 1.7911919
8: 1.9585061, 3.5917211, 1.9795928, 3.5855422, -1.3310163, 1.3143682
9: -8.0710878, -5.4991140, -8.0586882, -5.5638642, -2.0399046, 2.0751295

Time for backsubstitution: 22.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0214019, upper bound: 1.0336675
time: 4.39 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0214019, upper bound: 1.0365835
time: 4.38 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.3426771, -2.6533222, -5.3535914, -2.6487942, -2.2032695, 2.2101898
1: -6.3154831, -4.2597690, -6.3241682, -4.2543530, -1.7375331, 1.7219439
2: -4.6391554, -2.6384006, -4.6528420, -2.6336021, -1.5644512, 1.5819089
3: -7.8568945, -5.0982785, -7.8584566, -5.0932989, -2.0393934, 2.0631132
4: -11.7999535, -9.0489922, -11.8158817, -9.0326500, -2.3478165, 2.3340940
5: -6.3597641, -4.2336659, -6.3642960, -4.2322936, -1.7255039, 1.7299247
6: -10.4539289, -7.9424729, -10.4607182, -7.9384923, -1.9995337, 1.9614277
7: -2.8682394, -0.7765625, -2.8859212, -0.7579522, -1.7736797, 1.7874234
8: 1.9784789, 3.6004696, 1.9642310, 3.6081948, -1.3059757, 1.3381491
9: -8.0631924, -5.5671587, -8.0728340, -5.5576706, -2.0404959, 2.0410371

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0227767, upper bound: 1.0336011
time: 4.67 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0227767, upper bound: 1.0393023
time: 4.69 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.3829708, -2.6468740, -5.3535862, -2.6488020, -2.2437768, 2.2178688
1: -6.3296032, -4.2255659, -6.3241491, -4.2543583, -1.7483940, 1.7397218
2: -4.6570687, -2.6093845, -4.6528344, -2.6336040, -1.5851178, 1.5959076
3: -7.9024239, -5.0907564, -7.8584509, -5.0933175, -2.0645757, 2.0702238
4: -11.8120708, -8.9644938, -11.8158541, -9.0326605, -2.3619804, 2.3831940
5: -6.3800688, -4.2293873, -6.3642902, -4.2322950, -1.7597923, 1.7341905
6: -10.4801998, -7.9364491, -10.4607105, -7.9385014, -2.0207419, 1.9673426
7: -2.8796890, -0.7148786, -2.8858972, -0.7579610, -1.7837420, 1.8180579
8: 1.9572959, 3.6029749, 1.9642391, 3.6081939, -1.3294561, 1.3415883
9: -8.0759592, -5.4962778, -8.0728121, -5.5576820, -2.0538568, 2.0952251

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 6220

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0284672, upper bound: 1.0363822
time: 4.34 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0284672, upper bound: 1.0392989
time: 3.99 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.3483715, -2.6524577, -5.3440361, -2.6553988, -2.2000170, 2.1882591
1: -6.3058543, -4.2550778, -6.2944183, -4.2707782, -1.7110558, 1.7099953
2: -4.6496520, -2.6369407, -4.6456337, -2.6444864, -1.5710878, 1.5706799
3: -7.8543167, -5.1133270, -7.8348961, -5.1245527, -2.0335836, 2.0285778
4: -11.8044052, -9.0403681, -11.8022003, -9.0500679, -2.3122320, 2.3436995
5: -6.3623166, -4.2370043, -6.3594379, -4.2419491, -1.7172289, 1.7205865
6: -10.4376450, -7.9424601, -10.4176731, -7.9661322, -1.9557223, 1.9545426
7: -2.8853135, -0.7695198, -2.8767552, -0.7765782, -1.7572699, 1.7861714
8: 1.9678397, 3.6027694, 1.9791636, 3.5923219, -1.3270180, 1.3063524
9: -8.0640507, -5.5636158, -8.0617857, -5.5633960, -2.0186806, 2.0316720

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0265358, upper bound: 1.0308869
time: 4.25 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0265358, upper bound: 1.0365878
time: 4.35 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.3887081, -2.6460197, -5.3440289, -2.6554060, -2.2346063, 2.1959496
1: -6.3199520, -4.2209430, -6.2944002, -4.2707834, -1.7218904, 1.7288504
2: -4.6675820, -2.6079531, -4.6456261, -2.6444874, -1.5917907, 1.5833080
3: -7.8997612, -5.1058125, -7.8348904, -5.1245718, -2.0626936, 2.0356617
4: -11.8165274, -8.9558649, -11.8021746, -9.0500774, -2.3264046, 2.3844190
5: -6.3824859, -4.2327247, -6.3594313, -4.2419510, -1.7517300, 1.7248516
6: -10.4639053, -7.9364300, -10.4176655, -7.9661384, -1.9836540, 1.9604762
7: -2.8967903, -0.7078190, -2.8767309, -0.7765863, -1.7673807, 1.8082327
8: 1.9466653, 3.6052732, 1.9791727, 3.5923195, -1.3451533, 1.3097956
9: -8.0768318, -5.4927454, -8.0617638, -5.5634079, -2.0320406, 2.0806391

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0322336, upper bound: 1.0257575
time: 4.30 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0322351, upper bound: 1.0257575
time: 4.50 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.3522673, -2.6483359, -5.3544717, -2.6463172, -2.2167163, 2.2073689
1: -6.3202872, -4.2520580, -6.3247404, -4.2507687, -1.7449682, 1.7237499
2: -4.6509395, -2.6300554, -4.6537790, -2.6294832, -1.5803909, 1.5861239
3: -7.8576975, -5.0979457, -7.8594494, -5.0932417, -2.0400405, 2.0682945
4: -11.8143883, -9.0359135, -11.8232679, -9.0321760, -2.3400249, 2.3551035
5: -6.3637071, -4.2324858, -6.3656187, -4.2321186, -1.7282853, 1.7328498
6: -10.4590588, -7.9391670, -10.4613008, -7.9367967, -2.0075607, 1.9585106
7: -2.8906357, -0.7602222, -2.8968217, -0.7577932, -1.7650113, 1.8161173
8: 1.9666357, 3.6140232, 1.9638057, 3.6149745, -1.3254461, 1.3335741
9: -8.0689325, -5.5607710, -8.0759144, -5.5572162, -2.0326385, 2.0524282

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 5856

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0335999, upper bound: 1.0336018
time: 4.43 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0335999, upper bound: 1.0393030
time: 4.12 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.3925962, -2.6419020, -5.3544650, -2.6463256, -2.2513795, 2.2150531
1: -6.3343611, -4.2179222, -6.3247213, -4.2507739, -1.7557795, 1.7413974
2: -4.6688838, -2.6010621, -4.6537728, -2.6294842, -1.6010771, 1.5973234
3: -7.9031363, -5.0904322, -7.8594437, -5.0932617, -2.0688243, 2.0753756
4: -11.8265057, -8.9514103, -11.8232403, -9.0321894, -2.3541918, 2.3958859
5: -6.3838730, -4.2282066, -6.3656135, -4.2321181, -1.7627902, 1.7371171
6: -10.4853306, -7.9331312, -10.4612932, -7.9368052, -2.0253367, 1.9644344
7: -2.9021213, -0.6985197, -2.8967979, -0.7578018, -1.7751012, 1.8350983
8: 1.9454536, 3.6165271, 1.9638157, 3.6149716, -1.3435462, 1.3370178
9: -8.0817108, -5.4899101, -8.0758924, -5.5572281, -2.0459938, 2.1007335

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5871
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 6220
type: B, layer: 1, pos: 5856
type: B, layer: 1, pos: 4626
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 4573
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 4572
type: B, layer: 1, pos: 6140
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 887
type: B, layer: 1, pos: 5858
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 160

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5871

### Candidate
type: B, layer: 1, pos: 957

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0392988, upper bound: 1.0284740
time: 4.38 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0393003, upper bound: 1.0284740
time: 4.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.18 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0157121, upper bound: 1.0308854
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0157121, upper bound: 1.0365865
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0214019, upper bound: 1.0336675
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0214019, upper bound: 1.0365835
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0227767, upper bound: 1.0336011
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0227767, upper bound: 1.0393023
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0284672, upper bound: 1.0363822
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0284672, upper bound: 1.0392989
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0265358, upper bound: 1.0308869
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0265358, upper bound: 1.0365878
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0322336, upper bound: 1.0257575
NS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0322351, upper bound: 1.0257575
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0335999, upper bound: 1.0336018
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0335999, upper bound: 1.0393030
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0392988, upper bound: 1.0284740
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.18
Output dim: 7, lower bound: -1.0393003, upper bound: 1.0284740

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.3387828, -2.6574416, -5.3813252, -2.6534457, -2.1920562, 2.2263598
1: -6.3010492, -4.2627878, -6.3035073, -4.2414880, -1.7144163, 1.7142346
2: -4.6378722, -2.6452858, -4.6597338, -2.6201763, -1.5706005, 1.5836039
3: -7.8535137, -5.1136637, -7.8776298, -5.1212997, -2.0346360, 2.0418706
4: -11.7899714, -9.0534487, -11.7980728, -8.9697495, -2.3650534, 2.3278465
5: -6.3583736, -4.2381845, -6.3763890, -4.2382259, -1.7171049, 1.7366054
6: -10.4325123, -7.9457660, -10.4410744, -7.9641361, -1.9496741, 1.9793260
7: -2.8629193, -0.7858596, -2.8709059, -0.7174706, -1.8036385, 1.7609396
8: 1.9796786, 3.5892158, 1.9612255, 3.5870824, -1.3090134, 1.3268082
9: -8.0583153, -5.5700006, -8.0645380, -5.4965281, -2.0744987, 2.0253053

Time for backsubstitution: 22.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0128099, upper bound: 1.0365826
time: 5.21 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0157143, upper bound: 1.0365825
time: 4.17 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.3790779, -2.6510088, -5.3618398, -2.6531429, -2.2243304, 2.2247491
1: -6.3151908, -4.2285919, -6.3077426, -4.2666917, -1.7237160, 1.7329602
2: -4.6557636, -2.6162786, -4.6503263, -2.6272101, -1.5923269, 1.5856214
3: -7.8990421, -5.1061392, -7.8405361, -5.1178861, -2.0632033, 2.0401115
4: -11.8020916, -8.9689579, -11.8095474, -9.0456095, -2.3354511, 2.3723888
5: -6.3786778, -4.2339163, -6.3805466, -4.2381563, -1.7509246, 1.7453921
6: -10.4587679, -7.9397497, -10.4234390, -7.9475842, -1.9906306, 1.9692938
7: -2.8743494, -0.7241795, -2.8754661, -0.7551520, -1.7966490, 1.8012941
8: 1.9585161, 3.5917187, 1.9667802, 3.6055679, -1.3331890, 1.3347940
9: -8.0710773, -5.4991169, -8.0643463, -5.5478153, -2.0558748, 2.0797868

Time for backsubstitution: 22.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 4626

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.0199901, upper bound: 1.0326905
time: 4.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0213963, upper bound: 1.0365777
time: 4.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.3426771, -2.6533222, -5.3917227, -2.6443648, -2.2087507, 2.2471406
1: -6.3154831, -4.2597690, -6.3337679, -4.2214818, -1.7436342, 1.7279596
2: -4.6391554, -2.6384006, -4.6679039, -2.6051626, -1.5798891, 1.5990551
3: -7.8568945, -5.0982785, -7.9021673, -5.0900068, -2.0410733, 2.0751057
4: -11.7999535, -9.0489922, -11.8191338, -8.9518738, -2.3904142, 2.3392353
5: -6.3597641, -4.2336659, -6.3825436, -4.2283936, -1.7281680, 1.7488475
6: -10.4539289, -7.9424729, -10.4847183, -7.9347930, -2.0015206, 1.9833074
7: -2.8682394, -0.7765625, -2.8912408, -0.6986911, -1.8114617, 1.7909257
8: 1.9784789, 3.6004696, 1.9458747, 3.6097355, -1.3074408, 1.3503828
9: -8.0631924, -5.5671587, -8.0786648, -5.4903655, -2.0884500, 2.0460577

Time for backsubstitution: 21.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5871
type: A, layer: 1, pos: 6220
type: A, layer: 1, pos: 4626
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 4573
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 6140
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 887
type: A, layer: 1, pos: 5858
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 160

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5871

### Candidate
type: A, layer: 1, pos: 6220

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0198858, upper bound: 1.0392979
time: 4.29 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0227789, upper bound: 1.0392982
time: 4.30 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.13 + 547.91 = 605.03 seconds
