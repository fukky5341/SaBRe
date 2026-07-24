## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 7.432e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0081723, -0.0075867, -0.0081723, -0.0075867, -0.0002791, 0.0002791)
1: (-0.0052427, -0.0050776, -0.0052427, -0.0050776, -0.0000787, 0.0000787)
2: (-0.0001223, 0.0010960, -0.0001223, 0.0010960, -0.0005805, 0.0005805)
3: (0.0016111, 0.0017723, 0.0016111, 0.0017723, -0.0000768, 0.0000768)
4: (0.0052728, 0.0061832, 0.0052728, 0.0061832, -0.0004338, 0.0004338)
5: (0.9969711, 0.9972242, 0.9969711, 0.9972242, -0.0001205, 0.0001205)
6: (0.0051344, 0.0053640, 0.0051344, 0.0053640, -0.0001094, 0.0001094)
7: (-0.0042209, -0.0033640, -0.0042209, -0.0033640, -0.0004083, 0.0004083)
8: (-0.0065746, -0.0059078, -0.0065746, -0.0059078, -0.0003178, 0.0003178)
9: (-0.0035000, -0.0034425, -0.0035000, -0.0034425, -0.0000274, 0.0000274)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.40 + 1.29 = 2.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0000804, upper bound: 0.0000805

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 177

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000775, upper bound: 0.0000772
time: 0.44 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000774, upper bound: 0.0000777
time: 0.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.08 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 5, lower bound: -0.0000775, upper bound: 0.0000772
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.08
Output dim: 5, lower bound: -0.0000774, upper bound: 0.0000777

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0081610, -0.0075876, -0.0081693, -0.0075869, -0.0002670, 0.0002747
1: -0.0052396, -0.0050779, -0.0052419, -0.0050777, -0.0000753, 0.0000774
2: -0.0000988, 0.0010942, -0.0001159, 0.0010955, -0.0005555, 0.0005714
3: 0.0016142, 0.0017721, 0.0016120, 0.0017723, -0.0000735, 0.0000756
4: 0.0052741, 0.0061657, 0.0052731, 0.0061785, -0.0004270, 0.0004152
5: 0.9969715, 0.9972193, 0.9969713, 0.9972228, -0.0001186, 0.0001153
6: 0.0051347, 0.0053596, 0.0051345, 0.0053628, -0.0001077, 0.0001047
7: -0.0042196, -0.0033805, -0.0042205, -0.0033685, -0.0004019, 0.0003907
8: -0.0065618, -0.0059088, -0.0065712, -0.0059080, -0.0003041, 0.0003128
9: -0.0035000, -0.0034436, -0.0035000, -0.0034428, -0.0000270, 0.0000262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000739, upper bound: 0.0000745
time: 0.44 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000748, upper bound: 0.0000744
time: 0.44 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0081578, -0.0075864, -0.0081678, -0.0075870, -0.0002671, 0.0002771
1: -0.0052386, -0.0050776, -0.0052415, -0.0050777, -0.0000753, 0.0000781
2: -0.0000920, 0.0010966, -0.0001128, 0.0010953, -0.0005556, 0.0005765
3: 0.0016151, 0.0017724, 0.0016124, 0.0017722, -0.0000735, 0.0000763
4: 0.0052724, 0.0061607, 0.0052733, 0.0061762, -0.0004308, 0.0004153
5: 0.9969711, 0.9972179, 0.9969713, 0.9972222, -0.0001197, 0.0001154
6: 0.0051343, 0.0053583, 0.0051345, 0.0053622, -0.0001086, 0.0001047
7: -0.0042213, -0.0033853, -0.0042204, -0.0033707, -0.0004055, 0.0003908
8: -0.0065581, -0.0059075, -0.0065695, -0.0059081, -0.0003042, 0.0003156
9: -0.0035001, -0.0034439, -0.0035000, -0.0034430, -0.0000272, 0.0000262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000739, upper bound: 0.0000749
time: 0.46 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000748, upper bound: 0.0000749
time: 0.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 5, lower bound: -0.0000739, upper bound: 0.0000745
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 5, lower bound: -0.0000748, upper bound: 0.0000744
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 5, lower bound: -0.0000739, upper bound: 0.0000749
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.43
Output dim: 5, lower bound: -0.0000748, upper bound: 0.0000749

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0081534, -0.0075876, -0.0081468, -0.0075782, -0.0002479, 0.0002421
1: -0.0052374, -0.0050779, -0.0052356, -0.0050752, -0.0000699, 0.0000683
2: -0.0000829, 0.0010942, -0.0000692, 0.0011137, -0.0005157, 0.0005037
3: 0.0016163, 0.0017721, 0.0016181, 0.0017747, -0.0000683, 0.0000667
4: 0.0052742, 0.0061538, 0.0052595, 0.0061436, -0.0003764, 0.0003854
5: 0.9969715, 0.9972160, 0.9969675, 0.9972131, -0.0001046, 0.0001071
6: 0.0051347, 0.0053566, 0.0051310, 0.0053540, -0.0000949, 0.0000972
7: -0.0042196, -0.0033917, -0.0042333, -0.0034013, -0.0003542, 0.0003627
8: -0.0065531, -0.0059088, -0.0065456, -0.0058981, -0.0002823, 0.0002757
9: -0.0035000, -0.0034444, -0.0035009, -0.0034450, -0.0000238, 0.0000244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000744
time: 0.44 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000745
time: 0.43 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0081606, -0.0075876, -0.0081627, -0.0075869, -0.0002669, 0.0002364
1: -0.0052394, -0.0050779, -0.0052400, -0.0050777, -0.0000753, 0.0000666
2: -0.0000979, 0.0010942, -0.0001022, 0.0010955, -0.0005553, 0.0004917
3: 0.0016143, 0.0017721, 0.0016138, 0.0017723, -0.0000735, 0.0000651
4: 0.0052741, 0.0061650, 0.0052732, 0.0061683, -0.0003675, 0.0004150
5: 0.9969715, 0.9972191, 0.9969713, 0.9972200, -0.0001021, 0.0001153
6: 0.0051347, 0.0053594, 0.0051345, 0.0053602, -0.0000927, 0.0001047
7: -0.0042196, -0.0033812, -0.0042205, -0.0033781, -0.0003458, 0.0003906
8: -0.0065613, -0.0059088, -0.0065637, -0.0059080, -0.0003040, 0.0002692
9: -0.0035000, -0.0034437, -0.0035000, -0.0034435, -0.0000232, 0.0000262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000749, upper bound: 0.0000731
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000749, upper bound: 0.0000744
time: 0.43 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0081503, -0.0075864, -0.0081455, -0.0075783, -0.0002480, 0.0002475
1: -0.0052365, -0.0050776, -0.0052352, -0.0050753, -0.0000699, 0.0000698
2: -0.0000764, 0.0010965, -0.0000664, 0.0011135, -0.0005160, 0.0005149
3: 0.0016172, 0.0017724, 0.0016185, 0.0017747, -0.0000683, 0.0000681
4: 0.0052724, 0.0061490, 0.0052597, 0.0061415, -0.0003848, 0.0003856
5: 0.9969711, 0.9972146, 0.9969676, 0.9972126, -0.0001069, 0.0001071
6: 0.0051343, 0.0053554, 0.0051311, 0.0053535, -0.0000971, 0.0000972
7: -0.0042212, -0.0033963, -0.0042332, -0.0034033, -0.0003622, 0.0003629
8: -0.0065495, -0.0059075, -0.0065440, -0.0058982, -0.0002824, 0.0002819
9: -0.0035001, -0.0034447, -0.0035009, -0.0034451, -0.0000243, 0.0000244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000749
time: 0.45 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000748
time: 0.45 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0081573, -0.0075864, -0.0081609, -0.0075870, -0.0002670, 0.0002504
1: -0.0052385, -0.0050776, -0.0052395, -0.0050777, -0.0000753, 0.0000706
2: -0.0000911, 0.0010966, -0.0000985, 0.0010953, -0.0005554, 0.0005209
3: 0.0016152, 0.0017724, 0.0016143, 0.0017722, -0.0000735, 0.0000689
4: 0.0052724, 0.0061599, 0.0052733, 0.0061654, -0.0003893, 0.0004151
5: 0.9969711, 0.9972177, 0.9969713, 0.9972192, -0.0001081, 0.0001153
6: 0.0051343, 0.0053581, 0.0051345, 0.0053595, -0.0000982, 0.0001047
7: -0.0042213, -0.0033859, -0.0042204, -0.0033808, -0.0003663, 0.0003907
8: -0.0065576, -0.0059075, -0.0065616, -0.0059082, -0.0003040, 0.0002851
9: -0.0035001, -0.0034440, -0.0035000, -0.0034436, -0.0000246, 0.0000262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000748, upper bound: 0.0000740
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000748, upper bound: 0.0000749
time: 0.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.34 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000744
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000745
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 5, lower bound: -0.0000749, upper bound: 0.0000731
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 5, lower bound: -0.0000749, upper bound: 0.0000744
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000749
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000748
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 5, lower bound: -0.0000748, upper bound: 0.0000740
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 5, lower bound: -0.0000748, upper bound: 0.0000749

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0081534, -0.0075876, -0.0081386, -0.0075788, -0.0002472, 0.0002323
1: -0.0052374, -0.0050779, -0.0052332, -0.0050754, -0.0000697, 0.0000655
2: -0.0000829, 0.0010942, -0.0000521, 0.0011124, -0.0005142, 0.0004832
3: 0.0016163, 0.0017721, 0.0016204, 0.0017745, -0.0000680, 0.0000639
4: 0.0052742, 0.0061538, 0.0052605, 0.0061308, -0.0003611, 0.0003843
5: 0.9969715, 0.9972160, 0.9969678, 0.9972096, -0.0001003, 0.0001068
6: 0.0051347, 0.0053566, 0.0051313, 0.0053508, -0.0000911, 0.0000969
7: -0.0042196, -0.0033917, -0.0042324, -0.0034134, -0.0003399, 0.0003616
8: -0.0065531, -0.0059088, -0.0065362, -0.0058988, -0.0002815, 0.0002645
9: -0.0035000, -0.0034444, -0.0035008, -0.0034458, -0.0000228, 0.0000243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000735
time: 0.44 seconds

## Relational analysis of IS_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000745
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0081534, -0.0075876, -0.0081356, -0.0075768, -0.0002592, 0.0002402
1: -0.0052374, -0.0050779, -0.0052324, -0.0050748, -0.0000731, 0.0000677
2: -0.0000829, 0.0010942, -0.0000458, 0.0011165, -0.0005392, 0.0004997
3: 0.0016163, 0.0017721, 0.0016212, 0.0017750, -0.0000714, 0.0000661
4: 0.0052742, 0.0061538, 0.0052574, 0.0061261, -0.0003735, 0.0004030
5: 0.9969715, 0.9972160, 0.9969669, 0.9972082, -0.0001038, 0.0001120
6: 0.0051347, 0.0053566, 0.0051305, 0.0053496, -0.0000942, 0.0001016
7: -0.0042196, -0.0033917, -0.0042353, -0.0034178, -0.0003515, 0.0003792
8: -0.0065531, -0.0059088, -0.0065328, -0.0058965, -0.0002952, 0.0002735
9: -0.0035000, -0.0034444, -0.0035010, -0.0034461, -0.0000236, 0.0000255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000736
time: 0.44 seconds

## Relational analysis of IS_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000744
time: 0.44 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0081386, -0.0075788, -0.0081627, -0.0075869, -0.0002332, 0.0002669
1: -0.0052332, -0.0050754, -0.0052400, -0.0050777, -0.0000657, 0.0000752
2: -0.0000521, 0.0011124, -0.0001022, 0.0010955, -0.0004850, 0.0005552
3: 0.0016204, 0.0017745, 0.0016138, 0.0017723, -0.0000642, 0.0000735
4: 0.0052605, 0.0061308, 0.0052732, 0.0061683, -0.0004149, 0.0003625
5: 0.9969678, 0.9972096, 0.9969713, 0.9972200, -0.0001153, 0.0001007
6: 0.0051313, 0.0053508, 0.0051345, 0.0053602, -0.0001046, 0.0000914
7: -0.0042324, -0.0034134, -0.0042205, -0.0033781, -0.0003905, 0.0003411
8: -0.0065362, -0.0058988, -0.0065637, -0.0059080, -0.0002655, 0.0003039
9: -0.0035008, -0.0034458, -0.0035000, -0.0034435, -0.0000262, 0.0000229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000732
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000731
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0081545, -0.0075876, -0.0081627, -0.0075869, -0.0002273, 0.0002364
1: -0.0052377, -0.0050779, -0.0052400, -0.0050777, -0.0000641, 0.0000666
2: -0.0000851, 0.0010942, -0.0001022, 0.0010955, -0.0004728, 0.0004917
3: 0.0016160, 0.0017721, 0.0016138, 0.0017723, -0.0000626, 0.0000651
4: 0.0052741, 0.0061555, 0.0052732, 0.0061683, -0.0003675, 0.0003533
5: 0.9969715, 0.9972164, 0.9969713, 0.9972200, -0.0001021, 0.0000982
6: 0.0051347, 0.0053570, 0.0051345, 0.0053602, -0.0000927, 0.0000891
7: -0.0042196, -0.0033901, -0.0042205, -0.0033781, -0.0003458, 0.0003325
8: -0.0065543, -0.0059088, -0.0065637, -0.0059080, -0.0002588, 0.0002692
9: -0.0035000, -0.0034443, -0.0035000, -0.0034435, -0.0000232, 0.0000223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000745
time: 0.45 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000744
time: 0.43 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0081503, -0.0075864, -0.0081386, -0.0075788, -0.0002504, 0.0002357
1: -0.0052365, -0.0050776, -0.0052332, -0.0050754, -0.0000706, 0.0000664
2: -0.0000764, 0.0010965, -0.0000521, 0.0011124, -0.0005208, 0.0004903
3: 0.0016172, 0.0017724, 0.0016204, 0.0017745, -0.0000689, 0.0000649
4: 0.0052724, 0.0061490, 0.0052605, 0.0061308, -0.0003664, 0.0003892
5: 0.9969711, 0.9972146, 0.9969678, 0.9972096, -0.0001018, 0.0001081
6: 0.0051343, 0.0053554, 0.0051313, 0.0053508, -0.0000924, 0.0000982
7: -0.0042212, -0.0033963, -0.0042324, -0.0034134, -0.0003448, 0.0003663
8: -0.0065495, -0.0059075, -0.0065362, -0.0058988, -0.0002851, 0.0002684
9: -0.0035001, -0.0034447, -0.0035008, -0.0034458, -0.0000232, 0.0000246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000741
time: 0.46 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000749
time: 0.44 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0081503, -0.0075864, -0.0081356, -0.0075768, -0.0002472, 0.0002322
1: -0.0052365, -0.0050776, -0.0052324, -0.0050748, -0.0000697, 0.0000655
2: -0.0000764, 0.0010965, -0.0000458, 0.0011165, -0.0005143, 0.0004831
3: 0.0016172, 0.0017724, 0.0016212, 0.0017750, -0.0000681, 0.0000639
4: 0.0052724, 0.0061490, 0.0052574, 0.0061261, -0.0003610, 0.0003843
5: 0.9969711, 0.9972146, 0.9969669, 0.9972082, -0.0001003, 0.0001068
6: 0.0051343, 0.0053554, 0.0051305, 0.0053496, -0.0000910, 0.0000969
7: -0.0042212, -0.0033963, -0.0042353, -0.0034178, -0.0003398, 0.0003617
8: -0.0065495, -0.0059075, -0.0065328, -0.0058965, -0.0002815, 0.0002644
9: -0.0035001, -0.0034447, -0.0035010, -0.0034461, -0.0000228, 0.0000243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000735
time: 0.46 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000749
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0081356, -0.0075768, -0.0081609, -0.0075870, -0.0002332, 0.0002775
1: -0.0052324, -0.0050748, -0.0052395, -0.0050777, -0.0000658, 0.0000782
2: -0.0000458, 0.0011165, -0.0000985, 0.0010953, -0.0004852, 0.0005773
3: 0.0016212, 0.0017750, 0.0016143, 0.0017722, -0.0000642, 0.0000764
4: 0.0052574, 0.0061261, 0.0052733, 0.0061654, -0.0004315, 0.0003626
5: 0.9969669, 0.9972082, 0.9969713, 0.9972192, -0.0001199, 0.0001007
6: 0.0051305, 0.0053496, 0.0051345, 0.0053595, -0.0001088, 0.0000914
7: -0.0042353, -0.0034178, -0.0042204, -0.0033808, -0.0004061, 0.0003412
8: -0.0065328, -0.0058965, -0.0065616, -0.0059082, -0.0002656, 0.0003160
9: -0.0035010, -0.0034461, -0.0035000, -0.0034436, -0.0000273, 0.0000229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000739
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000740
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0081507, -0.0075864, -0.0081609, -0.0075870, -0.0002282, 0.0002504
1: -0.0052366, -0.0050776, -0.0052395, -0.0050777, -0.0000643, 0.0000706
2: -0.0000772, 0.0010966, -0.0000985, 0.0010953, -0.0004746, 0.0005209
3: 0.0016171, 0.0017724, 0.0016143, 0.0017722, -0.0000628, 0.0000689
4: 0.0052724, 0.0061496, 0.0052733, 0.0061654, -0.0003893, 0.0003547
5: 0.9969711, 0.9972148, 0.9969713, 0.9972192, -0.0001081, 0.0000985
6: 0.0051343, 0.0053555, 0.0051345, 0.0053595, -0.0000982, 0.0000895
7: -0.0042212, -0.0033957, -0.0042204, -0.0033808, -0.0003663, 0.0003338
8: -0.0065500, -0.0059075, -0.0065616, -0.0059082, -0.0002598, 0.0002851
9: -0.0035001, -0.0034446, -0.0035000, -0.0034436, -0.0000246, 0.0000224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 177

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000749
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000749
time: 0.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.34 seconds
IS_A1_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000735
IS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000745
IS_A1_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000736
IS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000744
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000732
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000731
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000745
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000731, upper bound: 0.0000744
IS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000741
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000749
IS_A2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000735
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000749
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000739
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000740
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000749
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 5, lower bound: -0.0000732, upper bound: 0.0000749

## BFS IS instance: IS_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0081545, -0.0075876, -0.0081386, -0.0075788, -0.0002584, 0.0002323
1: -0.0052377, -0.0050779, -0.0052332, -0.0050754, -0.0000729, 0.0000655
2: -0.0000851, 0.0010942, -0.0000521, 0.0011124, -0.0005376, 0.0004832
3: 0.0016160, 0.0017721, 0.0016204, 0.0017745, -0.0000711, 0.0000639
4: 0.0052741, 0.0061555, 0.0052605, 0.0061308, -0.0003611, 0.0004018
5: 0.9969715, 0.9972164, 0.9969678, 0.9972096, -0.0001003, 0.0001116
6: 0.0051347, 0.0053570, 0.0051313, 0.0053508, -0.0000911, 0.0001013
7: -0.0042196, -0.0033901, -0.0042324, -0.0034134, -0.0003399, 0.0003781
8: -0.0065543, -0.0059088, -0.0065362, -0.0058988, -0.0002943, 0.0002645
9: -0.0035000, -0.0034443, -0.0035008, -0.0034458, -0.0000228, 0.0000254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000711, upper bound: 0.0000726
time: 0.46 seconds

## Relational analysis of IS_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000728
time: 0.50 seconds

## BFS IS instance: IS_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0081545, -0.0075876, -0.0081356, -0.0075768, -0.0002705, 0.0002402
1: -0.0052377, -0.0050779, -0.0052324, -0.0050748, -0.0000763, 0.0000677
2: -0.0000851, 0.0010942, -0.0000458, 0.0011165, -0.0005626, 0.0004997
3: 0.0016160, 0.0017721, 0.0016212, 0.0017750, -0.0000745, 0.0000661
4: 0.0052741, 0.0061555, 0.0052574, 0.0061261, -0.0003735, 0.0004205
5: 0.9969715, 0.9972164, 0.9969669, 0.9972082, -0.0001038, 0.0001168
6: 0.0051347, 0.0053570, 0.0051305, 0.0053496, -0.0000942, 0.0001060
7: -0.0042196, -0.0033901, -0.0042353, -0.0034178, -0.0003515, 0.0003957
8: -0.0065543, -0.0059088, -0.0065328, -0.0058965, -0.0003080, 0.0002735
9: -0.0035000, -0.0034443, -0.0035010, -0.0034461, -0.0000236, 0.0000266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000714, upper bound: 0.0000726
time: 0.46 seconds

## Relational analysis of IS_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000723, upper bound: 0.0000727
time: 0.48 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0081545, -0.0075876, -0.0081545, -0.0075876, -0.0002265, 0.0002265
1: -0.0052377, -0.0050779, -0.0052377, -0.0050779, -0.0000639, 0.0000639
2: -0.0000851, 0.0010942, -0.0000851, 0.0010942, -0.0004713, 0.0004713
3: 0.0016160, 0.0017721, 0.0016160, 0.0017721, -0.0000624, 0.0000624
4: 0.0052741, 0.0061555, 0.0052741, 0.0061555, -0.0003522, 0.0003522
5: 0.9969715, 0.9972164, 0.9969715, 0.9972164, -0.0000979, 0.0000979
6: 0.0051347, 0.0053570, 0.0051347, 0.0053570, -0.0000888, 0.0000888
7: -0.0042196, -0.0033901, -0.0042196, -0.0033901, -0.0003315, 0.0003315
8: -0.0065543, -0.0059088, -0.0065543, -0.0059088, -0.0002580, 0.0002580
9: -0.0035000, -0.0034443, -0.0035000, -0.0034443, -0.0000223, 0.0000223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000714, upper bound: 0.0000724
time: 0.46 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000727
time: 0.49 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0081545, -0.0075876, -0.0081507, -0.0075864, -0.0002385, 0.0002345
1: -0.0052377, -0.0050779, -0.0052366, -0.0050776, -0.0000672, 0.0000661
2: -0.0000851, 0.0010942, -0.0000772, 0.0010966, -0.0004962, 0.0004877
3: 0.0016160, 0.0017721, 0.0016171, 0.0017724, -0.0000657, 0.0000645
4: 0.0052741, 0.0061555, 0.0052724, 0.0061496, -0.0003645, 0.0003708
5: 0.9969715, 0.9972164, 0.9969711, 0.9972148, -0.0001013, 0.0001030
6: 0.0051347, 0.0053570, 0.0051343, 0.0053555, -0.0000919, 0.0000935
7: -0.0042196, -0.0033901, -0.0042212, -0.0033957, -0.0003430, 0.0003490
8: -0.0065543, -0.0059088, -0.0065500, -0.0059075, -0.0002716, 0.0002670
9: -0.0035000, -0.0034443, -0.0035001, -0.0034446, -0.0000230, 0.0000234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000714, upper bound: 0.0000724
time: 0.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000727
time: 0.46 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0081507, -0.0075864, -0.0081386, -0.0075788, -0.0002570, 0.0002357
1: -0.0052366, -0.0050776, -0.0052332, -0.0050754, -0.0000724, 0.0000664
2: -0.0000772, 0.0010966, -0.0000521, 0.0011124, -0.0005345, 0.0004903
3: 0.0016171, 0.0017724, 0.0016204, 0.0017745, -0.0000707, 0.0000649
4: 0.0052724, 0.0061496, 0.0052605, 0.0061308, -0.0003664, 0.0003995
5: 0.9969711, 0.9972148, 0.9969678, 0.9972096, -0.0001018, 0.0001110
6: 0.0051343, 0.0053555, 0.0051313, 0.0053508, -0.0000924, 0.0001007
7: -0.0042212, -0.0033957, -0.0042324, -0.0034134, -0.0003448, 0.0003759
8: -0.0065500, -0.0059075, -0.0065362, -0.0058988, -0.0002926, 0.0002684
9: -0.0035001, -0.0034446, -0.0035008, -0.0034458, -0.0000232, 0.0000252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000711, upper bound: 0.0000732
time: 0.47 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000733
time: 0.47 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0081507, -0.0075864, -0.0081356, -0.0075768, -0.0002585, 0.0002322
1: -0.0052366, -0.0050776, -0.0052324, -0.0050748, -0.0000729, 0.0000655
2: -0.0000772, 0.0010966, -0.0000458, 0.0011165, -0.0005377, 0.0004831
3: 0.0016171, 0.0017724, 0.0016212, 0.0017750, -0.0000712, 0.0000639
4: 0.0052724, 0.0061496, 0.0052574, 0.0061261, -0.0003610, 0.0004018
5: 0.9969711, 0.9972148, 0.9969669, 0.9972082, -0.0001003, 0.0001116
6: 0.0051343, 0.0053555, 0.0051305, 0.0053496, -0.0000910, 0.0001013
7: -0.0042212, -0.0033957, -0.0042353, -0.0034178, -0.0003398, 0.0003782
8: -0.0065500, -0.0059075, -0.0065328, -0.0058965, -0.0002943, 0.0002644
9: -0.0035001, -0.0034446, -0.0035010, -0.0034461, -0.0000228, 0.0000254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000710, upper bound: 0.0000731
time: 0.47 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000733
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0081507, -0.0075864, -0.0081545, -0.0075876, -0.0002345, 0.0002385
1: -0.0052366, -0.0050776, -0.0052377, -0.0050779, -0.0000661, 0.0000672
2: -0.0000772, 0.0010966, -0.0000851, 0.0010942, -0.0004877, 0.0004962
3: 0.0016171, 0.0017724, 0.0016160, 0.0017721, -0.0000645, 0.0000657
4: 0.0052724, 0.0061496, 0.0052741, 0.0061555, -0.0003708, 0.0003645
5: 0.9969711, 0.9972148, 0.9969715, 0.9972164, -0.0001030, 0.0001013
6: 0.0051343, 0.0053555, 0.0051347, 0.0053570, -0.0000935, 0.0000919
7: -0.0042212, -0.0033957, -0.0042196, -0.0033901, -0.0003490, 0.0003430
8: -0.0065500, -0.0059075, -0.0065543, -0.0059088, -0.0002670, 0.0002716
9: -0.0035001, -0.0034446, -0.0035000, -0.0034443, -0.0000234, 0.0000230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000710, upper bound: 0.0000732
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000714, upper bound: 0.0000733
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0081507, -0.0075864, -0.0081507, -0.0075864, -0.0002273, 0.0002273
1: -0.0052366, -0.0050776, -0.0052366, -0.0050776, -0.0000641, 0.0000641
2: -0.0000772, 0.0010966, -0.0000772, 0.0010966, -0.0004728, 0.0004728
3: 0.0016171, 0.0017724, 0.0016171, 0.0017724, -0.0000626, 0.0000626
4: 0.0052724, 0.0061496, 0.0052724, 0.0061496, -0.0003534, 0.0003534
5: 0.9969711, 0.9972148, 0.9969711, 0.9972148, -0.0000982, 0.0000982
6: 0.0051343, 0.0053555, 0.0051343, 0.0053555, -0.0000891, 0.0000891
7: -0.0042212, -0.0033957, -0.0042212, -0.0033957, -0.0003326, 0.0003326
8: -0.0065500, -0.0059075, -0.0065500, -0.0059075, -0.0002588, 0.0002588
9: -0.0035001, -0.0034446, -0.0035001, -0.0034446, -0.0000223, 0.0000223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 205

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000726
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000714, upper bound: 0.0000733
time: 0.48 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.41 seconds
IS_A1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000711, upper bound: 0.0000726
IS_A1_B1_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000728
IS_A1_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000714, upper bound: 0.0000726
IS_A1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000723, upper bound: 0.0000727
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000714, upper bound: 0.0000724
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000727
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000714, upper bound: 0.0000724
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000727
IS_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000711, upper bound: 0.0000732
IS_A2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000733
IS_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000710, upper bound: 0.0000731
IS_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000733
IS_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000710, upper bound: 0.0000732
IS_A2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000714, upper bound: 0.0000733
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000715, upper bound: 0.0000726
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.41
Output dim: 5, lower bound: -0.0000714, upper bound: 0.0000733

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.69 + 53.08 = 55.77 seconds
