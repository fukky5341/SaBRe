## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.030627584


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9354143, -5.3531480, -8.9354143, -5.3531480, -2.9971504, 2.9971502)
1: (-7.3978767, -4.1556597, -7.3978767, -4.1556597, -2.4096041, 2.4096045)
2: (-7.4789824, -4.5742288, -7.4789824, -4.5742288, -2.3071775, 2.3071778)
3: (-11.2633400, -7.7441721, -11.2633400, -7.7441721, -2.6513076, 2.6513076)
4: (6.5621047, 8.8026104, 6.5621047, 8.8026104, -1.6619794, 1.6619792)
5: (-8.9045200, -5.9158406, -8.9045200, -5.9158406, -2.2816925, 2.2816925)
6: (-12.0150757, -8.2602491, -12.0150757, -8.2602491, -3.1651812, 3.1651816)
7: (-3.2182775, -0.5745691, -3.2182775, -0.5745691, -2.3866134, 2.3866131)
8: (-6.9675965, -3.5078909, -6.9675965, -3.5078909, -2.3960896, 2.3960896)
9: (-5.5373087, -3.0319777, -5.5373087, -3.0319777, -2.0071397, 2.0071399)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.37 + 36.23 = 59.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.0516605, upper bound: 1.0516610

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415916, upper bound: 1.0515312
time: 4.63 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516523, upper bound: 1.0516539
time: 7.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.32 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.32
Output dim: 4, lower bound: -1.0415916, upper bound: 1.0515312
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.32
Output dim: 4, lower bound: -1.0516523, upper bound: 1.0516539

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.9197674, -5.3591332, -8.9325085, -5.3542271, -2.9736238, 2.9822109
1: -7.3834295, -4.1582737, -7.3952045, -4.1561289, -2.3947172, 2.4047506
2: -7.4752841, -4.5928931, -7.4782968, -4.5776963, -2.2961621, 2.2872756
3: -11.2590904, -7.7627225, -11.2625790, -7.7476125, -2.6412959, 2.6309493
4: 6.5971785, 8.8024063, 6.5686030, 8.8025713, -1.6230602, 1.6513443
5: -8.9024868, -5.9274840, -8.9041462, -5.9180188, -2.2699537, 2.2675881
6: -11.9991131, -8.2676945, -12.0121117, -8.2615871, -3.1460953, 3.1550260
7: -3.1996951, -0.5760213, -3.2148366, -0.5748354, -2.3665943, 2.3809810
8: -6.9664283, -3.5248682, -6.9673815, -3.5110507, -2.3891392, 2.3760669
9: -5.5144119, -3.0330348, -5.5330515, -3.0321708, -1.9809425, 1.9987962

Time for backsubstitution: 21.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415900, upper bound: 1.0415896
time: 5.51 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415900, upper bound: 1.0515293
time: 6.32 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.9540691, -5.3130355, -8.9353790, -5.3531542, -3.0080986, 3.0405321
1: -7.4122467, -4.1019802, -7.3978658, -4.1556625, -2.4212656, 2.4459326
2: -7.5391769, -4.5643067, -7.4789762, -4.5742507, -2.3414292, 2.3133636
3: -11.3160086, -7.7315407, -11.2633362, -7.7442074, -2.6848221, 2.6607280
4: 6.5178432, 8.8160534, 6.5621443, 8.8026114, -1.6993644, 1.6724558
5: -8.9193316, -5.9010220, -8.9045153, -5.9158602, -2.2858095, 2.3005280
6: -12.0178509, -8.1924677, -12.0150404, -8.2602596, -3.1684685, 3.2192683
7: -3.2606452, -0.5148413, -3.2182617, -0.5745696, -2.4425821, 2.4113929
8: -7.0120811, -3.4878445, -6.9675951, -3.5079393, -2.4350729, 2.4111409
9: -5.5725112, -3.0261889, -5.5372543, -3.0319786, -2.0387962, 2.0100088

Time for backsubstitution: 22.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516348, upper bound: 1.0497404
time: 11.87 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516348, upper bound: 1.0516363
time: 8.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 43.09 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 43.09
Output dim: 4, lower bound: -1.0415900, upper bound: 1.0415896
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 43.09
Output dim: 4, lower bound: -1.0415900, upper bound: 1.0515293
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 43.09
Output dim: 4, lower bound: -1.0516348, upper bound: 1.0497404
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 43.09
Output dim: 4, lower bound: -1.0516348, upper bound: 1.0516363

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -8.9197674, -5.3591332, -8.9197674, -5.3591332, -2.9646912, 2.9646914
1: -7.3834295, -4.1582737, -7.3834295, -4.1582737, -2.3929381, 2.3929384
2: -7.4752841, -4.5928931, -7.4752841, -4.5928931, -2.2811103, 2.2811105
3: -11.2590904, -7.7627225, -11.2590904, -7.7627225, -2.6256661, 2.6256664
4: 6.5971785, 8.8024063, 6.5971785, 8.8024063, -1.6201737, 1.6201737
5: -8.9024868, -5.9274840, -8.9024868, -5.9274840, -2.2598753, 2.2598753
6: -11.9991131, -8.2676945, -11.9991131, -8.2676945, -3.1404953, 3.1404953
7: -3.1996951, -0.5760213, -3.1996951, -0.5760213, -2.3649559, 2.3649559
8: -6.9664283, -3.5248682, -6.9664283, -3.5248682, -2.3733406, 2.3733411
9: -5.5144119, -3.0330348, -5.5144119, -3.0330348, -1.9780006, 1.9780006

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415742, upper bound: 1.0396740
time: 7.35 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415742, upper bound: 1.0415723
time: 7.60 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.9197674, -5.3591332, -8.9531040, -5.3171086, -3.0054231, 2.9962945
1: -7.3834295, -4.1582737, -7.4116163, -4.1047363, -2.4291937, 2.4186809
2: -7.4752841, -4.5928931, -7.5386448, -4.5646191, -2.3077278, 2.3220353
3: -11.2590904, -7.7627225, -11.3139153, -7.7319965, -2.6546736, 2.6636143
4: 6.5971785, 8.8024063, 6.5190825, 8.8159256, -1.6340458, 1.6952922
5: -8.9024868, -5.9274840, -8.9186468, -5.9019966, -2.2858849, 2.2730551
6: -11.9991131, -8.2676945, -12.0176983, -8.1965790, -3.1964703, 3.1617355
7: -3.1996951, -0.5760213, -3.2595568, -0.5160203, -2.3903437, 2.4398718
8: -6.9664283, -3.5248682, -7.0120068, -3.4886198, -2.4089003, 2.4155576
9: -5.5144119, -3.0330348, -5.5712423, -3.0262704, -1.9844499, 2.0346630

Time for backsubstitution: 23.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396750, upper bound: 1.0515138
time: 5.61 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415739, upper bound: 1.0515138
time: 5.48 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -8.9520903, -5.3151503, -8.9353790, -5.3531542, -3.0047750, 3.0371962
1: -7.4083495, -4.1028442, -7.3978658, -4.1556625, -2.4168653, 2.4436922
2: -7.5371480, -4.5664854, -7.4789762, -4.5742507, -2.3395443, 2.3107958
3: -11.3139458, -7.7330923, -11.2633362, -7.7442074, -2.6821585, 2.6588411
4: 6.5195827, 8.8143463, 6.5621443, 8.8026114, -1.6976502, 1.6706095
5: -8.9155273, -5.9021783, -8.9045153, -5.9158602, -2.2816472, 2.2990565
6: -12.0140438, -8.1945267, -12.0150404, -8.2602596, -3.1646423, 3.2177081
7: -3.2581868, -0.5191920, -3.2182617, -0.5745696, -2.4398980, 2.4062209
8: -7.0045671, -3.4896250, -6.9675951, -3.5079393, -2.4274979, 2.4088774
9: -5.5702739, -3.0295439, -5.5372543, -3.0319786, -2.0365665, 2.0064597

Time for backsubstitution: 23.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0497380, upper bound: 1.0497401
time: 5.94 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0497380, upper bound: 1.0497423
time: 9.70 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -8.9702892, -5.3074450, -8.9353752, -5.3531618, -3.0245399, 3.0471923
1: -7.4318342, -4.0778494, -7.3978519, -4.1556644, -2.4463854, 2.4506569
2: -7.5578356, -4.5577469, -7.4789705, -4.5742559, -2.3602598, 2.3205929
3: -11.3224220, -7.6950336, -11.2633247, -7.7442122, -2.6930895, 2.6972134
4: 6.4450040, 8.8188009, 6.5621481, 8.8026056, -1.7677157, 1.6761451
5: -8.9219227, -5.8406072, -8.9044991, -5.9158626, -2.2886200, 2.3441300
6: -12.0291386, -8.1469898, -12.0150270, -8.2602634, -3.1814756, 3.2495999
7: -3.3067019, -0.5109186, -3.2182527, -0.5745854, -2.4832153, 2.4140391
8: -7.0198908, -3.4194520, -6.9675698, -3.5079436, -2.4454889, 2.4651871
9: -5.6438265, -3.0212150, -5.5372486, -3.0319920, -2.0926790, 2.0159802

Time for backsubstitution: 23.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0515097, upper bound: 1.0415729
time: 7.71 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0515111, upper bound: 1.0415720
time: 9.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 40.98 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 40.98
Output dim: 4, lower bound: -1.0415742, upper bound: 1.0396740
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 40.98
Output dim: 4, lower bound: -1.0415742, upper bound: 1.0415723
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 40.98
Output dim: 4, lower bound: -1.0396750, upper bound: 1.0515138
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 40.98
Output dim: 4, lower bound: -1.0415739, upper bound: 1.0515138
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 40.98
Output dim: 4, lower bound: -1.0497380, upper bound: 1.0497401
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 40.98
Output dim: 4, lower bound: -1.0497380, upper bound: 1.0497423
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 40.98
Output dim: 4, lower bound: -1.0515097, upper bound: 1.0415729
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 40.98
Output dim: 4, lower bound: -1.0515111, upper bound: 1.0415720

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.9177837, -5.3612571, -8.9197674, -5.3591332, -2.9613085, 2.9613545
1: -7.3795295, -4.1591473, -7.3834295, -4.1582737, -2.3885217, 2.3907049
2: -7.4732447, -4.5950756, -7.4752841, -4.5928931, -2.2793255, 2.2785294
3: -11.2570152, -7.7643356, -11.2590904, -7.7627225, -2.6230206, 2.6237414
4: 6.5990019, 8.8007011, 6.5971785, 8.8024063, -1.6183932, 1.6183267
5: -8.8986816, -5.9286923, -8.9024868, -5.9274840, -2.2557101, 2.2584085
6: -11.9953070, -8.2697411, -11.9991131, -8.2676945, -3.1366644, 3.1388807
7: -3.1971807, -0.5803645, -3.1996951, -0.5760213, -2.3621769, 2.3597708
8: -6.9589157, -3.5266891, -6.9664283, -3.5248682, -2.3657513, 2.3710566
9: -5.5121069, -3.0363903, -5.5144119, -3.0330348, -1.9758496, 1.9744506

Time for backsubstitution: 23.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396753, upper bound: 1.0396757
time: 11.36 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396753, upper bound: 1.0396780
time: 5.36 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.9358540, -5.3535461, -8.9197636, -5.3591371, -2.9809189, 2.9713445
1: -7.4024272, -4.1341619, -7.3834167, -4.1582785, -2.4177790, 2.4123409
2: -7.4954805, -4.5863051, -7.4752779, -4.5928998, -2.3041558, 2.2883704
3: -11.2654305, -7.7266445, -11.2590837, -7.7627277, -2.6341524, 2.6619484
4: 6.5251913, 8.8051538, 6.5971823, 8.8024006, -1.6853740, 1.6238644
5: -8.9050674, -5.8673158, -8.9024677, -5.9274864, -2.2626629, 2.3080761
6: -12.0103903, -8.2206955, -11.9990997, -8.2677002, -3.1534834, 3.1838064
7: -3.2457495, -0.5721505, -3.1996865, -0.5760376, -2.4028234, 2.3675408
8: -6.9742551, -3.4564888, -6.9664021, -3.5248733, -2.3837357, 2.4265187
9: -5.5852065, -3.0280557, -5.5144062, -3.0330462, -2.0314841, 1.9839749

Time for backsubstitution: 24.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396753, upper bound: 1.0415773
time: 5.89 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396753, upper bound: 1.0415772
time: 5.43 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -8.9197674, -5.3591332, -8.9511271, -5.3192234, -3.0020895, 2.9929695
1: -7.3834295, -4.1582737, -7.4077215, -4.1055984, -2.4269543, 2.4142823
2: -7.4752841, -4.5928931, -7.5366139, -4.5667973, -2.3051605, 2.3201489
3: -11.2590904, -7.7627225, -11.3118477, -7.7335520, -2.6527863, 2.6609507
4: 6.5971785, 8.8024063, 6.5208235, 8.8142223, -1.6321990, 1.6935771
5: -8.9024868, -5.9274840, -8.9148407, -5.9031534, -2.2844777, 2.2688913
6: -11.9991131, -8.2676945, -12.0138893, -8.1986408, -3.1949081, 3.1579075
7: -3.1996951, -0.5760213, -3.2570972, -0.5203695, -2.3851719, 2.4371867
8: -6.9664283, -3.5248682, -7.0044889, -3.4904008, -2.4066353, 2.4079828
9: -5.5144119, -3.0330348, -5.5690064, -3.0296226, -1.9809017, 2.0324330

Time for backsubstitution: 25.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396750, upper bound: 1.0496184
time: 9.59 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396750, upper bound: 1.0515098
time: 7.78 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -8.9197636, -5.3591371, -8.9693222, -5.3115177, -3.0120993, 3.0127344
1: -7.3834167, -4.1582785, -7.4311857, -4.0806074, -2.4339190, 2.4437950
2: -7.4752779, -4.5928998, -7.5572882, -4.5580573, -2.3149629, 2.3408437
3: -11.2590837, -7.7627277, -11.3203325, -7.6954966, -2.6911583, 2.6717641
4: 6.5971823, 8.8024006, 6.4462571, 8.8186760, -1.6377368, 1.7601869
5: -8.9024677, -5.9274864, -8.9212379, -5.8415518, -2.3331456, 2.2758656
6: -11.9990997, -8.2677002, -12.0289869, -8.1510868, -3.2267838, 3.1747417
7: -3.1996865, -0.5760376, -3.3056045, -0.5120976, -2.3929915, 2.4777324
8: -6.9664021, -3.5248733, -7.0198145, -3.4201672, -2.4602823, 2.4259727
9: -5.5144062, -3.0330462, -5.6425605, -3.0212946, -1.9904213, 2.0863254

Time for backsubstitution: 25.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415742, upper bound: 1.0496169
time: 6.86 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415742, upper bound: 1.0515110
time: 6.57 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -8.9520903, -5.3151503, -8.9334030, -5.3552794, -3.0014377, 3.0338192
1: -7.4083495, -4.1028442, -7.3939819, -4.1565304, -2.4146328, 2.4392684
2: -7.5371480, -4.5664854, -7.4769173, -4.5764313, -2.3369484, 2.3090034
3: -11.3139458, -7.7330923, -11.2612591, -7.7458291, -2.6802382, 2.6562300
4: 6.5195827, 8.8143463, 6.5639687, 8.8009062, -1.6958034, 1.6688287
5: -8.9155273, -5.9021783, -8.9007111, -5.9170809, -2.2801547, 2.2948923
6: -12.0140438, -8.1945267, -12.0112305, -8.2622318, -3.1630001, 3.2138243
7: -3.2581868, -0.5191920, -3.2157497, -0.5789135, -2.4347134, 2.4034345
8: -7.0045671, -3.4896250, -6.9600811, -3.5097632, -2.4251966, 2.4012873
9: -5.5702739, -3.0295439, -5.5349336, -3.0353355, -2.0330153, 2.0043085

Time for backsubstitution: 25.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0496158, upper bound: 1.0396737
time: 6.98 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0496173, upper bound: 1.0396731
time: 5.56 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -8.9520903, -5.3151503, -8.9512863, -5.3477535, -3.0110979, 3.0530381
1: -7.4083495, -4.1028442, -7.4166651, -4.1325521, -2.4356966, 2.4669447
2: -7.5371480, -4.5664854, -7.4985256, -4.5677285, -2.3466730, 2.3331914
3: -11.3139458, -7.7330923, -11.2696123, -7.7087088, -2.7067647, 2.6670923
4: 6.5195827, 8.8143463, 6.4912205, 8.8052511, -1.7007518, 1.7232113
5: -8.9155273, -5.9021783, -8.9070644, -5.8562822, -2.3235266, 2.3014240
6: -12.0140438, -8.1945267, -12.0257015, -8.2142525, -3.2069187, 3.2297821
7: -3.2581868, -0.5191920, -3.2640173, -0.5709370, -2.4423180, 2.4293458
8: -7.0045671, -3.4896250, -6.9751587, -3.4399588, -2.4417410, 2.4176226
9: -5.5702739, -3.0295439, -5.6070347, -3.0273280, -2.0418005, 2.0546715

Time for backsubstitution: 25.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0496158, upper bound: 1.0396731
time: 5.42 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0496173, upper bound: 1.0396736
time: 5.74 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -8.9693222, -5.3115177, -8.9197636, -5.3591371, -3.0127344, 3.0120997
1: -7.4311857, -4.0806074, -7.3834167, -4.1582785, -2.4437952, 2.4339187
2: -7.5572882, -4.5580573, -7.4752779, -4.5928998, -2.3408437, 2.3149629
3: -11.3203325, -7.6954966, -11.2590837, -7.7627277, -2.6717644, 2.6911590
4: 6.4462571, 8.8186760, 6.5971823, 8.8024006, -1.7601869, 1.6377363
5: -8.9212379, -5.8415518, -8.9024677, -5.9274864, -2.2758660, 2.3331454
6: -12.0289869, -8.1510868, -11.9990997, -8.2677002, -3.1747417, 3.2267828
7: -3.3056045, -0.5120976, -3.1996865, -0.5760376, -2.4777327, 2.3929918
8: -7.0198145, -3.4201672, -6.9664021, -3.5248733, -2.4259725, 2.4602823
9: -5.6425605, -3.0212946, -5.5144062, -3.0330462, -2.0863256, 1.9904213

Time for backsubstitution: 25.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0496158, upper bound: 1.0415725
time: 7.05 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0496158, upper bound: 1.0415754
time: 5.52 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -8.9715900, -5.3014936, -8.9553576, -5.3070869, -3.0806513, 3.0697117
1: -7.4327660, -4.0738564, -7.4131365, -4.0979872, -2.4861889, 2.4662995
2: -7.5586214, -4.5572996, -7.5399327, -4.5638609, -2.3513556, 2.3350043
3: -11.3251877, -7.6943636, -11.3187637, -7.7308908, -2.7067504, 2.7233531
4: 6.4431844, 8.8188581, 6.5160503, 8.8161049, -1.7705240, 1.7058420
5: -8.9226627, -5.8392239, -8.9200554, -5.8996010, -2.3107510, 2.3510461
6: -12.0293264, -8.1409960, -12.0180235, -8.1864300, -3.2432575, 3.2608747
7: -3.3082943, -0.5091918, -3.2622161, -0.5131311, -2.4993677, 2.4770210
8: -7.0199528, -3.4184101, -7.0121174, -3.4867320, -2.4246893, 2.4686096
9: -5.6456709, -3.0211248, -5.5743499, -3.0261121, -2.0957389, 2.0477333

Time for backsubstitution: 25.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0496172, upper bound: 1.0415750
time: 6.24 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0496172, upper bound: 1.0415721
time: 5.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 38.13 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0396753, upper bound: 1.0396757
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0396753, upper bound: 1.0396780
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0396753, upper bound: 1.0415773
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0396753, upper bound: 1.0415772
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0396750, upper bound: 1.0496184
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0396750, upper bound: 1.0515098
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0415742, upper bound: 1.0496169
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0415742, upper bound: 1.0515110
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0496158, upper bound: 1.0396737
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0496173, upper bound: 1.0396731
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0496158, upper bound: 1.0396731
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0496173, upper bound: 1.0396736
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0496158, upper bound: 1.0415725
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0496158, upper bound: 1.0415754
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0496172, upper bound: 1.0415750
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 38.13
Output dim: 4, lower bound: -1.0496172, upper bound: 1.0415721

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.60 + 550.73 = 610.33 seconds
