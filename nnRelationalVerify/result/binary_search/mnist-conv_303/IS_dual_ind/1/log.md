## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.38438213844
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801)
1: (-10.8713818, -7.8377485, -10.8713818, -7.8377485, -3.0336332, 3.0336332)
2: (-5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.7213984, 2.7213984)
3: (-6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040)
4: (-13.4648161, -9.8270741, -13.4648161, -9.8270741, -3.3610158, 3.3610163)
5: (-3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.8814673, 1.8814671)
6: (-10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.8215880, 2.8215880)
7: (-9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965)
8: (9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.6745625, 2.6745625)
9: (-7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.3670969, 3.3670969)

## BASE Result
execution time: IAR + LP analysis = 13.07 + 58.43 = 71.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.9845428, upper bound: 1.9845425


# Binary Search by BASE starts (time budget: 3528.50 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.359529972076416
rel_dist={8: [-1.3847687820365167, 1.3847690605870273]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.1324596405029297
rel_dist={8: [-1.0332529041562815, 1.0332525329195708]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=2.2081499099731445
rel_dist={8: [-1.1524706431871987, 1.1524726628995694]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=2.2838401794433594
rel_dist={8: [-1.269933294744563, 1.2699326298994738]}

## Binary Search Result
Binary search time: 195.64 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual_ind) starts
Time budget: 3332.86 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6835984, upper bound: 1.7089538
time: 5.23 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090377, upper bound: 1.7090379
time: 7.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.72
Output dim: 8, lower bound: -1.6835984, upper bound: 1.7089538
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.72
Output dim: 8, lower bound: -1.7090377, upper bound: 1.7090379

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.4018459, -5.7667723, -8.4365664, -5.7612162, -2.6406298, 2.6697941
1: -10.8537655, -7.9222918, -10.8703604, -7.8470478, -2.8555894, 2.7983685
2: -5.0640192, -2.3662400, -5.0786452, -2.3596354, -2.5588436, 2.5583825
3: -6.0806208, -2.8903439, -6.1212358, -2.8829832, -3.1976376, 3.2308919
4: -13.4631758, -9.8373919, -13.4647236, -9.8281202, -2.9806690, 2.9850576
5: -3.5557179, -1.5313683, -3.5625110, -1.4863625, -1.6756098, 1.6418672
6: -10.8669634, -8.1119480, -10.8799944, -8.0364714, -2.5185785, 2.4624271
7: -9.6405087, -6.2830420, -9.6593380, -6.2762232, -3.3642855, 3.3762960
8: 9.3482761, 11.9452114, 9.2870598, 11.9537258, -2.5167751, 2.5691695
9: -7.8587122, -4.4339256, -7.8696723, -4.4306927, -3.1139965, 3.1238356

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6834468, upper bound: 1.7010465
time: 6.51 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6835845, upper bound: 1.7089410
time: 7.23 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.4408264, -5.7609487, -8.4408283, -5.7609482, -2.6798782, 2.6798797
1: -10.8713799, -7.8377523, -10.8713818, -7.8377485, -2.8853192, 2.8299923
2: -5.0804291, -2.3590336, -5.0804310, -2.3590326, -2.5722761, 2.5735354
3: -6.1261244, -2.8826213, -6.1261268, -2.8826227, -3.2435017, 3.2435055
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -3.0141525, 2.9943209
5: -3.5627644, -1.4808015, -3.5627654, -1.4808009, -1.6920600, 1.6678989
6: -10.8806114, -8.0272913, -10.8806124, -8.0272884, -2.5431190, 2.4884329
7: -9.6614304, -6.2756400, -9.6614361, -6.2756395, -3.3857908, 3.3857961
8: 9.2794495, 11.9540062, 9.2794437, 11.9540062, -2.5493817, 2.5866008
9: -7.8706479, -4.4301829, -7.8706493, -4.4301844, -3.1460524, 3.1288066

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7089539, upper bound: 1.6835990
time: 5.57 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7089541, upper bound: 1.7090381
time: 6.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.45
Output dim: 8, lower bound: -1.6834468, upper bound: 1.7010465
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.45
Output dim: 8, lower bound: -1.6835845, upper bound: 1.7089410
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.45
Output dim: 8, lower bound: -1.7089539, upper bound: 1.6835990
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.45
Output dim: 8, lower bound: -1.7089541, upper bound: 1.7090381

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.3985720, -5.7669649, -8.4007025, -5.7633276, -2.6352444, 2.6337376
1: -10.8529272, -7.9225717, -10.8613014, -7.8502116, -2.8341775, 2.7746329
2: -5.0618043, -2.3667021, -5.0543079, -2.3645720, -2.5499482, 2.5314665
3: -6.0796609, -2.8908565, -6.1106424, -2.8883703, -3.1912906, 3.2197859
4: -13.4630308, -9.8390446, -13.4631510, -9.8457870, -2.9582405, 2.9815910
5: -3.5552819, -1.5345352, -3.5581057, -1.5210671, -1.6396687, 1.6327221
6: -10.8665857, -8.1163626, -10.8759747, -8.0849161, -2.4657307, 2.4527202
7: -9.6386528, -6.2833967, -9.6390495, -6.2798514, -3.3588014, 3.3556528
8: 9.3497791, 11.9449768, 9.3033619, 11.9512682, -2.5125256, 2.5531878
9: -7.8581133, -4.4345016, -7.8634005, -4.4368649, -3.0985193, 3.1058044

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6833625, upper bound: 1.6927016
time: 11.65 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6834337, upper bound: 1.7010337
time: 7.10 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.4018431, -5.7667723, -8.4451294, -5.7172432, -2.6845999, 2.6783571
1: -10.8537645, -7.9222908, -10.8836374, -7.8315849, -2.8665705, 2.7994380
2: -5.0640149, -2.3662419, -5.0823359, -2.3191953, -2.5932584, 2.5634079
3: -6.0806179, -2.8903453, -6.1500888, -2.8799546, -3.2006633, 3.2597435
4: -13.4631739, -9.8373947, -13.4878559, -9.8176260, -2.9959946, 3.0086856
5: -3.5557179, -1.5313737, -3.6048863, -1.4835569, -1.6743859, 1.6655815
6: -10.8669653, -8.1119499, -10.9468307, -8.0168819, -2.5419817, 2.4994030
7: -9.6405010, -6.2830405, -9.6823311, -6.2425041, -3.3979969, 3.3992906
8: 9.3482800, 11.9452133, 9.2799883, 11.9702320, -2.5340400, 2.5753899
9: -7.8587093, -4.4339275, -7.9081106, -4.4250660, -3.1114330, 3.1673703

Time for backsubstitution: 12.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6835004, upper bound: 1.7005941
time: 6.33 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6835714, upper bound: 1.7089260
time: 6.46 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.4408264, -5.7609487, -8.4018459, -5.7667723, -2.6740541, 2.6408973
1: -10.8713799, -7.8377523, -10.8537655, -7.9222918, -2.7996702, 2.8608043
2: -5.0804291, -2.3590336, -5.0640192, -2.3662400, -2.5598869, 2.5595345
3: -6.1261244, -2.8826213, -6.0806208, -2.8903439, -3.2357805, 3.1979995
4: -13.4648161, -9.8270741, -13.4631758, -9.8373919, -2.9837377, 2.9812653
5: -3.5627644, -1.4808015, -3.5557179, -1.5313683, -1.6423240, 1.6764445
6: -10.8806114, -8.0272913, -10.8669634, -8.1119480, -2.4631643, 2.5234964
7: -9.6614304, -6.2756400, -9.6405087, -6.2830420, -3.3783884, 3.3648686
8: 9.2794495, 11.9540062, 9.3482761, 11.9452114, -2.5768700, 2.5171368
9: -7.8706479, -4.4301829, -7.8587122, -4.4339256, -3.1237373, 3.1135430

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7088685, upper bound: 1.6752531
time: 10.08 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7089396, upper bound: 1.6835849
time: 6.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.4408264, -5.7609487, -8.4408264, -5.7609487, -2.6798778, 2.6798778
1: -10.8713799, -7.8377523, -10.8713799, -7.8377523, -2.8299923, 2.8299918
2: -5.0804291, -2.3590336, -5.0804291, -2.3590336, -2.5735354, 2.5735350
3: -6.1261244, -2.8826213, -6.1261244, -2.8826213, -3.2435031, 3.2435031
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -3.0141521, 3.0141521
5: -3.5627644, -1.4808015, -3.5627644, -1.4808015, -1.6678987, 1.6678989
6: -10.8806114, -8.0272913, -10.8806114, -8.0272913, -2.4884329, 2.4884329
7: -9.6614304, -6.2756400, -9.6614304, -6.2756400, -3.3857903, 3.3857903
8: 9.2794495, 11.9540062, 9.2794495, 11.9540062, -2.5493822, 2.5493824
9: -7.8706479, -4.4301829, -7.8706479, -4.4301829, -3.1460514, 3.1460514

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7088697, upper bound: 1.6754330
time: 6.67 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7089410, upper bound: 1.6837647
time: 8.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.69 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.69
Output dim: 8, lower bound: -1.6833625, upper bound: 1.6927016
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.69
Output dim: 8, lower bound: -1.6834337, upper bound: 1.7010337
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.69
Output dim: 8, lower bound: -1.6835004, upper bound: 1.7005941
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.69
Output dim: 8, lower bound: -1.6835714, upper bound: 1.7089260
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.69
Output dim: 8, lower bound: -1.7088685, upper bound: 1.6752531
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.69
Output dim: 8, lower bound: -1.7089396, upper bound: 1.6835849
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.69
Output dim: 8, lower bound: -1.7088697, upper bound: 1.6754330
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.69
Output dim: 8, lower bound: -1.7089410, upper bound: 1.6837647

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.3955956, -5.7746921, -8.4005432, -5.7637310, -2.6318645, 2.6258512
1: -10.8499861, -7.9286566, -10.8611488, -7.8505278, -2.8190856, 2.7676816
2: -5.0603089, -2.3743072, -5.0542259, -2.3649859, -2.5484776, 2.5239267
3: -6.0606542, -2.8949156, -6.1096487, -2.8885820, -3.1720722, 3.2147331
4: -13.4334679, -9.8451042, -13.4616108, -9.8461208, -2.9269631, 2.9748785
5: -3.5548544, -1.5446749, -3.5580835, -1.5216160, -1.6351728, 1.6161604
6: -10.8635159, -8.1353130, -10.8758163, -8.0859118, -2.4632473, 2.4319170
7: -9.5951958, -6.2867045, -9.6367741, -6.2800236, -3.3151722, 3.3500695
8: 9.3527098, 11.9363127, 9.3035183, 11.9508076, -2.5069604, 2.5421133
9: -7.8379660, -4.4361887, -7.8622823, -4.4369555, -3.0652552, 3.0962558

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6751008, upper bound: 1.6927005
time: 6.37 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6751008, upper bound: 1.6927007
time: 7.18 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.4088154, -5.7661219, -8.4007034, -5.7633290, -2.6454864, 2.6345816
1: -10.8588314, -7.9194312, -10.8612995, -7.8502159, -2.8411713, 2.7772622
2: -5.0739427, -2.3645043, -5.0543060, -2.3645759, -2.5622497, 2.5331960
3: -6.0856161, -2.8736746, -6.1106381, -2.8883729, -3.1972432, 3.2369635
4: -13.4641256, -9.7979193, -13.4631405, -9.8457890, -2.9580212, 3.0113168
5: -3.5679002, -1.5336913, -3.5581069, -1.5210676, -1.6383781, 1.6378808
6: -10.8794603, -8.1112747, -10.8759737, -8.0849247, -2.4715352, 2.4567840
7: -9.6477833, -6.2363667, -9.6390448, -6.2798553, -3.3679280, 3.4026780
8: 9.3345814, 11.9465036, 9.3033638, 11.9512644, -2.5255370, 2.5578022
9: -7.8637433, -4.4229994, -7.8633928, -4.4368663, -3.1083107, 3.1104727

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6834337, upper bound: 1.6756729
time: 6.89 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6834337, upper bound: 1.7010349
time: 7.75 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.3988724, -5.7744985, -8.4449749, -5.7176476, -2.6812248, 2.6704764
1: -10.8508282, -7.9283791, -10.8834858, -7.8319087, -2.8514071, 2.7924862
2: -5.0625210, -2.3738484, -5.0822544, -2.3196106, -2.5916491, 2.5558679
3: -6.0616097, -2.8944054, -6.1490841, -2.8801630, -3.1814468, 3.2546787
4: -13.4336138, -9.8434649, -13.4863138, -9.8179512, -2.9647160, 3.0019736
5: -3.5552902, -1.5415133, -3.6048656, -1.4841067, -1.6695219, 1.6494384
6: -10.8638916, -8.1308851, -10.9466705, -8.0178699, -2.5387187, 2.4786465
7: -9.5970592, -6.2863488, -9.6800632, -6.2426744, -3.3543849, 3.3937144
8: 9.3512106, 11.9365473, 9.2801447, 11.9697723, -2.5284748, 2.5643167
9: -7.8385582, -4.4356108, -7.9069777, -4.4251528, -3.0781403, 3.1573544

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6752386, upper bound: 1.7005950
time: 7.76 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6752387, upper bound: 1.7005933
time: 9.15 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.4120903, -5.7659292, -8.4451294, -5.7172451, -2.6948452, 2.6792002
1: -10.8596678, -7.9191570, -10.8836384, -7.8315878, -2.8693976, 2.8020639
2: -5.0761538, -2.3640404, -5.0823340, -2.3191972, -2.5963354, 2.5651405
3: -6.0865684, -2.8731644, -6.1500835, -2.8799551, -3.2066133, 3.2769191
4: -13.4642706, -9.7962818, -13.4878445, -9.8176279, -2.9957747, 3.0186799
5: -3.5683343, -1.5305295, -3.6048861, -1.4835573, -1.6725397, 1.6668464
6: -10.8798370, -8.1068649, -10.9468279, -8.0168867, -2.5433416, 2.5035155
7: -9.6496325, -6.2360120, -9.6823235, -6.2425065, -3.4071259, 3.4463115
8: 9.3330860, 11.9467382, 9.2799902, 11.9702263, -2.5470514, 2.5800028
9: -7.8643274, -4.4224248, -7.9081049, -4.4250660, -3.1212268, 3.1619523

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6835716, upper bound: 1.6835706
time: 6.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6835716, upper bound: 1.7089275
time: 5.68 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.4379053, -5.7686729, -8.4016838, -5.7671766, -2.6707287, 2.6330109
1: -10.8684692, -7.8437529, -10.8536100, -7.9226141, -2.7846098, 2.8538053
2: -5.0789452, -2.3666506, -5.0639353, -2.3666573, -2.5584221, 2.5519803
3: -6.1072655, -2.8866818, -6.0796180, -2.8905535, -3.2167120, 3.1929362
4: -13.4352531, -9.8332701, -13.4616346, -9.8377190, -2.9524541, 2.9746616
5: -3.5623391, -1.4909371, -3.5556955, -1.5319189, -1.6378267, 1.6603110
6: -10.8775501, -8.0460224, -10.8668022, -8.1129398, -2.4606848, 2.5028298
7: -9.6181622, -6.2789412, -9.6382294, -6.2832122, -3.3349500, 3.3592882
8: 9.2823687, 11.9453373, 9.3484335, 11.9447527, -2.5710697, 2.5060635
9: -7.8503776, -4.4318485, -7.8575935, -4.4340134, -3.0903234, 3.1040096

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7087163, upper bound: 1.6673418
time: 7.35 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7088541, upper bound: 1.6752392
time: 7.63 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.4511385, -5.7600975, -8.4018459, -5.7667737, -2.6843648, 2.6417484
1: -10.8772697, -7.8346090, -10.8537645, -7.9222956, -2.8066430, 2.8633366
2: -5.0925636, -2.3568187, -5.0640173, -2.3662438, -2.5721827, 2.5612741
3: -6.1320682, -2.8654466, -6.0806146, -2.8903461, -3.2417221, 3.2151680
4: -13.4659081, -9.7859364, -13.4631634, -9.8373947, -2.9835100, 3.0121439
5: -3.5753779, -1.4799509, -3.5557175, -1.5313709, -1.6526902, 1.6777132
6: -10.8934755, -8.0221643, -10.8669624, -8.1119499, -2.4761462, 2.5276415
7: -9.6705675, -6.2286172, -9.6404982, -6.2830415, -3.3875260, 3.4118810
8: 9.2642775, 11.9555416, 9.3482780, 11.9452076, -2.5780454, 2.5217459
9: -7.8761716, -4.4186754, -7.8587055, -4.4339256, -3.1335092, 3.1182227

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7087875, upper bound: 1.6756727
time: 7.10 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7089252, upper bound: 1.6835705
time: 8.11 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.4379053, -5.7686729, -8.4406691, -5.7613506, -2.6765547, 2.6719961
1: -10.8684692, -7.8437529, -10.8712273, -7.8380675, -2.8149328, 2.8231225
2: -5.0789452, -2.3666506, -5.0803475, -2.3594475, -2.5720701, 2.5659809
3: -6.1072655, -2.8866818, -6.1251297, -2.8828318, -3.2244337, 3.2384479
4: -13.4352531, -9.8332701, -13.4632711, -9.8273983, -2.9828732, 3.0075490
5: -3.5623391, -1.4909371, -3.5627427, -1.4813519, -1.6634026, 1.6513374
6: -10.8775501, -8.0460224, -10.8804522, -8.0282755, -2.4859266, 2.4675441
7: -9.6181622, -6.2789412, -9.6591663, -6.2758107, -3.3423514, 3.3802252
8: 9.2823687, 11.9453373, 9.2796040, 11.9535484, -2.5438094, 2.5383084
9: -7.8503776, -4.4318485, -7.8695278, -4.4302707, -3.1126680, 3.1364918

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7088018, upper bound: 1.6675261
time: 6.85 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7089388, upper bound: 1.6754187
time: 8.39 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.4511385, -5.7600975, -8.4408264, -5.7609487, -2.6901898, 2.6807289
1: -10.8772697, -7.8346090, -10.8713789, -7.8377528, -2.8369632, 2.8326240
2: -5.0925636, -2.3568187, -5.0804291, -2.3590355, -2.5858326, 2.5752754
3: -6.1320682, -2.8654466, -6.1261201, -2.8826237, -3.2494445, 3.2606735
4: -13.4659081, -9.7859364, -13.4648046, -9.8270760, -3.0139265, 3.0442030
5: -3.5753779, -1.4799509, -3.5627639, -1.4808037, -1.6782646, 1.6730517
6: -10.8934755, -8.0221643, -10.8806114, -8.0272980, -2.5014143, 2.4924181
7: -9.6705675, -6.2286172, -9.6614265, -6.2756400, -3.3949275, 3.4328094
8: 9.2642775, 11.9555416, 9.2794514, 11.9540014, -2.5623484, 2.5539913
9: -7.8761716, -4.4186754, -7.8706446, -4.4301844, -3.1558185, 3.1507006

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7088729, upper bound: 1.6758567
time: 6.69 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090102, upper bound: 1.6837503
time: 5.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.09 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.6751008, upper bound: 1.6927005
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.6751008, upper bound: 1.6927007
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.6834337, upper bound: 1.6756729
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.6834337, upper bound: 1.7010349
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.6752386, upper bound: 1.7005950
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.6752387, upper bound: 1.7005933
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.6835716, upper bound: 1.6835706
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.6835716, upper bound: 1.7089275
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.7087163, upper bound: 1.6673418
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.7088541, upper bound: 1.6752392
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.7087875, upper bound: 1.6756727
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.7089252, upper bound: 1.6835705
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.7088018, upper bound: 1.6675261
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.7089388, upper bound: 1.6754187
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.7088729, upper bound: 1.6758567
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.09
Output dim: 8, lower bound: -1.7090102, upper bound: 1.6837503

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.3955956, -5.7746921, -8.3977451, -5.7710547, -2.6245408, 2.6230531
1: -10.8499861, -7.9286566, -10.8583727, -7.8562031, -2.8136187, 2.7539983
2: -5.0603089, -2.3743072, -5.0528202, -2.3721805, -2.5414190, 2.5229547
3: -6.0606542, -2.8949156, -6.0918088, -2.8924317, -3.1682224, 3.1968932
4: -13.4334679, -9.8451042, -13.4335899, -9.8521681, -2.9225197, 2.9458473
5: -3.5548544, -1.5446749, -3.5576768, -1.5312061, -1.6223657, 1.6153504
6: -10.8635159, -8.1353130, -10.8729067, -8.1038389, -2.4439545, 2.4306812
7: -9.5951958, -6.2867045, -9.5956192, -6.2831473, -3.3120484, 3.3089147
8: 9.3527098, 11.9363127, 9.3062897, 11.9426031, -2.4968529, 2.5375051
9: -7.8379660, -4.4361887, -7.8431888, -4.4385672, -3.0633507, 3.0706244

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6673427, upper bound: 1.6927034
time: 6.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6673427, upper bound: 1.6927015
time: 6.50 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.3955956, -5.7746921, -8.4109669, -5.7624750, -2.6331205, 2.6362748
1: -10.8499861, -7.9286566, -10.8670807, -7.8470221, -2.8229494, 2.7727604
2: -5.0603089, -2.3743072, -5.0648866, -2.3623848, -2.5509524, 2.5346627
3: -6.0606542, -2.8949156, -6.1166816, -2.8715663, -3.1890879, 3.2217660
4: -13.4334679, -9.8451042, -13.4641523, -9.8047438, -2.9583585, 2.9774413
5: -3.5548544, -1.5446749, -3.5707231, -1.5202165, -1.6348684, 1.6291628
6: -10.8635159, -8.1353130, -10.8888464, -8.0798187, -2.4685450, 2.4450686
7: -9.5951958, -6.2867045, -9.6481171, -6.2328730, -3.3623228, 3.3614125
8: 9.3527098, 11.9363127, 9.2881880, 11.9525871, -2.5067654, 2.5499275
9: -7.8379660, -4.4361887, -7.8690276, -4.4253531, -3.0764532, 3.1010809

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6673428, upper bound: 1.6927016
time: 6.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6673428, upper bound: 1.6927020
time: 5.51 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4088154, -5.7661219, -8.3664303, -5.7689090, -2.6399064, 2.6003084
1: -10.8588314, -7.9194312, -10.8445721, -7.9255071, -2.7651525, 2.7588358
2: -5.0739427, -2.3645043, -5.0401182, -2.3712897, -2.5495596, 2.5212967
3: -6.0856161, -2.8736746, -6.0700102, -2.8957303, -3.1898859, 3.1963356
4: -13.4641256, -9.7979193, -13.4615612, -9.8550196, -2.9492118, 2.9983695
5: -3.5679002, -1.5336913, -3.5512590, -1.5657668, -1.6058493, 1.6273184
6: -10.8794603, -8.1112747, -10.8628483, -8.1596689, -2.4092531, 2.4429083
7: -9.6477833, -6.2363667, -9.6205635, -6.2868404, -3.3609428, 3.3841968
8: 9.3345814, 11.9465036, 9.3645554, 11.9426365, -2.5159612, 2.4961360
9: -7.8637433, -4.4229994, -7.8520842, -4.4402099, -3.1039672, 3.0958939

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6756749, upper bound: 1.6756725
time: 18.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6756749, upper bound: 1.6756729
time: 9.09 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4088154, -5.7661219, -8.4049206, -5.7630577, -2.6457577, 2.6387987
1: -10.8588314, -7.9194312, -10.8623352, -7.8409052, -2.8418126, 2.7785668
2: -5.0739427, -2.3645043, -5.0560708, -2.3639598, -2.5629539, 2.5346973
3: -6.0856161, -2.8736746, -6.1155381, -2.8880136, -3.1976025, 3.2418635
4: -13.4641256, -9.7979193, -13.4632349, -9.8448582, -2.9585233, 3.0095730
5: -3.5679002, -1.5336913, -3.5583665, -1.5155203, -1.6387074, 1.6383357
6: -10.8794603, -8.1112747, -10.8766003, -8.0758133, -2.4724107, 2.4575348
7: -9.6477833, -6.2363667, -9.6411057, -6.2792563, -3.3685269, 3.4047389
8: 9.3345814, 11.9465036, 9.2957573, 11.9515524, -2.5259161, 2.5619833
9: -7.8637433, -4.4229994, -7.8644094, -4.4363456, -3.1078615, 3.1104298

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6756749, upper bound: 1.7010335
time: 10.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6756749, upper bound: 1.7010335
time: 6.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.3988724, -5.7744985, -8.4423008, -5.7249722, -2.6739001, 2.6678023
1: -10.8508282, -7.9283791, -10.8807192, -7.8376865, -2.8459582, 2.7788172
2: -5.0625210, -2.3738484, -5.0808611, -2.3268194, -2.5847654, 2.5548847
3: -6.0616097, -2.8944054, -6.1310329, -2.8840148, -3.1775949, 3.2366276
4: -13.4336138, -9.8434649, -13.4582911, -9.8238182, -2.9602516, 2.9729400
5: -3.5552902, -1.5415133, -3.6044636, -1.4936986, -1.6575201, 1.6486481
6: -10.8638916, -8.1308851, -10.9437733, -8.0357170, -2.5198536, 2.4771085
7: -9.5970592, -6.2863488, -9.6390572, -6.2458048, -3.3512545, 3.3527083
8: 9.3512106, 11.9365473, 9.2829018, 11.9615660, -2.5183725, 2.5597076
9: -7.8385582, -4.4356108, -7.8875818, -4.4267340, -3.0762868, 3.1319928

Time for backsubstitution: 12.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6673411, upper bound: 1.7004586
time: 6.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6673411, upper bound: 1.6927034
time: 6.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.3988724, -5.7744985, -8.4554195, -5.7163982, -2.6824741, 2.6809211
1: -10.8508282, -7.9283791, -10.8894329, -7.8283968, -2.8550777, 2.7975826
2: -5.0625210, -2.3738484, -5.0929103, -2.3169765, -2.5939088, 2.5665941
3: -6.0616097, -2.8944054, -6.1561503, -2.8631597, -3.1984501, 3.2617450
4: -13.4336138, -9.8434649, -13.4888573, -9.7765923, -2.9899225, 3.0045357
5: -3.5552902, -1.5415133, -3.6174984, -1.4827027, -1.6690415, 1.6502296
6: -10.8638916, -8.1308851, -10.9596920, -8.0115499, -2.5435085, 2.4801693
7: -9.5970592, -6.2863488, -9.6916437, -6.1954865, -3.4015727, 3.4052949
8: 9.3512106, 11.9365473, 9.2648153, 11.9715557, -2.5282927, 2.5722492
9: -7.8385582, -4.4356108, -7.9137630, -4.4135051, -3.0894318, 3.1620455

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6673412, upper bound: 1.7004566
time: 9.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6673412, upper bound: 1.6935988
time: 8.37 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4120903, -5.7659292, -8.4104805, -5.7227778, -2.6893125, 2.6445513
1: -10.8596678, -7.9191570, -10.8670692, -7.9061656, -2.8204722, 2.7834311
2: -5.0761538, -2.3640404, -5.0678639, -2.3257823, -2.5833702, 2.5536420
3: -6.0865684, -2.8731644, -6.1102157, -2.8873544, -3.1992140, 3.2370512
4: -13.4642706, -9.7962818, -13.4863157, -9.8274202, -2.9872322, 3.0057356
5: -3.5683343, -1.5305295, -3.5981362, -1.5283642, -1.6415012, 1.6559670
6: -10.8798370, -8.1068649, -10.9338808, -8.0919418, -2.4910126, 2.4895413
7: -9.6496325, -6.2360120, -9.6640511, -6.2492342, -3.4003983, 3.4280391
8: 9.3330860, 11.9467382, 9.3411732, 11.9617348, -2.5376835, 2.5183063
9: -7.8643274, -4.4224248, -7.8970013, -4.4283123, -3.1169410, 3.1471331

Time for backsubstitution: 12.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6756733, upper bound: 1.6834336
time: 6.17 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6756733, upper bound: 1.6765732
time: 11.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4120903, -5.7659292, -8.4493847, -5.7169771, -2.6951132, 2.6834555
1: -10.8596678, -7.9191570, -10.8846531, -7.8223672, -2.8700404, 2.8033400
2: -5.0761538, -2.3640404, -5.0841136, -2.3185983, -2.5957475, 2.5665860
3: -6.0865684, -2.8731644, -6.1548877, -2.8795888, -3.2069795, 3.2817233
4: -13.4642706, -9.7962818, -13.4879370, -9.8166380, -2.9962869, 3.0169413
5: -3.5683343, -1.5305295, -3.6051385, -1.4780290, -1.6728835, 1.6628108
6: -10.8798370, -8.1068649, -10.9474363, -8.0077496, -2.5442777, 2.4963663
7: -9.6496325, -6.2360120, -9.6843805, -6.2419271, -3.4077053, 3.4483685
8: 9.3330860, 11.9467382, 9.2723827, 11.9705029, -2.5474110, 2.5843017
9: -7.8643274, -4.4224248, -7.9090910, -4.4245596, -3.1207676, 3.1606483

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6756733, upper bound: 1.7087886
time: 6.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6756733, upper bound: 1.7010348
time: 11.58 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.4345875, -5.7688656, -8.3662682, -5.7693105, -2.6652770, 2.5974026
1: -10.8676443, -7.8440304, -10.8444176, -7.9258261, -2.7632570, 2.8290389
2: -5.0767050, -2.3670998, -5.0400362, -2.3717027, -2.5493908, 2.5253625
3: -6.1063094, -2.8871922, -6.0690145, -2.8959363, -3.2103732, 3.1818223
4: -13.4351101, -9.8349485, -13.4600296, -9.8553343, -2.9298143, 2.9711590
5: -3.5619082, -1.4941276, -3.5512376, -1.5663137, -1.6020360, 1.6491004
6: -10.8771820, -8.0505142, -10.8626900, -8.1606722, -2.4083033, 2.4899809
7: -9.6162720, -6.2792850, -9.6182890, -6.2870073, -3.3292646, 3.3390040
8: 9.2838726, 11.9451103, 9.3647108, 11.9421864, -2.5662007, 2.4901965
9: -7.8498201, -4.4324164, -7.8509750, -4.4403019, -3.0747595, 3.0855217

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7009617, upper bound: 1.6673415
time: 6.18 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7009617, upper bound: 1.6673437
time: 5.36 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.4379005, -5.7686763, -8.4103241, -5.7231803, -2.7147202, 2.6416478
1: -10.8684673, -7.8437529, -10.8669176, -7.9064865, -2.8185716, 2.8483963
2: -5.0789413, -2.3666530, -5.0677838, -2.3261933, -2.5870171, 2.5575354
3: -6.1072636, -2.8866837, -6.1092081, -2.8875637, -3.2196999, 3.2225244
4: -13.4352541, -9.8332748, -13.4847851, -9.8277378, -2.9676027, 2.9983215
5: -3.5623384, -1.4909455, -3.5981154, -1.5289125, -1.6375456, 1.6608425
6: -10.8775520, -8.0460339, -10.9337215, -8.0929327, -2.4899383, 2.5033767
7: -9.6181583, -6.2789421, -9.6617832, -6.2494040, -3.3687544, 3.3828411
8: 9.2823734, 11.9453363, 9.3413277, 11.9612808, -2.5726314, 2.5123489
9: -7.8503776, -4.4318519, -7.8958769, -4.4284034, -3.0876570, 3.1475677

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7009615, upper bound: 1.6751013
time: 6.21 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7009617, upper bound: 1.6752411
time: 5.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4478188, -5.7602901, -8.3664303, -5.7689090, -2.6789098, 2.6061401
1: -10.8764448, -7.8348856, -10.8445721, -7.9255071, -2.7852902, 2.8385677
2: -5.0903230, -2.3572702, -5.0401182, -2.3712897, -2.5631495, 2.5346537
3: -6.1311173, -2.8659589, -6.0700102, -2.8957303, -3.2353871, 3.2040513
4: -13.4657650, -9.7876177, -13.4615612, -9.8550196, -2.9608684, 3.0076277
5: -3.5749500, -1.4831415, -3.5512590, -1.5657668, -1.6168981, 1.6664953
6: -10.8931055, -8.0266409, -10.8628483, -8.1596689, -2.4237704, 2.5147924
7: -9.6686916, -6.2289610, -9.6205635, -6.2868404, -3.3818512, 3.3916025
8: 9.2657814, 11.9553146, 9.3645554, 11.9426365, -2.5731754, 2.5058789
9: -7.8756132, -4.4192410, -7.8520842, -4.4402099, -3.1179948, 3.0997286

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7010329, upper bound: 1.6756730
time: 6.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7010332, upper bound: 1.6756744
time: 5.15 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4511318, -5.7600980, -8.4104805, -5.7227778, -2.7283540, 2.6503825
1: -10.8772688, -7.8346105, -10.8670692, -7.9061656, -2.8406072, 2.8579278
2: -5.0925589, -2.3568196, -5.0678639, -2.3257823, -2.5917087, 2.5668306
3: -6.1320648, -2.8654487, -6.1102157, -2.8873544, -3.2447104, 3.2447670
4: -13.4659081, -9.7859449, -13.4863157, -9.8274202, -2.9986579, 3.0149634
5: -3.5753775, -1.4799604, -3.5981362, -1.5283642, -1.6524076, 1.6782539
6: -10.8934755, -8.0221758, -10.9338808, -8.0919418, -2.5054574, 2.5281882
7: -9.6705618, -6.2286177, -9.6640511, -6.2492342, -3.4213276, 3.4354334
8: 9.2642813, 11.9555407, 9.3411732, 11.9617348, -2.5796084, 2.5280321
9: -7.8761692, -4.4186764, -7.8970013, -4.4283123, -3.1309237, 3.1521599

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7010329, upper bound: 1.6834329
time: 10.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7010329, upper bound: 1.6752400
time: 7.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.4345875, -5.7688656, -8.4047604, -5.7634592, -2.6711283, 2.6358948
1: -10.8676443, -7.8440304, -10.8621817, -7.8412151, -2.7935867, 2.7993598
2: -5.0767050, -2.3670998, -5.0559912, -2.3643723, -2.5631704, 2.5391543
3: -6.1063094, -2.8871922, -6.1145506, -2.8882215, -3.2180879, 3.2273583
4: -13.4351101, -9.8349485, -13.4617062, -9.8451900, -2.9602063, 3.0040941
5: -3.5619082, -1.4941276, -3.5583451, -1.5160671, -1.6275251, 1.6421993
6: -10.8771820, -8.0505142, -10.8764400, -8.0767984, -2.4341359, 2.4578810
7: -9.6162720, -6.2792850, -9.6388378, -6.2794242, -3.3368478, 3.3595529
8: 9.2838726, 11.9451103, 9.2959118, 11.9510994, -2.5395708, 2.5223348
9: -7.8498201, -4.4324164, -7.8632994, -4.4364343, -3.0970821, 3.1185226

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7010473, upper bound: 1.6675264
time: 6.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7010466, upper bound: 1.6675260
time: 4.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.39 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6673427, upper bound: 1.6927034
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6673427, upper bound: 1.6927015
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6673428, upper bound: 1.6927016
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6673428, upper bound: 1.6927020
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6756749, upper bound: 1.6756725
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6756749, upper bound: 1.6756729
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6756749, upper bound: 1.7010335
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6756749, upper bound: 1.7010335
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6673411, upper bound: 1.7004586
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6673411, upper bound: 1.6927034
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6673412, upper bound: 1.7004566
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6673412, upper bound: 1.6935988
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6756733, upper bound: 1.6834336
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6756733, upper bound: 1.6765732
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6756733, upper bound: 1.7087886
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.6756733, upper bound: 1.7010348
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.7009617, upper bound: 1.6673415
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.7009617, upper bound: 1.6673437
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.7009615, upper bound: 1.6751013
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.7009617, upper bound: 1.6752411
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.7010329, upper bound: 1.6756730
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.7010332, upper bound: 1.6756744
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.7010329, upper bound: 1.6834329
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.7010329, upper bound: 1.6752400
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.7010473, upper bound: 1.6675264
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 8, lower bound: -1.7010466, upper bound: 1.6675260
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 8, lower bound: -1.7089388, upper bound: 1.6754187
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 8, lower bound: -1.7088729, upper bound: 1.6758567
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.39
Output dim: 8, lower bound: -1.7090102, upper bound: 1.6837503
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.5866003036499023
rel_dist={8: [-1.7090580688748975, 1.709057719074428]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771588, upper bound: 1.4968898
time: 6.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969226, upper bound: 1.4969250
time: 5.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.76 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.76
Output dim: 8, lower bound: -1.4771588, upper bound: 1.4968898
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.76
Output dim: 8, lower bound: -1.4969226, upper bound: 1.4969250

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.4018459, -5.7667723, -8.4348211, -5.7613282, -2.5426764, 2.5684617
1: -10.8537655, -7.9222918, -10.8699417, -7.8508549, -2.6956215, 2.6446996
2: -5.0640192, -2.3662400, -5.0779166, -2.3598819, -2.3930798, 2.3917656
3: -6.0806208, -2.8903439, -6.1192336, -2.8831315, -3.1974893, 3.2288897
4: -13.4631758, -9.8373919, -13.4646845, -9.8285551, -2.7365859, 2.7414854
5: -3.5557179, -1.5313683, -3.5624058, -1.4886398, -1.5420057, 1.5154083
6: -10.8669634, -8.1119480, -10.8797417, -8.0402298, -2.3244352, 2.2764790
7: -9.6405087, -6.2830420, -9.6584826, -6.2764664, -3.3640423, 3.3754406
8: 9.3482761, 11.9452114, 9.2901764, 11.9536123, -2.3652534, 2.4146342
9: -7.8587122, -4.4339256, -7.8692694, -4.4309006, -2.9555788, 2.9651461

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4769124, upper bound: 1.4906706
time: 5.88 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771480, upper bound: 1.4968784
time: 10.28 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.4408264, -5.7609487, -8.4408283, -5.7609482, -2.5665846, 2.5827990
1: -10.8713799, -7.8377523, -10.8713818, -7.8377485, -2.7323327, 2.6696162
2: -5.0804291, -2.3590336, -5.0804310, -2.3590326, -2.4068308, 2.4080086
3: -6.1261244, -2.8826213, -6.1261268, -2.8826227, -3.2435017, 3.2435055
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7692547, 2.7507858
5: -3.5627644, -1.4808015, -3.5627654, -1.4808009, -1.5657887, 1.5384035
6: -10.8806114, -8.0272913, -10.8806124, -8.0272884, -2.3574739, 2.2954426
7: -9.6614304, -6.2756400, -9.6614361, -6.2756395, -3.3857908, 3.3857961
8: 9.2794495, 11.9540062, 9.2794437, 11.9540062, -2.3930421, 2.4352202
9: -7.8706479, -4.4301829, -7.8706493, -4.4301844, -2.9866476, 2.9706144

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4968898, upper bound: 1.4771608
time: 5.24 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4968898, upper bound: 1.4969229
time: 7.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.41 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 8, lower bound: -1.4769124, upper bound: 1.4906706
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 8, lower bound: -1.4771480, upper bound: 1.4968784
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 8, lower bound: -1.4968898, upper bound: 1.4771608
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.41
Output dim: 8, lower bound: -1.4968898, upper bound: 1.4969229

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.3938560, -5.7672443, -8.3989792, -5.7634406, -2.5326500, 2.5322330
1: -10.8517132, -7.9229774, -10.8608761, -7.8540254, -2.6702156, 2.6184173
2: -5.0586123, -2.3673687, -5.0535870, -2.3648252, -2.3807416, 2.3639774
3: -6.0782714, -2.8915911, -6.1086359, -2.8885198, -3.1897516, 3.2170448
4: -13.4628191, -9.8414116, -13.4631100, -9.8461723, -2.7139530, 2.7350397
5: -3.5546451, -1.5390961, -3.5579987, -1.5233393, -1.5052276, 1.5016098
6: -10.8660431, -8.1227188, -10.8757172, -8.0886469, -2.2713070, 2.2599030
7: -9.6359825, -6.2839179, -9.6382027, -6.2800989, -3.3558836, 3.3542848
8: 9.3519421, 11.9446383, 9.3064804, 11.9511471, -2.3589182, 2.3982625
9: -7.8572474, -4.4353366, -7.8629785, -4.4370794, -2.9378395, 2.9452200

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4766441, upper bound: 1.4839848
time: 6.77 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4769019, upper bound: 1.4906595
time: 8.15 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.4018402, -5.7667723, -8.4433861, -5.7173548, -2.5591044, 2.5699525
1: -10.8537636, -7.9222898, -10.8832188, -7.8353643, -2.6989775, 2.6457782
2: -5.0640135, -2.3662419, -5.0816073, -2.3194408, -2.4239860, 2.3922863
3: -6.0806170, -2.8903489, -6.1481233, -2.8801057, -3.2005112, 3.2577744
4: -13.4631748, -9.8374023, -13.4878197, -9.8180389, -2.7488220, 2.7651110
5: -3.5557177, -1.5313759, -3.6047831, -1.4858227, -1.5340033, 1.5360765
6: -10.8669634, -8.1119556, -10.9465780, -8.0206251, -2.3350801, 2.3076770
7: -9.6404991, -6.2830420, -9.6814880, -6.2427449, -3.3977542, 3.3984461
8: 9.3482847, 11.9452114, 9.2831059, 11.9701176, -2.3825159, 2.4193807
9: -7.8587103, -4.4339285, -7.9077005, -4.4252739, -2.9530220, 3.0053296

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4768798, upper bound: 1.4901923
time: 8.12 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771375, upper bound: 1.4968694
time: 5.32 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.4408264, -5.7609487, -8.4018459, -5.7667723, -2.5702448, 2.5431848
1: -10.8713799, -7.8377523, -10.8537655, -7.9222918, -2.6465368, 2.6964059
2: -5.0804291, -2.3590336, -5.0640192, -2.3662400, -2.3938828, 2.3940883
3: -6.1261244, -2.8826213, -6.0806208, -2.8903439, -3.2357805, 3.1979995
4: -13.4648161, -9.8270741, -13.4631758, -9.8373919, -2.7402017, 2.7375097
5: -3.5627644, -1.4808015, -3.5557179, -1.5313683, -1.5160527, 1.5424441
6: -10.8806114, -8.0272913, -10.8669634, -8.1119480, -2.2775192, 2.3255851
7: -9.6614304, -6.2756400, -9.6405087, -6.2830420, -3.3783884, 3.3648686
8: 9.2794495, 11.9540062, 9.3482761, 11.9452114, -2.4187269, 2.3657627
9: -7.8706479, -4.4301829, -7.8587122, -4.4339256, -2.9655361, 2.9553509

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966204, upper bound: 1.4704731
time: 7.06 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4968781, upper bound: 1.4771479
time: 5.73 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.4408264, -5.7609487, -8.4408264, -5.7609487, -2.5665846, 2.5665851
1: -10.8713799, -7.8377523, -10.8713799, -7.8377523, -2.6696148, 2.6696153
2: -5.0804291, -2.3590336, -5.0804291, -2.3590336, -2.4080081, 2.4080076
3: -6.1261244, -2.8826213, -6.1261244, -2.8826213, -3.2435031, 3.2435031
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7692533, 2.7692537
5: -3.5627644, -1.4808015, -3.5627644, -1.4808015, -1.5384035, 1.5384035
6: -10.8806114, -8.0272913, -10.8806114, -8.0272913, -2.2954426, 2.2954426
7: -9.6614304, -6.2756400, -9.6614304, -6.2756400, -3.3857903, 3.3857903
8: 9.2794495, 11.9540062, 9.2794495, 11.9540062, -2.3930426, 2.3930426
9: -7.8706479, -4.4301829, -7.8706479, -4.4301829, -2.9866467, 2.9866467

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966216, upper bound: 1.4705606
time: 9.13 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4968793, upper bound: 1.4772335
time: 6.52 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.31 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 8, lower bound: -1.4766441, upper bound: 1.4839848
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 8, lower bound: -1.4769019, upper bound: 1.4906595
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 8, lower bound: -1.4768798, upper bound: 1.4901923
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 8, lower bound: -1.4771375, upper bound: 1.4968694
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 8, lower bound: -1.4966204, upper bound: 1.4704731
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 8, lower bound: -1.4968781, upper bound: 1.4771479
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 8, lower bound: -1.4966216, upper bound: 1.4705606
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.31
Output dim: 8, lower bound: -1.4968793, upper bound: 1.4772335

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.3908768, -5.7749734, -8.3983908, -5.7649345, -2.5363145, 2.5217974
1: -10.8487730, -7.9290619, -10.8603001, -7.8551922, -2.6534328, 2.6109247
2: -5.0571165, -2.3749709, -5.0532918, -2.3663583, -2.3781595, 2.3562398
3: -6.0592718, -2.8956504, -6.1049662, -2.8893039, -3.1699679, 3.2093158
4: -13.4332590, -9.8474693, -13.4573994, -9.8474092, -2.6816897, 2.7234147
5: -3.5542159, -1.5492375, -3.5579159, -1.5253639, -1.4979506, 1.4833825
6: -10.8629694, -8.1416912, -10.8751154, -8.0923214, -2.2643113, 2.2386789
7: -9.5925064, -6.2872219, -9.6297836, -6.2807403, -3.3117661, 3.3425617
8: 9.3548727, 11.9359751, 9.3070574, 11.9494581, -2.3518085, 2.3864746
9: -7.8371086, -4.4370289, -7.8588567, -4.4374118, -2.9026356, 2.9325180

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4766441, upper bound: 1.4642542
time: 5.44 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4766441, upper bound: 1.4839849
time: 6.96 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.4040947, -5.7664018, -8.3989763, -5.7634420, -2.5419092, 2.5328782
1: -10.8576088, -7.9198341, -10.8608751, -7.8540297, -2.6728621, 2.6210537
2: -5.0706434, -2.3651733, -5.0535855, -2.3648300, -2.3929281, 2.3649135
3: -6.0842414, -2.8744361, -6.1086297, -2.8885214, -3.1957200, 3.2341936
4: -13.4639082, -9.8002853, -13.4630919, -9.8461752, -2.7098508, 2.7603564
5: -3.5672634, -1.5382533, -3.5579994, -1.5233417, -1.5033813, 1.5044737
6: -10.8789139, -8.1176338, -10.8757162, -8.0886574, -2.2726636, 2.2634499
7: -9.6451387, -6.2368846, -9.6381941, -6.2801042, -3.3650346, 3.4013095
8: 9.3367453, 11.9461508, 9.3064814, 11.9511433, -2.3719239, 2.3993714
9: -7.8628893, -4.4238338, -7.8629723, -4.4370794, -2.9448647, 2.9498911

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4769019, upper bound: 1.4709288
time: 6.82 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4769019, upper bound: 1.4906595
time: 8.21 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.3988686, -5.7744999, -8.4428253, -5.7188463, -2.5614676, 2.5595405
1: -10.8508263, -7.9283791, -10.8826494, -7.8365517, -2.6821685, 2.6382847
2: -5.0625191, -2.3738480, -5.0813155, -2.3209777, -2.4209771, 2.3845463
3: -6.0616083, -2.8944061, -6.1444111, -2.8808918, -3.1807165, 3.2500050
4: -13.4336157, -9.8434677, -13.4821072, -9.8192368, -2.7165542, 2.7522566
5: -3.5552902, -1.5415162, -3.6047025, -1.4878463, -1.5267242, 1.5182552
6: -10.8638897, -8.1308899, -10.9459763, -8.0242805, -2.3280492, 2.2864952
7: -9.5970592, -6.2863488, -9.6730976, -6.2433829, -3.3536763, 3.3867488
8: 9.3512154, 11.9365482, 9.2836800, 11.9684258, -2.3754101, 2.4075933
9: -7.8385549, -4.4356136, -7.9035234, -4.4256001, -2.9177732, 2.9912996

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4768799, upper bound: 1.4704621
time: 6.06 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4768798, upper bound: 1.4901920
time: 6.52 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.4120874, -5.7659287, -8.4433842, -5.7173567, -2.5617676, 2.5705962
1: -10.8596592, -7.9191561, -10.8832197, -7.8353672, -2.7016239, 2.6484032
2: -5.0760431, -2.3640432, -5.0816069, -2.3194456, -2.4269977, 2.3932290
3: -6.0865655, -2.8731930, -6.1481147, -2.8801081, -3.2064574, 3.2749217
4: -13.4642620, -9.7962837, -13.4878025, -9.8180418, -2.7447233, 2.7716014
5: -3.5683331, -1.5305332, -3.6047835, -1.4858248, -1.5321569, 1.5350747
6: -10.8798361, -8.1068697, -10.9465771, -8.0206299, -2.3364401, 2.3112655
7: -9.6496229, -6.2360110, -9.6814785, -6.2427440, -3.4068789, 3.4454675
8: 9.3330860, 11.9467239, 9.2831059, 11.9701118, -2.3955240, 2.4205718
9: -7.8643303, -4.4224253, -7.9076929, -4.4252748, -2.9600439, 2.9999113

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771376, upper bound: 1.4771390
time: 5.78 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771375, upper bound: 1.4968695
time: 4.84 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.4379053, -5.7686729, -8.4012566, -5.7682648, -2.5724664, 2.5326931
1: -10.8684692, -7.8437529, -10.8531895, -7.9234743, -2.6300297, 2.6887951
2: -5.0789452, -2.3666506, -5.0637236, -2.3677750, -2.3913050, 2.3863330
3: -6.1072655, -2.8866818, -6.0769167, -2.8911309, -3.2161345, 3.1902349
4: -13.4352531, -9.8332701, -13.4574671, -9.8385830, -2.7079339, 2.7259898
5: -3.5623391, -1.4909371, -3.5556359, -1.5333940, -1.5098317, 1.5246301
6: -10.8775501, -8.0460224, -10.8663607, -8.1156187, -2.2719941, 2.3044677
7: -9.6181622, -6.2789412, -9.6320839, -6.2836823, -3.3344798, 3.3531427
8: 9.2823687, 11.9453373, 9.3488541, 11.9435215, -2.4105005, 2.3539772
9: -7.8503776, -4.4318485, -7.8545904, -4.4342546, -2.9301605, 2.9426651

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4963733, upper bound: 1.4642543
time: 6.68 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966093, upper bound: 1.4704641
time: 5.81 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.4511385, -5.7600985, -8.4018469, -5.7667747, -2.5728803, 2.5438240
1: -10.8772612, -7.8346090, -10.8537655, -7.9222956, -2.6533275, 2.6989336
2: -5.0924554, -2.3568182, -5.0640173, -2.3662457, -2.4060640, 2.3950386
3: -6.1320677, -2.8654737, -6.0806112, -2.8903477, -3.2417200, 3.2151375
4: -13.4659023, -9.7859373, -13.4631586, -9.8373976, -2.7360973, 2.7648909
5: -3.5753779, -1.4799497, -3.5557170, -1.5313704, -1.5264163, 1.5414461
6: -10.8934755, -8.0221691, -10.8669624, -8.1119537, -2.2904997, 2.3291924
7: -9.6705627, -6.2286172, -9.6404963, -6.2830420, -3.3875208, 3.4118791
8: 9.2642794, 11.9555264, 9.3482771, 11.9452019, -2.4196491, 2.3682723
9: -7.8761702, -4.4186759, -7.8587027, -4.4339256, -2.9725370, 2.9600258

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966312, upper bound: 1.4709310
time: 5.88 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4968670, upper bound: 1.4771389
time: 5.45 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.4379053, -5.7686729, -8.4402485, -5.7624407, -2.5703020, 2.5561080
1: -10.8684692, -7.8437529, -10.8708096, -7.8389220, -2.6531224, 2.6622033
2: -5.0789452, -2.3666506, -5.0801353, -2.3605685, -2.4054279, 2.4002538
3: -6.1072655, -2.8866818, -6.1224499, -2.8834074, -3.2238581, 3.2357681
4: -13.4352531, -9.8332701, -13.4591055, -9.8282776, -2.7370005, 2.7577360
5: -3.5623391, -1.4909371, -3.5626836, -1.4828240, -1.5321829, 1.5201683
6: -10.8775501, -8.0460224, -10.8800125, -8.0309238, -2.2898769, 2.2741299
7: -9.6181622, -6.2789412, -9.6530437, -6.2762799, -3.3418822, 3.3741026
8: 9.2823687, 11.9453373, 9.2800255, 11.9523163, -2.3859277, 2.3812547
9: -7.8503776, -4.4318485, -7.8665113, -4.4305096, -2.9512997, 2.9739084

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4964076, upper bound: 1.4643399
time: 6.05 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966432, upper bound: 1.4705476
time: 6.79 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.4511385, -5.7600985, -8.4408255, -5.7609501, -2.5759749, 2.5672221
1: -10.8772612, -7.8346090, -10.8713779, -7.8377542, -2.6764030, 2.6722465
2: -5.0924554, -2.3568182, -5.0804291, -2.3590369, -2.4201899, 2.4089599
3: -6.1320677, -2.8654737, -6.1261158, -2.8826239, -3.2494438, 3.2606421
4: -13.4659023, -9.7859373, -13.4647980, -9.8270788, -2.7651494, 2.7958012
5: -3.5753779, -1.4799497, -3.5627651, -1.4808042, -1.5487669, 1.5412610
6: -10.8934755, -8.0221691, -10.8806095, -8.0272980, -2.3084226, 2.2989058
7: -9.6705627, -6.2286172, -9.6614256, -6.2756405, -3.3949223, 3.4328084
8: 9.2642794, 11.9555264, 9.2794495, 11.9539986, -2.4060068, 2.3955536
9: -7.8761702, -4.4186759, -7.8706384, -4.4301848, -2.9936461, 2.9912906

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966654, upper bound: 1.4710148
time: 5.00 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969014, upper bound: 1.4772221
time: 6.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.37 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4766441, upper bound: 1.4642542
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4766441, upper bound: 1.4839849
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4769019, upper bound: 1.4709288
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4769019, upper bound: 1.4906595
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4768799, upper bound: 1.4704621
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4768798, upper bound: 1.4901920
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4771376, upper bound: 1.4771390
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4771375, upper bound: 1.4968695
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4963733, upper bound: 1.4642543
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4966093, upper bound: 1.4704641
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4966312, upper bound: 1.4709310
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4968670, upper bound: 1.4771389
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4964076, upper bound: 1.4643399
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4966432, upper bound: 1.4705476
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4966654, upper bound: 1.4710148
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.37
Output dim: 8, lower bound: -1.4969014, upper bound: 1.4772221

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.3908768, -5.7749734, -8.3658371, -5.7704005, -2.5290179, 2.4895215
1: -10.8487730, -7.9290619, -10.8439913, -7.9266872, -2.5856223, 2.5930691
2: -5.0571165, -2.3749709, -5.0398207, -2.3728189, -2.3652263, 2.3449483
3: -6.0592718, -2.8956504, -6.0663199, -2.8965125, -3.1627593, 3.1706696
4: -13.4332590, -9.8474693, -13.4558649, -9.8561974, -2.6732373, 2.7103639
5: -3.5542159, -1.5492375, -3.5511761, -1.5677879, -1.4616218, 1.4725163
6: -10.8629694, -8.1416912, -10.8622437, -8.1633902, -2.2043414, 2.2250264
7: -9.5925064, -6.2872219, -9.6121264, -6.2874775, -3.3050289, 3.3249044
8: 9.3548727, 11.9359751, 9.3651342, 11.9409513, -2.3423920, 2.3279662
9: -7.8371086, -4.4370289, -7.8479800, -4.4405437, -2.8984756, 2.9184828

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4706729, upper bound: 1.4642540
time: 5.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4706729, upper bound: 1.4642541
time: 6.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.3908768, -5.7749734, -8.4043350, -5.7645497, -2.5368347, 2.5235486
1: -10.8487730, -7.9290619, -10.8617649, -7.8420653, -2.6542969, 2.6127687
2: -5.0571165, -2.3749709, -5.0557780, -2.3654909, -2.3791857, 2.3583517
3: -6.0592718, -2.8956504, -6.1118774, -2.8887973, -3.1704745, 3.2162271
4: -13.4332590, -9.8474693, -13.4575367, -9.8460960, -2.6825323, 2.7221367
5: -3.5542159, -1.5492375, -3.5582836, -1.5175415, -1.4984083, 1.4840226
6: -10.8629694, -8.1416912, -10.8760004, -8.0794754, -2.2655232, 2.2397394
7: -9.5925064, -6.2872219, -9.6326981, -6.2798948, -3.3126116, 3.3454762
8: 9.3548727, 11.9359751, 9.2963352, 11.9498644, -2.3523464, 2.3902040
9: -7.8371086, -4.4370289, -7.8602858, -4.4366760, -2.9024153, 2.9329810

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4706729, upper bound: 1.4839849
time: 6.13 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4706729, upper bound: 1.4839847
time: 6.25 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4040947, -5.7664018, -8.3664312, -5.7689085, -2.5346336, 2.5006557
1: -10.8576088, -7.9198341, -10.8445702, -7.9255090, -2.6089115, 2.6031647
2: -5.0706434, -2.3651733, -5.0401182, -2.3712916, -2.3799934, 2.3536243
3: -6.0842414, -2.8744361, -6.0700078, -2.8957283, -3.1885130, 3.1955717
4: -13.4639082, -9.8002853, -13.4615536, -9.8550186, -2.7013869, 2.7472024
5: -3.5672634, -1.5382533, -3.5512586, -1.5657660, -1.4782133, 1.4935958
6: -10.8789139, -8.1176338, -10.8628492, -8.1596718, -2.2228565, 2.2498012
7: -9.6451387, -6.2368846, -9.6205597, -6.2868395, -3.3582993, 3.3836751
8: 9.3367453, 11.9461508, 9.3645554, 11.9426355, -2.3625069, 2.3422699
9: -7.8628893, -4.4238338, -7.8520823, -4.4402089, -2.9407339, 2.9358253

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709306, upper bound: 1.4709310
time: 6.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709306, upper bound: 1.4709289
time: 7.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4040947, -5.7664018, -8.4049187, -5.7630596, -2.5424247, 2.5346315
1: -10.8576088, -7.9198341, -10.8623371, -7.8409052, -2.6737280, 2.6228952
2: -5.0706434, -2.3651733, -5.0560708, -2.3639617, -2.3939557, 2.3670263
3: -6.0842414, -2.8744361, -6.1155343, -2.8880146, -3.1962268, 3.2410982
4: -13.4639082, -9.8002853, -13.4632311, -9.8448601, -2.7106984, 2.7586534
5: -3.5672634, -1.5382533, -3.5583668, -1.5155205, -1.5038390, 1.5051157
6: -10.8789139, -8.1176338, -10.8766012, -8.0758171, -2.2738791, 2.2645094
7: -9.6451387, -6.2368846, -9.6411037, -6.2792578, -3.3658810, 3.4042192
8: 9.3367453, 11.9461508, 9.2957573, 11.9515514, -2.3724618, 2.4011385
9: -7.8628893, -4.4238338, -7.8644094, -4.4363451, -2.9446383, 2.9503608

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709306, upper bound: 1.4906597
time: 7.43 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709306, upper bound: 1.4906598
time: 7.41 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.3988686, -5.7744999, -8.4099112, -5.7242727, -2.5538855, 2.5267501
1: -10.8508263, -7.9283791, -10.8664961, -7.9073639, -2.6407881, 2.6201754
2: -5.0625191, -2.3738480, -5.0675697, -2.3273129, -2.4078360, 2.3736353
3: -6.0616083, -2.8944061, -6.1064844, -2.8881392, -3.1734691, 3.2120783
4: -13.4336157, -9.8434677, -13.4806156, -9.8286037, -2.7083559, 2.7391355
5: -3.5552902, -1.5415162, -3.5980549, -1.5303876, -1.4918816, 1.5071406
6: -10.8638897, -8.1308899, -10.9332790, -8.0956230, -2.2779741, 2.2727518
7: -9.5970592, -6.2863488, -9.6556482, -6.2498765, -3.3471828, 3.3692994
8: 9.3512154, 11.9365482, 9.3417501, 11.9600506, -2.3661900, 2.3490562
9: -7.8385549, -4.4356136, -7.8928423, -4.4286404, -2.9136705, 2.9770334

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4706716, upper bound: 1.4702271
time: 5.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4706716, upper bound: 1.4649435
time: 5.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.3988686, -5.7744999, -8.4488220, -5.7184691, -2.5587282, 2.5614018
1: -10.8508263, -7.9283791, -10.8840828, -7.8235493, -2.6830578, 2.6400871
2: -5.0625191, -2.3738480, -5.0838218, -2.3201323, -2.4207139, 2.3865809
3: -6.0616083, -2.8944061, -6.1511812, -2.8803747, -3.1812336, 3.2567751
4: -13.4336157, -9.8434677, -13.4822359, -9.8178396, -2.7173934, 2.7505615
5: -3.5552902, -1.5415162, -3.6050570, -1.4800515, -1.5272036, 1.5144039
6: -10.8638897, -8.1308899, -10.9468403, -8.0113964, -2.3293505, 2.2796595
7: -9.5970592, -6.2863488, -9.6760015, -6.2425652, -3.3544941, 3.3896527
8: 9.3512154, 11.9365482, 9.2729549, 11.9688168, -2.3759184, 2.4114246
9: -7.8385549, -4.4356136, -7.9049177, -4.4248857, -2.9175396, 2.9905186

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4706716, upper bound: 1.4899568
time: 5.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4706716, upper bound: 1.4846764
time: 5.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4120874, -5.7659287, -8.4104805, -5.7227788, -2.5542059, 2.5378580
1: -10.8596592, -7.9191561, -10.8670692, -7.9061675, -2.6641536, 2.6302958
2: -5.0760431, -2.3640432, -5.0678630, -2.3257833, -2.4138527, 2.3823195
3: -6.0865655, -2.8731930, -6.1102152, -2.8873549, -3.1992106, 3.2370222
4: -13.4642620, -9.7962837, -13.4863081, -9.8274212, -2.7365127, 2.7584825
5: -3.5683331, -1.5305332, -3.5981364, -1.5283656, -1.5084739, 1.5239551
6: -10.8798361, -8.1068697, -10.9338789, -8.0919456, -2.2965565, 2.2975254
7: -9.6496229, -6.2360110, -9.6640463, -6.2492361, -3.4003868, 3.4280353
8: 9.3330860, 11.9467239, 9.3411751, 11.9617329, -2.3863058, 2.3633575
9: -7.8643303, -4.4224253, -7.8969989, -4.4283123, -2.9559689, 2.9856153

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709294, upper bound: 1.4769012
time: 6.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709294, upper bound: 1.4716176
time: 7.17 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4120874, -5.7659287, -8.4493828, -5.7169790, -2.5590239, 2.5724599
1: -10.8596592, -7.9191561, -10.8846521, -7.8223681, -2.7025127, 2.6502042
2: -5.0760431, -2.3640432, -5.0841141, -2.3185992, -2.4267330, 2.3952620
3: -6.0865655, -2.8731930, -6.1548843, -2.8795898, -3.2069757, 3.2816913
4: -13.4642620, -9.7962837, -13.4879303, -9.8166389, -2.7455668, 2.7699060
5: -3.5683331, -1.5305332, -3.6051393, -1.4780300, -1.5326358, 1.5312245
6: -10.8798361, -8.1068697, -10.9474401, -8.0077534, -2.3377430, 2.3044291
7: -9.6496229, -6.2360110, -9.6843777, -6.2419281, -3.4076948, 3.4483666
8: 9.3330860, 11.9467239, 9.2723827, 11.9705009, -2.3953004, 2.4223595
9: -7.8643303, -4.4224253, -7.9090891, -4.4245586, -2.9598041, 2.9991307

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709293, upper bound: 1.4966318
time: 7.16 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709293, upper bound: 1.4913490
time: 7.72 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.4298067, -5.7691431, -8.3658371, -5.7704005, -2.5589790, 2.4972644
1: -10.8664494, -7.8444295, -10.8439913, -7.9266872, -2.6057534, 2.6606383
2: -5.0734763, -2.3677502, -5.0398207, -2.3728189, -2.3787990, 2.3588581
3: -6.1049271, -2.8879311, -6.0663199, -2.8965125, -3.2084146, 3.1783888
4: -13.4349041, -9.8373661, -13.4558649, -9.8561974, -2.6851168, 2.7194910
5: -3.5612791, -1.4987278, -3.5511761, -1.5677879, -1.4731772, 1.5069067
6: -10.8766518, -8.0569811, -10.8622437, -8.1633902, -2.2189608, 2.2821751
7: -9.6135473, -6.2797852, -9.6121264, -6.2874775, -3.3260698, 3.3323412
8: 9.2860394, 11.9447880, 9.3651342, 11.9409513, -2.4030433, 2.3377423
9: -7.8490114, -4.4332399, -7.8479800, -4.4405437, -2.9124203, 2.9223318

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904010, upper bound: 1.4642537
time: 6.59 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904010, upper bound: 1.4642543
time: 10.42 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.4378977, -5.7686763, -8.4099112, -5.7242727, -2.5748758, 2.5344963
1: -10.8684683, -7.8437529, -10.8664961, -7.9073639, -2.6609774, 2.6833868
2: -5.0789385, -2.3666520, -5.0675697, -2.3273129, -2.4152884, 2.3873720
3: -6.1072612, -2.8866832, -6.1064844, -2.8881392, -3.2191219, 3.2198012
4: -13.4352522, -9.8332777, -13.4806156, -9.8286037, -2.7200246, 2.7483370
5: -3.5623386, -1.4909507, -3.5980549, -1.5303876, -1.5032997, 1.5251621
6: -10.8775530, -8.0460415, -10.9332790, -8.0956230, -2.2925076, 2.3050134
7: -9.6181545, -6.2789431, -9.6556482, -6.2498765, -3.3682780, 3.3767052
8: 9.2823753, 11.9453373, 9.3417501, 11.9600506, -2.4120617, 2.3587861
9: -7.8503752, -4.4318519, -7.8928423, -4.4286404, -2.9275017, 2.9820683

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904010, upper bound: 1.4702267
time: 7.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904009, upper bound: 1.4702262
time: 7.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4430370, -5.7605667, -8.3664312, -5.7689085, -2.5593443, 2.5084066
1: -10.8752441, -7.8352842, -10.8445702, -7.9255090, -2.6290550, 2.6707654
2: -5.0869856, -2.3579240, -5.0401182, -2.3712916, -2.3935547, 2.3675566
3: -6.1297417, -2.8667223, -6.0700078, -2.8957283, -3.2340133, 3.2032855
4: -13.4655523, -9.7900352, -13.4615536, -9.8550186, -2.7132668, 2.7565067
5: -3.5743196, -1.4877414, -3.5512586, -1.5657660, -1.4897616, 1.5237107
6: -10.8925781, -8.0330915, -10.8628492, -8.1596718, -2.2374706, 2.3069043
7: -9.6659861, -6.2294583, -9.6205597, -6.2868395, -3.3791466, 3.3911014
8: 9.2679482, 11.9549770, 9.3645554, 11.9426355, -2.4121933, 2.3520417
9: -7.8748121, -4.4200573, -7.8520823, -4.4402089, -2.9548264, 2.9396734

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906587, upper bound: 1.4709309
time: 5.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906587, upper bound: 1.4709310
time: 4.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4511280, -5.7600980, -8.4104805, -5.7227788, -2.5752707, 2.5456111
1: -10.8772583, -7.8346100, -10.8670692, -7.9061675, -2.6822939, 2.6935244
2: -5.0924482, -2.3568206, -5.0678630, -2.3257833, -2.4213142, 2.3960791
3: -6.1320634, -2.8654745, -6.1102152, -2.8873549, -3.2447085, 3.2447407
4: -13.4658995, -9.7859478, -13.4863081, -9.8274212, -2.7481833, 2.7677097
5: -3.5753777, -1.4799656, -3.5981364, -1.5283656, -1.5198841, 1.5419866
6: -10.8934765, -8.0221825, -10.9338789, -8.0919456, -2.3110876, 2.3297386
7: -9.6705532, -6.2286177, -9.6640463, -6.2492361, -3.4213171, 3.4354286
8: 9.2642851, 11.9555254, 9.3411751, 11.9617329, -2.4212112, 2.3730841
9: -7.8761683, -4.4186773, -7.8969989, -4.4283123, -2.9699507, 2.9906521

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906587, upper bound: 1.4769035
time: 6.18 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906587, upper bound: 1.4771392
time: 5.27 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.4298067, -5.7691431, -8.4043350, -5.7645497, -2.5601039, 2.5199428
1: -10.8664494, -7.8444295, -10.8617649, -7.8420653, -2.6288486, 2.6358919
2: -5.0734763, -2.3677502, -5.0557780, -2.3654909, -2.3930655, 2.3725691
3: -6.1049271, -2.8879311, -6.1118774, -2.8887973, -3.2161298, 3.2239463
4: -13.4349041, -9.8373661, -13.4575367, -9.8460960, -2.7141554, 2.7513008
5: -3.5612791, -1.4987278, -3.5582836, -1.5175415, -1.4954422, 1.5063705
6: -10.8766518, -8.0569811, -10.8760004, -8.0794754, -2.2374320, 2.2576785
7: -9.6135473, -6.2797852, -9.6326981, -6.2798948, -3.3336525, 3.3529129
8: 9.2860394, 11.9447880, 9.2963352, 11.9498644, -2.3795981, 2.3649135
9: -7.8490114, -4.4332399, -7.8602858, -4.4366760, -2.9335423, 2.9540806

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904353, upper bound: 1.4643401
time: 6.00 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904349, upper bound: 1.4643394
time: 7.48 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.4378977, -5.7686763, -8.4488220, -5.7184691, -2.5844965, 2.5576787
1: -10.8684683, -7.8437529, -10.8840828, -7.8235493, -2.6832995, 2.6632376
2: -5.0789385, -2.3666520, -5.0838218, -2.3201323, -2.4358993, 2.4006219
3: -6.1072612, -2.8866832, -6.1511812, -2.8803747, -3.2268865, 3.2644980
4: -13.4352522, -9.8332777, -13.4822359, -9.8178396, -2.7483182, 2.7794027
5: -3.5623386, -1.4909507, -3.6050570, -1.4800515, -1.5253286, 1.5400800
6: -10.8775530, -8.0460415, -10.9468403, -8.0113964, -2.3107281, 2.3151147
7: -9.6181545, -6.2789431, -9.6760015, -6.2425652, -3.3755894, 3.3970585
8: 9.2823753, 11.9453373, 9.2729549, 11.9688168, -2.4031963, 2.3859940
9: -7.8503752, -4.4318519, -7.9049177, -4.4248857, -2.9485550, 3.0104201

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904354, upper bound: 1.4703141
time: 6.97 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904349, upper bound: 1.4705478
time: 7.08 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4430370, -5.7605667, -8.4049187, -5.7630596, -2.5657377, 2.5310678
1: -10.8752441, -7.8352842, -10.8623371, -7.8409052, -2.6521358, 2.6459417
2: -5.0869856, -2.3579240, -5.0560708, -2.3639617, -2.4078236, 2.3812656
3: -6.1297417, -2.8667223, -6.1155343, -2.8880146, -3.2417271, 3.2488120
4: -13.4655523, -9.7900352, -13.4632311, -9.8448601, -2.7422898, 2.7874222
5: -3.5743196, -1.4877414, -3.5583668, -1.5155205, -1.5120263, 1.5274398
6: -10.8925781, -8.0330915, -10.8766012, -8.0758171, -2.2559853, 2.2824774
7: -9.6659861, -6.2294583, -9.6411037, -6.2792578, -3.3867283, 3.4116454
8: 9.2679482, 11.9549770, 9.2957573, 11.9515514, -2.3996758, 2.3792138
9: -7.8748121, -4.4200573, -7.8644094, -4.4363451, -2.9759164, 2.9714451

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906930, upper bound: 1.4710165
time: 5.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906930, upper bound: 1.4710165
time: 5.29 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.34 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4706729, upper bound: 1.4642540
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4706729, upper bound: 1.4642541
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4706729, upper bound: 1.4839849
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4706729, upper bound: 1.4839847
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4709306, upper bound: 1.4709310
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4709306, upper bound: 1.4709289
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4709306, upper bound: 1.4906597
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4709306, upper bound: 1.4906598
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4706716, upper bound: 1.4702271
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4706716, upper bound: 1.4649435
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4706716, upper bound: 1.4899568
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4706716, upper bound: 1.4846764
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4709294, upper bound: 1.4769012
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4709294, upper bound: 1.4716176
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4709293, upper bound: 1.4966318
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4709293, upper bound: 1.4913490
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4904010, upper bound: 1.4642537
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4904010, upper bound: 1.4642543
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4904010, upper bound: 1.4702267
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4904009, upper bound: 1.4702262
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4906587, upper bound: 1.4709309
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4906587, upper bound: 1.4709310
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4906587, upper bound: 1.4769035
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4906587, upper bound: 1.4771392
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4904353, upper bound: 1.4643401
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4904349, upper bound: 1.4643394
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4904354, upper bound: 1.4703141
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4904349, upper bound: 1.4705478
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4906930, upper bound: 1.4710165
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.34
Output dim: 8, lower bound: -1.4906930, upper bound: 1.4710165
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.34
Output dim: 8, lower bound: -1.4969014, upper bound: 1.4772221
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.435220718383789
rel_dist={8: [-1.4969366874007388, 1.496938647628781]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5736
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3678170, upper bound: 1.3847379
time: 7.24 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847574, upper bound: 1.3847575
time: 7.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.84
Output dim: 8, lower bound: -1.3678170, upper bound: 1.3847379
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.84
Output dim: 8, lower bound: -1.3847574, upper bound: 1.3847575

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.4018459, -5.7667723, -8.4336243, -5.7614031, -2.4569197, 2.4803391
1: -10.8537655, -7.9222918, -10.8696527, -7.8534660, -2.6132751, 2.5677667
2: -5.0640192, -2.3662400, -5.0774169, -2.3600531, -2.3101377, 2.3083444
3: -6.0806208, -2.8903439, -6.1178594, -2.8832345, -3.1552048, 3.1763668
4: -13.4631758, -9.8373919, -13.4646597, -9.8288527, -2.6144834, 2.6196928
5: -3.5557179, -1.5313683, -3.5623341, -1.4902031, -1.4749213, 1.4521446
6: -10.8669634, -8.1119480, -10.8795643, -8.0428057, -2.2252612, 2.1834486
7: -9.6405087, -6.2830420, -9.6578960, -6.2766342, -3.3638744, 3.3748541
8: 9.3482761, 11.9452114, 9.2923164, 11.9535341, -2.2894645, 2.3367796
9: -7.8587122, -4.4339256, -7.8689923, -4.4310431, -2.8763285, 2.8857059

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3675326, upper bound: 1.3793384
time: 7.12 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3678079, upper bound: 1.3847259
time: 7.05 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.4408264, -5.7609487, -8.4408283, -5.7609482, -2.4799757, 2.4971442
1: -10.8713799, -7.8377523, -10.8713818, -7.8377485, -2.6558404, 2.5894279
2: -5.0804291, -2.3590336, -5.0804310, -2.3590326, -2.3241072, 2.3252444
3: -6.1261244, -2.8826213, -6.1261268, -2.8826227, -3.2022972, 3.2104063
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6468058, 2.6290185
5: -3.5627644, -1.4808015, -3.5627654, -1.4808009, -1.5026531, 1.4736557
6: -10.8806114, -8.0272913, -10.8806124, -8.0272884, -2.2646508, 2.1989474
7: -9.6614304, -6.2756400, -9.6614361, -6.2756395, -3.3857908, 3.3857961
8: 9.2794495, 11.9540062, 9.2794437, 11.9540062, -2.3148723, 2.3595300
9: -7.8706479, -4.4301829, -7.8706493, -4.4301844, -2.9069452, 2.8915186

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847359, upper bound: 1.3678166
time: 6.16 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847359, upper bound: 1.3847581
time: 7.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.79 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 25.79
Output dim: 8, lower bound: -1.3675326, upper bound: 1.3793384
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.79
Output dim: 8, lower bound: -1.3678079, upper bound: 1.3847259
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.79
Output dim: 8, lower bound: -1.3847359, upper bound: 1.3678166
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.79
Output dim: 8, lower bound: -1.3847359, upper bound: 1.3847581

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.4018383, -5.7667727, -8.4421921, -5.7174292, -2.4726706, 2.4797912
1: -10.8537636, -7.9222908, -10.8829327, -7.8379531, -2.6151352, 2.5688510
2: -5.0640125, -2.3662415, -5.0811095, -2.3196111, -2.3392906, 2.3066182
3: -6.0806160, -2.8903465, -6.1467772, -2.8802114, -3.1522217, 3.2109079
4: -13.4631748, -9.8374023, -13.4877949, -9.8183193, -2.6251726, 2.6432548
5: -3.5557163, -1.5313779, -3.6047127, -1.4873757, -1.4637868, 1.4712895
6: -10.8669634, -8.1119576, -10.9464045, -8.0231915, -2.2315636, 2.2117565
7: -9.6404991, -6.2830410, -9.6809111, -6.2429056, -3.3975935, 3.3978701
8: 9.3482828, 11.9452105, 9.2852449, 11.9700413, -2.3067274, 2.3407903
9: -7.8587070, -4.4339285, -7.9074221, -4.4254165, -2.8737726, 2.9242108

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3674422, upper bound: 1.3788721
time: 5.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3677986, upper bound: 1.3847185
time: 4.94 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.4408283, -5.7609487, -8.4018459, -5.7667723, -2.4824715, 2.4575300
1: -10.8713808, -7.8377638, -10.8537655, -7.9222918, -2.5699711, 2.6141996
2: -5.0804281, -2.3590326, -5.0640192, -2.3662400, -2.3108807, 2.3113666
3: -6.1261239, -2.8826218, -6.0806208, -2.8903439, -3.1848269, 3.1553330
4: -13.4648142, -9.8270731, -13.4631758, -9.8373919, -2.6184337, 2.6156316
5: -3.5627644, -1.4808030, -3.5557179, -1.5313683, -1.4529171, 1.4754444
6: -10.8806095, -8.0272923, -10.8669634, -8.1119480, -2.1846962, 2.2266297
7: -9.6614332, -6.2756395, -9.6405087, -6.2830420, -3.3783913, 3.3648691
8: 9.2794514, 11.9540062, 9.3482761, 11.9452114, -2.3395267, 2.2900753
9: -7.8706484, -4.4301844, -7.8587122, -4.4339256, -2.8864355, 2.8762541

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3843688, upper bound: 1.3619618
time: 28.44 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847254, upper bound: 1.3678071
time: 7.08 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.4408264, -5.7609487, -8.4408264, -5.7609487, -2.4799757, 2.4799762
1: -10.8713799, -7.8377523, -10.8713799, -7.8377523, -2.5894279, 2.5894275
2: -5.0804291, -2.3590336, -5.0804291, -2.3590336, -2.3252444, 2.3252439
3: -6.1261244, -2.8826213, -6.1261244, -2.8826213, -3.2104044, 3.2104053
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6468043, 2.6468046
5: -3.5627644, -1.4808015, -3.5627644, -1.4808015, -1.4736557, 1.4736557
6: -10.8806114, -8.0272913, -10.8806114, -8.0272913, -2.1989474, 2.1989474
7: -9.6614304, -6.2756400, -9.6614304, -6.2756400, -3.3857903, 3.3857903
8: 9.2794495, 11.9540062, 9.2794495, 11.9540062, -2.3148727, 2.3148727
9: -7.8706479, -4.4301829, -7.8706479, -4.4301829, -2.9069443, 2.9069438

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3843699, upper bound: 1.3620314
time: 6.29 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847265, upper bound: 1.3678779
time: 4.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.95 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 23.95
Output dim: 8, lower bound: -1.3674422, upper bound: 1.3788721
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.95
Output dim: 8, lower bound: -1.3677986, upper bound: 1.3847185
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 23.95
Output dim: 8, lower bound: -1.3843688, upper bound: 1.3619618
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 23.95
Output dim: 8, lower bound: -1.3847254, upper bound: 1.3678071
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 23.95
Output dim: 8, lower bound: -1.3843699, upper bound: 1.3620314
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.95
Output dim: 8, lower bound: -1.3847265, upper bound: 1.3678779

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.4120846, -5.7659287, -8.4421921, -5.7174349, -2.4753351, 2.4802630
1: -10.8596563, -7.9191561, -10.8829317, -7.8379583, -2.6176910, 2.5714750
2: -5.0759869, -2.3640423, -5.0811081, -2.3196163, -2.3422661, 2.3071651
3: -6.0865641, -2.8732040, -6.1467657, -2.8802111, -3.1544971, 3.2202168
4: -13.4642611, -9.7962856, -13.4877701, -9.8183250, -2.6191349, 2.6480539
5: -3.5683336, -1.5305338, -3.6047125, -1.4873798, -1.4619409, 1.4691540
6: -10.8798351, -8.1068726, -10.9464045, -8.0232038, -2.2329230, 2.2150817
7: -9.6496201, -6.2360120, -9.6808987, -6.2429085, -3.4067116, 3.4448867
8: 9.3330879, 11.9467144, 9.2852459, 11.9700317, -2.3197327, 2.3392453
9: -7.8643303, -4.4224238, -7.9074111, -4.4254184, -2.8794122, 2.9187922

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3677985, upper bound: 1.3678000
time: 4.93 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3677985, upper bound: 1.3847186
time: 4.92 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.4511385, -5.7600975, -8.4018450, -5.7667737, -2.4851036, 2.4579968
1: -10.8772564, -7.8346210, -10.8537636, -7.9222956, -2.5766678, 2.6167243
2: -5.0923986, -2.3568187, -5.0640187, -2.3662462, -2.3230019, 2.3119211
3: -6.1320677, -2.8654864, -6.0806108, -2.8903461, -3.1871099, 3.1726322
4: -13.4658985, -9.7859383, -13.4631538, -9.8373966, -2.6123891, 2.6412637
5: -3.5753777, -1.4799505, -3.5557165, -1.5313709, -1.4632792, 1.4733120
6: -10.8934708, -8.0221691, -10.8669624, -8.1119556, -2.1976757, 2.2299676
7: -9.6705580, -6.2286167, -9.6404953, -6.2830420, -3.3875160, 3.4118786
8: 9.2642794, 11.9555168, 9.3482761, 11.9452009, -2.3404498, 2.2915363
9: -7.8761711, -4.4186754, -7.8587008, -4.4339266, -2.8920507, 2.8809261

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3844406, upper bound: 1.3624100
time: 7.41 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847160, upper bound: 1.3677982
time: 4.37 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.4511395, -5.7600970, -8.4408255, -5.7609501, -2.4893651, 2.4804423
1: -10.8772564, -7.8346090, -10.8713779, -7.8377562, -2.5961237, 2.5920587
2: -5.0923996, -2.3568187, -5.0804281, -2.3590384, -2.3373661, 2.3258004
3: -6.1320672, -2.8654873, -6.1261153, -2.8826244, -3.2126875, 3.2277026
4: -13.4659004, -9.7859373, -13.4647923, -9.8270779, -2.6407590, 2.6715989
5: -3.5753770, -1.4799504, -3.5627646, -1.4808050, -1.4840183, 1.4753642
6: -10.8934765, -8.0221691, -10.8806095, -8.0273018, -2.2119269, 2.2021494
7: -9.6705589, -6.2286153, -9.6614227, -6.2756405, -3.3949184, 3.4328074
8: 9.2642765, 11.9555159, 9.2794495, 11.9539995, -2.3278351, 2.3163328
9: -7.8761711, -4.4186749, -7.8706369, -4.4301844, -2.9125576, 2.9115863

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3844643, upper bound: 1.3624891
time: 7.03 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847387, upper bound: 1.3678680
time: 8.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 28.45 seconds
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 28.45
Output dim: 8, lower bound: -1.3677985, upper bound: 1.3678000
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 8, lower bound: -1.3677985, upper bound: 1.3847186
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 8, lower bound: -1.3844406, upper bound: 1.3624100
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 8, lower bound: -1.3847160, upper bound: 1.3677982
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 8, lower bound: -1.3844643, upper bound: 1.3624891
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.45
Output dim: 8, lower bound: -1.3847387, upper bound: 1.3678680

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4120846, -5.7659287, -8.4493828, -5.7169790, -2.4726934, 2.4824927
1: -10.8596563, -7.9191561, -10.8846512, -7.8223886, -2.6187396, 2.5736370
2: -5.0759869, -2.3640423, -5.0841126, -2.3186002, -2.3422208, 2.3096004
3: -6.0865641, -2.8732040, -6.1548824, -2.8795919, -3.1546421, 3.2228065
4: -13.4642611, -9.7962856, -13.4879284, -9.8166399, -2.6202071, 2.6463816
5: -3.5683336, -1.5305338, -3.6051385, -1.4780298, -1.4625124, 1.4654270
6: -10.8798351, -8.1068726, -10.9474354, -8.0077534, -2.2344766, 2.2084587
7: -9.6496201, -6.2360120, -9.6843767, -6.2419291, -3.4076910, 3.4483647
8: 9.3330879, 11.9467144, 9.2723846, 11.9704981, -2.3187828, 2.3413858
9: -7.8643303, -4.4224238, -7.9090877, -4.4245586, -2.8793225, 2.9183726

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3624108, upper bound: 1.3844415
time: 6.20 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3624108, upper bound: 1.3793311
time: 5.23 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.4402637, -5.7607250, -8.3664303, -5.7689080, -2.4683299, 2.4224200
1: -10.8745489, -7.8355579, -10.8445683, -7.9255090, -2.5507135, 2.5868070
2: -5.0850496, -2.3583040, -5.0401173, -2.3712921, -2.3084750, 2.2839389
3: -6.1289430, -2.8671620, -6.0700064, -2.8957293, -3.1704969, 3.1527581
4: -13.4654284, -9.7914362, -13.4615536, -9.8550186, -2.5894501, 2.6308763
5: -3.5739503, -1.4904033, -3.5512598, -1.5657670, -1.4261284, 1.4522984
6: -10.8922653, -8.0368252, -10.8628473, -8.1596718, -2.1442699, 2.2029033
7: -9.6644192, -6.2297535, -9.6205587, -6.2868409, -3.3775783, 3.3908052
8: 9.2692013, 11.9547777, 9.3645563, 11.9426346, -2.3316245, 2.2750883
9: -7.8743448, -4.4205346, -7.8520784, -4.4402094, -2.8730679, 2.8595009

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3793282, upper bound: 1.3624105
time: 5.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3793282, upper bound: 1.3624104
time: 8.18 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4511280, -5.7600985, -8.4104795, -5.7227807, -2.4874945, 2.4577341
1: -10.8772545, -7.8346248, -10.8670692, -7.9061680, -2.6020403, 2.6113169
2: -5.0923891, -2.3568196, -5.0678635, -2.3257852, -2.3361154, 2.3107033
3: -6.1320620, -2.8654914, -6.1102104, -2.8873551, -3.1839638, 3.2080550
4: -13.4658976, -9.7859516, -13.4863052, -9.8274212, -2.6229451, 2.6440828
5: -3.5753763, -1.4799670, -3.5981367, -1.5283660, -1.4536214, 1.4738525
6: -10.8934708, -8.0221901, -10.9338799, -8.0919456, -2.2118640, 2.2305140
7: -9.6705475, -6.2286177, -9.6640472, -6.2492371, -3.4213104, 3.4354296
8: 9.2642879, 11.9555178, 9.3411751, 11.9617290, -2.3420105, 2.2956104
9: -7.8761663, -4.4186792, -7.8969975, -4.4283137, -2.8894649, 2.9098983

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3793282, upper bound: 1.3675250
time: 5.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3793282, upper bound: 1.3677983
time: 6.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4402819, -5.7607265, -8.4049187, -5.7630587, -2.4763632, 2.4441276
1: -10.8745518, -7.8355160, -10.8623352, -7.8409052, -2.5701685, 2.5642815
2: -5.0850658, -2.3583016, -5.0560708, -2.3639631, -2.3229971, 2.2976089
3: -6.1289454, -2.8671603, -6.1155343, -2.8880160, -3.1961632, 3.2075815
4: -13.4654303, -9.7914257, -13.4632263, -9.8448601, -2.6177921, 2.6612117
5: -3.5739496, -1.4903983, -3.5583668, -1.5155213, -1.4467821, 1.4588470
6: -10.8922710, -8.0368099, -10.8766003, -8.0758171, -2.1591120, 2.1817997
7: -9.6644239, -6.2297525, -9.6411018, -6.2792578, -3.3851662, 3.4113493
8: 9.2691965, 11.9547787, 9.2957573, 11.9515514, -2.3203001, 2.2997813
9: -7.8743453, -4.4205341, -7.8644047, -4.4363461, -2.8935595, 2.8906584

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3793558, upper bound: 1.3624892
time: 6.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3793558, upper bound: 1.3624893
time: 6.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4511280, -5.7600975, -8.4493828, -5.7169781, -2.4975505, 2.4799490
1: -10.8772535, -7.8346114, -10.8846540, -7.8223686, -2.6248155, 2.5930872
2: -5.0923901, -2.3568201, -5.0841126, -2.3186011, -2.3573670, 2.3239112
3: -6.1320629, -2.8654916, -6.1548820, -2.8795907, -3.2097225, 3.2619476
4: -13.4658985, -9.7859497, -13.4879274, -9.8166409, -2.6505418, 2.6744757
5: -3.5753763, -1.4799681, -3.6051385, -1.4780298, -1.4740405, 1.4894958
6: -10.8934755, -8.0221901, -10.9474373, -8.0077553, -2.2284913, 2.2402267
7: -9.6705494, -6.2286186, -9.6843777, -6.2419291, -3.4286203, 3.4557590
8: 9.2642889, 11.9555168, 9.2723846, 11.9704971, -2.3450994, 2.3203354
9: -7.8761673, -4.4186783, -7.9090891, -4.4245591, -2.9098854, 2.9376621

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3793559, upper bound: 1.3675941
time: 6.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3793555, upper bound: 1.3678685
time: 12.15 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 31.54 seconds
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.54
Output dim: 8, lower bound: -1.3624108, upper bound: 1.3844415
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 31.54
Output dim: 8, lower bound: -1.3624108, upper bound: 1.3793311
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 31.54
Output dim: 8, lower bound: -1.3793282, upper bound: 1.3624105
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 31.54
Output dim: 8, lower bound: -1.3793282, upper bound: 1.3624104
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 31.54
Output dim: 8, lower bound: -1.3793282, upper bound: 1.3675250
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 31.54
Output dim: 8, lower bound: -1.3793282, upper bound: 1.3677983
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 31.54
Output dim: 8, lower bound: -1.3793558, upper bound: 1.3624892
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 31.54
Output dim: 8, lower bound: -1.3793558, upper bound: 1.3624893
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 31.54
Output dim: 8, lower bound: -1.3793559, upper bound: 1.3675941
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 31.54
Output dim: 8, lower bound: -1.3793555, upper bound: 1.3678685

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.3766317, -5.7680597, -8.4491234, -5.7169828, -2.4373360, 2.4777966
1: -10.8504677, -7.9223251, -10.8846474, -7.8226485, -2.5874543, 2.5543652
2: -5.0520873, -2.3691111, -5.0838614, -2.3186054, -2.3161426, 2.3063531
3: -6.0760603, -2.8785810, -6.1548605, -2.8796194, -3.1384706, 3.2078328
4: -13.4626627, -9.8139248, -13.4879045, -9.8167810, -2.6145339, 2.6238992
5: -3.5638819, -1.5649216, -3.6051371, -1.4781240, -1.4559196, 1.4302166
6: -10.8757286, -8.1546412, -10.9474287, -8.0079870, -2.2249205, 2.1564975
7: -9.6298027, -6.2398481, -9.6843348, -6.2419467, -3.3878560, 3.4444866
8: 9.3493586, 11.9441395, 9.2723904, 11.9704723, -2.3028798, 2.3382926
9: -7.8578162, -4.4287047, -7.9090571, -4.4245615, -2.8621264, 2.8990612

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3565715, upper bound: 1.3840846
time: 7.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3565715, upper bound: 1.3844405
time: 5.25 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 25.08 seconds
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 25.08
Output dim: 8, lower bound: -1.3565715, upper bound: 1.3840846
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 25.08
Output dim: 8, lower bound: -1.3565715, upper bound: 1.3844405

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.3766327, -5.7680602, -8.4594250, -5.7161317, -2.4378242, 2.4805241
1: -10.8504820, -7.9223228, -10.8905640, -7.8194528, -2.5909715, 2.5620022
2: -5.0522594, -2.3691120, -5.0959964, -2.3163781, -2.3168268, 2.3130598
3: -6.0760608, -2.8785384, -6.1609282, -2.8624485, -3.1423588, 3.2033329
4: -13.4626694, -9.8139248, -13.4890194, -9.7757387, -2.6283493, 2.6179593
5: -3.5638821, -1.5649203, -3.6177483, -1.4772674, -1.4562982, 1.4308829
6: -10.8757296, -8.1546383, -10.9602947, -8.0026245, -2.2285824, 2.1578920
7: -9.6298141, -6.2398477, -9.6937237, -6.1949272, -3.4348869, 3.4538760
8: 9.3493586, 11.9441643, 9.2572203, 11.9720173, -2.2960129, 2.3410378
9: -7.8578148, -4.4287043, -7.9147034, -4.4129982, -2.8633661, 2.9066906

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5832

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3521405, upper bound: 1.3837816
time: 6.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3565677, upper bound: 1.3844383
time: 5.50 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 25.06 seconds
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 25.06
Output dim: 8, lower bound: -1.3521405, upper bound: 1.3837816
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 25.06
Output dim: 8, lower bound: -1.3565677, upper bound: 1.3844383

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.3783655, -5.7660456, -8.4594154, -5.7161355, -2.4391451, 2.4816229
1: -10.8673296, -7.9208965, -10.8905621, -7.8194580, -2.5932312, 2.5596814
2: -5.0529251, -2.3633251, -5.0959926, -2.3163810, -2.3163567, 2.3191872
3: -6.0783014, -2.8747618, -6.1609259, -2.8624482, -3.1445265, 3.2065678
4: -13.4638786, -9.8006945, -13.4890175, -9.7757435, -2.6268845, 2.6209679
5: -3.5647008, -1.5642459, -3.6177471, -1.4772701, -1.4567478, 1.4302305
6: -10.8832159, -8.1532001, -10.9602938, -8.0026274, -2.2286932, 2.1570435
7: -9.6352634, -6.2371721, -9.6937218, -6.1949320, -3.4403315, 3.4565496
8: 9.3466845, 11.9505739, 9.2572241, 11.9720154, -2.2962937, 2.3424225
9: -7.8646355, -4.4271607, -7.9147024, -4.4130015, -2.8659325, 2.9062309

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3574599, upper bound: 1.3811942
time: 9.49 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3574599, upper bound: 1.3844333
time: 4.92 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 27.06 seconds
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 27.06
Output dim: 8, lower bound: -1.3574599, upper bound: 1.3811942
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 27.06
Output dim: 8, lower bound: -1.3574599, upper bound: 1.3844333

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.3783646, -5.7660503, -8.4824867, -5.7156343, -2.4334450, 2.4842107
1: -10.8673296, -7.9209089, -10.9273996, -7.8175931, -2.5847769, 2.5791161
2: -5.0529227, -2.3633351, -5.1317186, -2.3150072, -2.3066869, 2.3213010
3: -6.0782948, -2.8747628, -6.1635556, -2.8426437, -3.1641817, 3.2037930
4: -13.4638796, -9.8006973, -13.4982624, -9.7738152, -2.6255517, 2.6205997
5: -3.5646977, -1.5642468, -3.6180058, -1.4647615, -1.4570007, 1.4268115
6: -10.8832150, -8.1532040, -10.9756632, -7.9981341, -2.2293870, 2.1568067
7: -9.6352634, -6.2371731, -9.7020292, -6.1937675, -3.4414959, 3.4648561
8: 9.3466892, 11.9505730, 9.2553072, 11.9805431, -2.3021998, 2.3412344
9: -7.8646336, -4.4271617, -7.9223404, -4.4125185, -2.8663535, 2.9102058

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3532703, upper bound: 1.3785782
time: 8.23 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3574530, upper bound: 1.3844244
time: 6.68 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 27.80 seconds
IS_A1_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 27.80
Output dim: 8, lower bound: -1.3532703, upper bound: 1.3785782
IS_A1_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 27.80
Output dim: 8, lower bound: -1.3574530, upper bound: 1.3844244

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.3783598, -5.7660527, -8.4824867, -5.7156343, -2.4334431, 2.5095410
1: -10.8673258, -7.9209208, -10.9273996, -7.8175931, -2.5746684, 2.5493221
2: -5.0529141, -2.3633385, -5.1317186, -2.3150072, -2.2784004, 2.3106275
3: -6.0782814, -2.8747649, -6.1635556, -2.8426437, -3.1459208, 3.1901479
4: -13.4638786, -9.8007059, -13.4982624, -9.7738152, -2.6199191, 2.5741975
5: -3.5646985, -1.5642545, -3.6180058, -1.4647615, -1.4517238, 1.4176558
6: -10.8832130, -8.1532164, -10.9756632, -7.9981341, -2.2215021, 2.1422863
7: -9.6352491, -6.2371774, -9.7020292, -6.1937675, -3.4414816, 3.4648519
8: 9.3466930, 11.9505739, 9.2553072, 11.9805431, -2.2930756, 2.3369958
9: -7.8646331, -4.4271679, -7.9223404, -4.4125185, -2.8663526, 2.8953545

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6137

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3574530, upper bound: 1.3802440
time: 5.25 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3574531, upper bound: 1.3844266
time: 4.92 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 22.95 seconds
IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 22.95
Output dim: 8, lower bound: -1.3574530, upper bound: 1.3802440
IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 22.95
Output dim: 8, lower bound: -1.3574531, upper bound: 1.3844266

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.3783598, -5.7660527, -8.4824848, -5.7156353, -2.4587741, 2.5095403
1: -10.8673258, -7.9209208, -10.9274025, -7.8176007, -2.5550184, 2.5493207
2: -5.0529141, -2.3633385, -5.1317096, -2.3150077, -2.2741375, 2.2887528
3: -6.0782814, -2.8747649, -6.1635389, -2.8426476, -3.1459179, 3.1760216
4: -13.4638786, -9.8007059, -13.4982624, -9.7738285, -2.5792072, 2.5719519
5: -3.5646985, -1.5642545, -3.6180062, -1.4647694, -1.4446431, 1.4144505
6: -10.8832130, -8.1532164, -10.9756622, -7.9981475, -2.2102723, 2.1376929
7: -9.6352491, -6.2371774, -9.7020168, -6.1937718, -3.4414773, 3.4648395
8: 9.3466930, 11.9505739, 9.2553120, 11.9805422, -2.2930746, 2.3321257
9: -7.8646331, -4.4271679, -7.9223394, -4.4125237, -2.8515100, 2.8914678

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4556

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3543931, upper bound: 1.3802288
time: 8.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3574390, upper bound: 1.3802291
time: 8.44 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 29.96 seconds
IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 29.96
Output dim: 8, lower bound: -1.3543931, upper bound: 1.3802288
IS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 29.96
Output dim: 8, lower bound: -1.3574390, upper bound: 1.3802291
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.359529972076416
rel_dist={8: [-1.3847687820365167, 1.3847690605870273]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 2196.77 seconds
