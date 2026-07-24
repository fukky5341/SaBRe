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
execution time: IAR + LP analysis = 13.90 + 57.69 = 71.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.9845428, upper bound: 1.9845425


# Binary Search by BASE starts (time budget: 3528.41 seconds, max iter: 100)

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
Binary search time: 193.93 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual) starts
Time budget: 3334.48 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

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
time: 7.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.65 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.65
Output dim: 8, lower bound: -1.6835984, upper bound: 1.7089538
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.65
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

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6773079, upper bound: 1.7089442
time: 7.66 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6835883, upper bound: 1.7089438
time: 5.59 seconds

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

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7027470, upper bound: 1.7090300
time: 5.50 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090276, upper bound: 1.7090282
time: 6.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.29 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 24.29
Output dim: 8, lower bound: -1.6773079, upper bound: 1.7089442
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 24.29
Output dim: 8, lower bound: -1.6835883, upper bound: 1.7089438
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 24.29
Output dim: 8, lower bound: -1.7027470, upper bound: 1.7090300
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 24.29
Output dim: 8, lower bound: -1.7090276, upper bound: 1.7090282

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -8.3766346, -5.7815909, -8.4329529, -5.7627630, -2.6138716, 2.6513619
1: -10.8338366, -7.9634423, -10.8692474, -7.8538485, -2.8168612, 2.7544093
2: -5.0076952, -2.3868999, -5.0695200, -2.3609076, -2.4951134, 2.5238447
3: -5.9643888, -2.9212120, -6.1023827, -2.8842039, -3.0801849, 3.1811707
4: -13.4435635, -9.9077702, -13.4638796, -9.8396969, -2.9494679, 2.9126248
5: -3.5470784, -1.5622070, -3.5621121, -1.4912126, -1.6502910, 1.6093289
6: -10.8550863, -8.1602345, -10.8795338, -8.0441151, -2.4880364, 2.4133825
7: -9.5659828, -6.3136139, -9.6477757, -6.2786455, -3.2873373, 3.3341618
8: 9.3722181, 11.9387741, 9.2907581, 11.9534855, -2.4876080, 2.5537043
9: -7.8385153, -4.4696131, -7.8679147, -4.4364452, -3.0855207, 3.0853729

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6771561, upper bound: 1.7010366
time: 5.82 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6772940, upper bound: 1.7089299
time: 8.88 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -8.4018412, -5.7667747, -8.4365664, -5.7612162, -2.6406250, 2.6697917
1: -10.8537636, -7.9223051, -10.8703604, -7.8470478, -2.8501272, 2.7781720
2: -5.0640087, -2.3662405, -5.0786452, -2.3596354, -2.5353012, 2.5583801
3: -6.0806050, -2.8903475, -6.1212358, -2.8829832, -3.1976218, 3.2308884
4: -13.4631748, -9.8374052, -13.4647236, -9.8281202, -2.9806681, 2.9464295
5: -3.5557165, -1.5313776, -3.5625110, -1.4863625, -1.6708539, 1.6342429
6: -10.8669624, -8.1119614, -10.8799944, -8.0364714, -2.5147824, 2.4503436
7: -9.6404953, -6.2830453, -9.6593380, -6.2762232, -3.3642721, 3.3762927
8: 9.3482828, 11.9452095, 9.2870598, 11.9537258, -2.5118217, 2.5683022
9: -7.8587065, -4.4339314, -7.8696723, -4.4306927, -3.1139927, 3.1114612

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6834367, upper bound: 1.7010363
time: 6.64 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6835744, upper bound: 1.7089287
time: 5.58 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -8.4157314, -5.7757835, -8.4372158, -5.7624941, -2.6532373, 2.6614323
1: -10.8514919, -7.8788619, -10.8702688, -7.8445463, -2.8550558, 2.7860727
2: -5.0242176, -2.3796759, -5.0713034, -2.3603048, -2.5086198, 2.5390358
3: -6.0100431, -2.9134889, -6.1072726, -2.8838422, -3.1262009, 3.1937838
4: -13.4452019, -9.8976803, -13.4639711, -9.8386488, -2.9829497, 2.9218192
5: -3.5541248, -1.5116261, -3.5623677, -1.4856489, -1.6759648, 1.6353877
6: -10.8687630, -8.0754852, -10.8801517, -8.0349302, -2.5228724, 2.4393206
7: -9.5869789, -6.3062191, -9.6498680, -6.2780619, -3.3089170, 3.3436489
8: 9.3034201, 11.9475269, 9.2831469, 11.9537640, -2.5201793, 2.5711007
9: -7.8504090, -4.4658976, -7.8688931, -4.4359369, -3.1174908, 3.0903349

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7025948, upper bound: 1.7011207
time: 8.52 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7027326, upper bound: 1.7090134
time: 6.67 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -8.4408216, -5.7609487, -8.4408283, -5.7609482, -2.6798735, 2.6798797
1: -10.8713799, -7.8377633, -10.8713818, -7.8377485, -2.8843503, 2.8098145
2: -5.0804210, -2.3590345, -5.0804310, -2.3590326, -2.5487337, 2.5735345
3: -6.1261096, -2.8826232, -6.1261268, -2.8826227, -3.2434869, 3.2435036
4: -13.4648132, -9.8270864, -13.4648161, -9.8270741, -3.0141516, 2.9556935
5: -3.5627639, -1.4808104, -3.5627654, -1.4808009, -1.6920595, 1.6602744
6: -10.8806114, -8.0273066, -10.8806124, -8.0272884, -2.5431185, 2.4763503
7: -9.6614227, -6.2756424, -9.6614361, -6.2756395, -3.3857832, 3.3857937
8: 9.2794533, 11.9540052, 9.2794437, 11.9540062, -2.5443711, 2.5857353
9: -7.8706450, -4.4301906, -7.8706493, -4.4301844, -3.1460485, 3.1164331

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7088753, upper bound: 1.7011207
time: 7.77 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090132, upper bound: 1.7090137
time: 6.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.68 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 8, lower bound: -1.6771561, upper bound: 1.7010366
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 8, lower bound: -1.6772940, upper bound: 1.7089299
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 8, lower bound: -1.6834367, upper bound: 1.7010363
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 8, lower bound: -1.6835744, upper bound: 1.7089287
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 8, lower bound: -1.7025948, upper bound: 1.7011207
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 8, lower bound: -1.7027326, upper bound: 1.7090134
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 8, lower bound: -1.7088753, upper bound: 1.7011207
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 26.68
Output dim: 8, lower bound: -1.7090132, upper bound: 1.7090137

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -8.3733549, -5.7817841, -8.3970852, -5.7648745, -2.6084805, 2.6153011
1: -10.8329926, -7.9637232, -10.8601770, -7.8570104, -2.7950010, 2.7306738
2: -5.0054803, -2.3873644, -5.0451818, -2.3658524, -2.4862094, 2.4969244
3: -5.9634304, -2.9217224, -6.0917902, -2.8895924, -3.0738380, 3.1700678
4: -13.4434195, -9.9094124, -13.4623060, -9.8573618, -2.9270384, 2.9091573
5: -3.5466423, -1.5653715, -3.5577075, -1.5259149, -1.6143897, 1.6001842
6: -10.8547077, -8.1646528, -10.8755112, -8.0925694, -2.4355440, 2.4036736
7: -9.5641289, -6.3139739, -9.6274929, -6.2822738, -3.2818551, 3.3135190
8: 9.3737116, 11.9385386, 9.3070488, 11.9510250, -2.4833660, 2.5377235
9: -7.8379202, -4.4701886, -7.8616476, -4.4426227, -3.0700521, 3.0673490

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 918

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6688110, upper bound: 1.7009526
time: 6.23 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6771427, upper bound: 1.7010237
time: 6.64 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.3766308, -5.7815905, -8.4415207, -5.7187920, -2.6578388, 2.6599302
1: -10.8338356, -7.9634452, -10.8825245, -7.8383880, -2.8232021, 2.7554913
2: -5.0076914, -2.3869019, -5.0732141, -2.3204684, -2.5294137, 2.5288749
3: -5.9643869, -2.9212115, -6.1312461, -2.8811767, -3.0832102, 3.2100346
4: -13.4435644, -9.9077768, -13.4870100, -9.8291998, -2.9613888, 2.9362531
5: -3.5470774, -1.5622104, -3.6044884, -1.4884026, -1.6485655, 1.6330081
6: -10.8550863, -8.1602392, -10.9463701, -8.0245256, -2.5073700, 2.4502337
7: -9.5659781, -6.3136153, -9.6707802, -6.2449245, -3.3210535, 3.3571649
8: 9.3722219, 11.9387722, 9.2836895, 11.9699860, -2.5048685, 2.5599108
9: -7.8385124, -4.4696145, -7.9063559, -4.4308033, -3.0829997, 3.1287916

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6689489, upper bound: 1.7088467
time: 5.25 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6772806, upper bound: 1.7089160
time: 6.09 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.3985672, -5.7669663, -8.4007025, -5.7633276, -2.6352396, 2.6337361
1: -10.8529224, -7.9225807, -10.8613014, -7.8502116, -2.8282499, 2.7543826
2: -5.0617943, -2.3667030, -5.0543079, -2.3645720, -2.5264053, 2.5314643
3: -6.0796456, -2.8908565, -6.1106424, -2.8883703, -3.1912754, 3.2197859
4: -13.4630280, -9.8390589, -13.4631510, -9.8457870, -2.9582396, 2.9429634
5: -3.5552833, -1.5345434, -3.5581057, -1.5210671, -1.6349481, 1.6250975
6: -10.8665857, -8.1163750, -10.8759747, -8.0849161, -2.4622922, 2.4406374
7: -9.6386414, -6.2834020, -9.6390495, -6.2798514, -3.3587899, 3.3556476
8: 9.3497829, 11.9449749, 9.3033619, 11.9512682, -2.5075865, 2.5523210
9: -7.8581114, -4.4345083, -7.8634005, -4.4368649, -3.0985165, 3.0934324

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6750917, upper bound: 1.7009524
time: 5.71 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6834232, upper bound: 1.7010238
time: 6.21 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.4018383, -5.7667723, -8.4451294, -5.7172432, -2.6845951, 2.6783571
1: -10.8537636, -7.9223042, -10.8836374, -7.8315849, -2.8564625, 2.7791600
2: -5.0640063, -2.3662434, -5.0823359, -2.3191953, -2.5696902, 2.5634065
3: -6.0806036, -2.8903465, -6.1500888, -2.8799546, -3.2006490, 3.2597423
4: -13.4631729, -9.8374100, -13.4878559, -9.8176260, -2.9959927, 2.9700580
5: -3.5557165, -1.5313810, -3.6048863, -1.4835569, -1.6691104, 1.6579528
6: -10.8669624, -8.1119652, -10.9468307, -8.0168819, -2.5340981, 2.4873047
7: -9.6404886, -6.2830434, -9.6823311, -6.2425041, -3.3979845, 3.3992877
8: 9.3482866, 11.9452114, 9.2799883, 11.9702320, -2.5290470, 2.5745234
9: -7.8587074, -4.4339318, -7.9081106, -4.4250660, -3.1114321, 3.1549969

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6752294, upper bound: 1.7088445
time: 7.25 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6835610, upper bound: 1.7089160
time: 5.90 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.4124107, -5.7759757, -8.4013052, -5.7646036, -2.6478071, 2.6253295
1: -10.8506584, -7.8791389, -10.8612165, -7.8476963, -2.8337173, 2.7623067
2: -5.0219769, -2.3801289, -5.0469465, -2.3652363, -2.4997106, 2.5122027
3: -6.0090885, -2.9139996, -6.0966945, -2.8892322, -3.1198564, 3.1826949
4: -13.4450598, -9.8993578, -13.4623995, -9.8564310, -2.9602807, 2.9183903
5: -3.5536962, -1.5148163, -3.5579679, -1.5203658, -1.6400974, 1.6262341
6: -10.8683939, -8.0799637, -10.8761425, -8.0834551, -2.4701295, 2.4296756
7: -9.5851021, -6.3065653, -9.6295557, -6.2816768, -3.3034253, 3.3229904
8: 9.3049164, 11.9473038, 9.2994385, 11.9513159, -2.5159502, 2.5551286
9: -7.8498497, -4.4664650, -7.8626642, -4.4421039, -3.1019669, 3.0723724

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6942497, upper bound: 1.7010390
time: 4.79 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7025814, upper bound: 1.7011098
time: 5.68 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.4157257, -5.7757835, -8.4457769, -5.7185221, -2.6972036, 2.6699934
1: -10.8514881, -7.8788638, -10.8835411, -7.8291612, -2.8822241, 2.7871175
2: -5.0242128, -2.3796778, -5.0749941, -2.3198681, -2.5442486, 2.5439229
3: -6.0100422, -2.9134867, -6.1360507, -2.8808105, -3.1292317, 3.2225640
4: -13.4452009, -9.8976879, -13.4871025, -9.8282099, -2.9932554, 2.9454460
5: -3.5541258, -1.5116347, -3.6047399, -1.4828701, -1.6753933, 1.6590613
6: -10.8687639, -8.0754957, -10.9469814, -8.0153828, -2.5508404, 2.4862077
7: -9.5869694, -6.3062201, -9.6728411, -6.2443471, -3.3426223, 3.3666210
8: 9.3034239, 11.9475288, 9.2760773, 11.9702625, -2.5374460, 2.5773020
9: -7.8504076, -4.4658985, -7.9073434, -4.4302959, -3.1148634, 3.1350427

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6943876, upper bound: 1.7089291
time: 7.58 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7027191, upper bound: 1.7090001
time: 6.33 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4375067, -5.7611403, -8.4049234, -5.7630568, -2.6744499, 2.6437831
1: -10.8705530, -7.8380389, -10.8623381, -7.8408966, -2.8630123, 2.7859941
2: -5.0781808, -2.3594837, -5.0560732, -2.3639569, -2.5398316, 2.5467057
3: -6.1251502, -2.8831367, -6.1155438, -2.8880143, -3.2371359, 3.2324071
4: -13.4646711, -9.8287611, -13.4632463, -9.8448553, -2.9914804, 2.9522645
5: -3.5623345, -1.4840006, -3.5583673, -1.5155169, -1.6561904, 1.6511202
6: -10.8802423, -8.0317822, -10.8766012, -8.0758047, -2.4903774, 2.4667065
7: -9.6595421, -6.2759871, -9.6411133, -6.2792544, -3.3802876, 3.3651261
8: 9.2809572, 11.9537821, 9.2957544, 11.9515572, -2.5401459, 2.5697641
9: -7.8700843, -4.4307542, -7.8644161, -4.4363446, -3.1305180, 3.0984674

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7005304, upper bound: 1.7010390
time: 6.12 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7088618, upper bound: 1.7011080
time: 8.01 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4408169, -5.7609501, -8.4493856, -5.7169747, -2.7238421, 2.6884356
1: -10.8713770, -7.8377652, -10.8846550, -7.8223615, -2.9155674, 2.8107657
2: -5.0804157, -2.3590331, -5.0841165, -2.3185945, -2.5844851, 2.5784178
3: -6.1261082, -2.8826244, -6.1548944, -2.8795891, -3.2465191, 3.2722700
4: -13.4648113, -9.8270922, -13.4879475, -9.8166351, -3.0285275, 2.9793205
5: -3.5627646, -1.4808184, -3.6051393, -1.4780251, -1.6914828, 1.6839787
6: -10.8806105, -8.0273142, -10.9474392, -8.0077410, -2.5710831, 2.5232985
7: -9.6614151, -6.2756433, -9.6843891, -6.2419243, -3.4194908, 3.4087458
8: 9.2794590, 11.9540043, 9.2723780, 11.9705076, -2.5616016, 2.5919514
9: -7.8706450, -4.4301896, -7.9090972, -4.4245591, -3.1433768, 3.1612563

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7006681, upper bound: 1.7089292
time: 5.94 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7089996, upper bound: 1.7090001
time: 6.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.86 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.6688110, upper bound: 1.7009526
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.6771427, upper bound: 1.7010237
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.6689489, upper bound: 1.7088467
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.6772806, upper bound: 1.7089160
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.6750917, upper bound: 1.7009524
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.6834232, upper bound: 1.7010238
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.6752294, upper bound: 1.7088445
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.6835610, upper bound: 1.7089160
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.6942497, upper bound: 1.7010390
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.7025814, upper bound: 1.7011098
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.6943876, upper bound: 1.7089291
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.7027191, upper bound: 1.7090001
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.7005304, upper bound: 1.7010390
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.7088618, upper bound: 1.7011080
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.7006681, upper bound: 1.7089292
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.86
Output dim: 8, lower bound: -1.7089996, upper bound: 1.7090001

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.3731918, -5.7821894, -8.3941259, -5.7726021, -2.6005898, 2.6119366
1: -10.8328362, -7.9640431, -10.8572445, -7.8629951, -2.7880402, 2.7156215
2: -5.0053964, -2.3877792, -5.0436921, -2.3734579, -2.4786649, 2.4954653
3: -5.9624290, -2.9219308, -6.0729637, -2.8936534, -3.0687757, 3.1510329
4: -13.4418764, -9.9097300, -13.4327478, -9.8637409, -2.9204898, 2.8778141
5: -3.5466199, -1.5659223, -3.5572767, -1.5360541, -1.5982531, 1.5956912
6: -10.8545475, -8.1656466, -10.8724451, -8.1114807, -2.4149203, 2.4009669
7: -9.5618525, -6.3141451, -9.5840750, -6.2855716, -3.2762809, 3.2699299
8: 9.3738661, 11.9380836, 9.3099728, 11.9423637, -2.4722943, 2.5321507
9: -7.8368082, -4.4702787, -7.8413844, -4.4443269, -3.0605211, 3.0341024

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6688100, upper bound: 1.6926910
time: 6.06 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6688101, upper bound: 1.7009528
time: 5.81 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.3733568, -5.7817879, -8.4073563, -5.7640219, -2.6093349, 2.6255684
1: -10.8329906, -7.9637256, -10.8660746, -7.8538237, -2.7976012, 2.7376838
2: -5.0054803, -2.3873687, -5.0573206, -2.3636689, -2.4879246, 2.5065389
3: -5.9634266, -2.9217203, -6.0978274, -2.8724086, -3.0910180, 3.1761072
4: -13.4434061, -9.9094133, -13.4634018, -9.8163109, -2.9347341, 2.9088638
5: -3.5466428, -1.5653725, -3.5703259, -1.5250647, -1.6156657, 1.6105616
6: -10.8547077, -8.1646566, -10.8883858, -8.0874500, -2.4397602, 2.4166679
7: -9.5641232, -6.3139758, -9.6366758, -6.2352901, -3.3288331, 3.3227000
8: 9.3737116, 11.9385357, 9.2918673, 11.9525509, -2.4879789, 2.5403469
9: -7.8379169, -4.4701896, -7.8672538, -4.4311061, -3.0747910, 3.0771198

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6771425, upper bound: 1.6756651
time: 5.98 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6771427, upper bound: 1.7010237
time: 6.77 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.3764687, -5.7819939, -8.4386940, -5.7265182, -2.6499505, 2.6567001
1: -10.8336811, -7.9637647, -10.8796034, -7.8444796, -2.8162479, 2.7404346
2: -5.0076089, -2.3873167, -5.0717359, -2.3280916, -2.5218859, 2.5274055
3: -5.9633827, -2.9214213, -6.1121945, -2.8852391, -3.0781436, 3.1907732
4: -13.4420233, -9.9080925, -13.4574471, -9.8353939, -2.9535999, 2.9049044
5: -3.5470550, -1.5627606, -3.6040657, -1.4985454, -1.6324567, 1.6281449
6: -10.8549261, -8.1612301, -10.9433165, -8.0433540, -2.4868579, 2.4468055
7: -9.5637016, -6.3137875, -9.6275473, -6.2482333, -3.3154683, 3.3137598
8: 9.3723745, 11.9383154, 9.2865973, 11.9613228, -2.4938021, 2.5543370
9: -7.8374000, -4.4697051, -7.8857746, -4.4324732, -3.0735197, 3.0953512

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6689478, upper bound: 1.7005850
time: 4.80 seconds

## Relational analysis of IS_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6689478, upper bound: 1.7088467
time: 5.19 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.3766308, -5.7815914, -8.4518166, -5.7179432, -2.6586876, 2.6702251
1: -10.8338356, -7.9634457, -10.8884401, -7.8351989, -2.8258672, 2.7625182
2: -5.0076904, -2.3869052, -5.0853496, -2.3182545, -2.5311723, 2.5346587
3: -5.9643803, -2.9212132, -6.1373053, -2.8640063, -3.1003740, 3.2160921
4: -13.4435511, -9.9077768, -13.4881067, -9.7881527, -2.9663115, 2.9359572
5: -3.5470777, -1.5622115, -3.6171007, -1.4875495, -1.6498482, 1.6311616
6: -10.8550854, -8.1602459, -10.9592342, -8.0191727, -2.5117760, 2.4516008
7: -9.5659695, -6.3136158, -9.6802111, -6.1979032, -3.3680663, 3.3665953
8: 9.3722210, 11.9387684, 9.2685108, 11.9715137, -2.5094771, 2.5626490
9: -7.8385096, -4.4696155, -7.9119835, -4.4192333, -3.0878258, 3.1344790

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6772806, upper bound: 1.6835612
time: 6.41 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6772806, upper bound: 1.7089163
time: 6.02 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.3984051, -5.7673712, -8.3977451, -5.7710547, -2.6273503, 2.6303740
1: -10.8527699, -7.9229031, -10.8583727, -7.8562031, -2.8212733, 2.7392855
2: -5.0617113, -2.3671188, -5.0528202, -2.3721805, -2.5188603, 2.5300078
3: -6.0786424, -2.8910666, -6.0918088, -2.8924317, -3.1862106, 3.2007422
4: -13.4614878, -9.8393803, -13.4335899, -9.8521681, -2.9515285, 2.9116647
5: -3.5552607, -1.5350912, -3.5576768, -1.5312061, -1.6188509, 1.6206052
6: -10.8664255, -8.1173687, -10.8729067, -8.1038389, -2.4416542, 2.4381514
7: -9.6363611, -6.2835717, -9.5956192, -6.2831473, -3.3532138, 3.3120475
8: 9.3499393, 11.9445200, 9.3062897, 11.9426031, -2.4965229, 2.5467453
9: -7.8569932, -4.4345970, -7.8431888, -4.4385672, -3.0889826, 3.0601726

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6750907, upper bound: 1.6926925
time: 5.69 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6750908, upper bound: 1.7009545
time: 5.40 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.3985643, -5.7669678, -8.4109688, -5.7624741, -2.6360903, 2.6440010
1: -10.8529224, -7.9225845, -10.8671970, -7.8470240, -2.8308325, 2.7613621
2: -5.0617938, -2.3667068, -5.0664387, -2.3623848, -2.5281277, 2.5437622
3: -6.0796404, -2.8908596, -6.1166840, -2.8711908, -3.2084496, 3.2258244
4: -13.4630165, -9.8390589, -13.4642458, -9.8047438, -2.9765203, 2.9427207
5: -3.5552824, -1.5345435, -3.5707226, -1.5202155, -1.6362195, 1.6354733
6: -10.8665848, -8.1163807, -10.8888464, -8.0797968, -2.4664845, 2.4536300
7: -9.6386337, -6.2834010, -9.6482086, -6.2328739, -3.4057598, 3.3648076
8: 9.3497849, 11.9449711, 9.2881908, 11.9527979, -2.5121737, 2.5563879
9: -7.8581066, -4.4345083, -7.8690290, -4.4253550, -3.1032352, 3.1031981

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6756644, upper bound: 1.7010252
time: 5.21 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6756644, upper bound: 1.7010252
time: 8.08 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.4016762, -5.7671771, -8.4423008, -5.7249722, -2.6767039, 2.6751237
1: -10.8536091, -7.9226232, -10.8807192, -7.8376865, -2.8494911, 2.7640777
2: -5.0639229, -2.3666582, -5.0808611, -2.3268194, -2.5621619, 2.5619376
3: -6.0795989, -2.8905585, -6.1310329, -2.8840148, -3.1955841, 3.2404745
4: -13.4616327, -9.8377390, -13.4582911, -9.8238182, -2.9892604, 2.9387562
5: -3.5556948, -1.5319309, -3.6044636, -1.4936986, -1.6530423, 1.6530893
6: -10.8668041, -8.1129608, -10.9437733, -8.0357170, -2.5135713, 2.4841015
7: -9.6382132, -6.2832155, -9.6390572, -6.2458048, -3.3924084, 3.3558416
8: 9.3484411, 11.9447508, 9.2829018, 11.9615660, -2.5179911, 2.5689459
9: -7.8575916, -4.4340224, -7.8875818, -4.4267340, -3.1019526, 3.1215398

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6752285, upper bound: 1.7005832
time: 7.03 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6752286, upper bound: 1.7088451
time: 6.91 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.4018373, -5.7667742, -8.4554205, -5.7163973, -2.6854401, 2.6886463
1: -10.8537655, -7.9223042, -10.8895512, -7.8283958, -2.8591080, 2.7861586
2: -5.0640049, -2.3662472, -5.0944629, -2.3169765, -2.5714569, 2.5756941
3: -6.0805974, -2.8903489, -6.1561499, -2.8627875, -3.2178099, 3.2658010
4: -13.4631634, -9.8374109, -13.4889517, -9.7765913, -3.0080836, 2.9698138
5: -3.5557168, -1.5313820, -3.6174989, -1.4827029, -1.6703891, 1.6561062
6: -10.8669624, -8.1119709, -10.9596930, -8.0115280, -2.5384808, 2.4886723
7: -9.6404819, -6.2830482, -9.6917362, -6.1954861, -3.4449959, 3.4086881
8: 9.3482857, 11.9452066, 9.2648163, 11.9717674, -2.5336313, 2.5787082
9: -7.8587017, -4.4339333, -7.9137635, -4.4135046, -3.1162462, 3.1607075

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6835610, upper bound: 1.6835609
time: 7.21 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6835611, upper bound: 1.7089158
time: 6.31 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.4122505, -5.7763782, -8.3983507, -5.7723312, -2.6399193, 2.6219726
1: -10.8505096, -7.8794546, -10.8582850, -7.8536711, -2.8268738, 2.7472391
2: -5.0218940, -2.3805437, -5.0454545, -2.3728423, -2.4921656, 2.5107431
3: -6.0080929, -2.9142098, -6.0778818, -2.8932950, -3.1147978, 3.1636720
4: -13.4435158, -9.8996830, -13.4328403, -9.8628187, -2.9538724, 2.8870387
5: -3.5536733, -1.5153638, -3.5575390, -1.5305061, -1.6234956, 1.6217413
6: -10.8682318, -8.0809441, -10.8730745, -8.1023436, -2.4496064, 2.4269409
7: -9.5828352, -6.3067346, -9.5861588, -6.2849727, -3.2978625, 3.2794242
8: 9.3050709, 11.9468479, 9.3023586, 11.9426527, -2.5048771, 2.5495553
9: -7.8487325, -4.4665513, -7.8423891, -4.4438028, -3.0924168, 3.0391426

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6942488, upper bound: 1.6927770
time: 5.01 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6942488, upper bound: 1.7010390
time: 4.83 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.4124107, -5.7759790, -8.4115820, -5.7637501, -2.6486607, 2.6356030
1: -10.8506603, -7.8791409, -10.8671074, -7.8445086, -2.8363867, 2.7693110
2: -5.0219765, -2.3801322, -5.0590839, -2.3630519, -2.5014277, 2.5245073
3: -6.0090828, -2.9139998, -6.1027265, -2.8720508, -3.1370320, 3.1887267
4: -13.4450474, -9.8993597, -13.4634981, -9.8153801, -2.9670563, 2.9180954
5: -3.5536947, -1.5148172, -3.5705855, -1.5195148, -1.6452508, 1.6366103
6: -10.8683910, -8.0799694, -10.8890142, -8.0783329, -2.4743567, 2.4426689
7: -9.5850945, -6.3065648, -9.6387091, -6.2346916, -3.3504028, 3.3321443
8: 9.3049173, 11.9473000, 9.2842607, 11.9528408, -2.5205612, 2.5681038
9: -7.8498445, -4.4664636, -7.8682542, -4.4305873, -3.1066866, 3.0821409

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7024965, upper bound: 1.6756651
time: 4.91 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7024977, upper bound: 1.6758493
time: 5.35 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.4155655, -5.7761874, -8.4429550, -5.7262502, -2.6893153, 2.6667676
1: -10.8513365, -7.8791761, -10.8806248, -7.8352451, -2.8753214, 2.7720652
2: -5.0241308, -2.3800931, -5.0735173, -2.3274922, -2.5367203, 2.5424519
3: -6.0090466, -2.9136994, -6.1170135, -2.8848732, -3.1241734, 3.2033141
4: -13.4436617, -9.8980112, -13.4575386, -9.8344069, -2.9855723, 2.9140923
5: -3.5541031, -1.5121845, -3.6043162, -1.4930127, -1.6588316, 1.6541981
6: -10.8686037, -8.0764751, -10.9439297, -8.0341883, -2.5301685, 2.4827592
7: -9.5847054, -6.3063927, -9.6295929, -6.2476521, -3.3370533, 3.3232002
8: 9.3035774, 11.9470730, 9.2789879, 11.9615984, -2.5263805, 2.5717278
9: -7.8492913, -4.4659867, -7.8867512, -4.4319630, -3.1053653, 3.1016188

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6943866, upper bound: 1.7006670
time: 7.03 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6943866, upper bound: 1.7089287
time: 7.86 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.4157257, -5.7757864, -8.4560823, -5.7176752, -2.6980505, 2.6802959
1: -10.8514891, -7.8788643, -10.8894567, -7.8259692, -2.8849115, 2.7941413
2: -5.0242133, -2.3796802, -5.0871296, -2.3176532, -2.5460076, 2.5539391
3: -6.0100369, -2.9134893, -6.1421070, -2.8636403, -3.1463966, 3.2286177
4: -13.4451904, -9.8976889, -13.4881983, -9.7871609, -2.9982042, 2.9451513
5: -3.5541239, -1.5116355, -3.6173534, -1.4820180, -1.6805401, 1.6572144
6: -10.8687639, -8.0754976, -10.9598465, -8.0100288, -2.5551109, 2.4875762
7: -9.5869656, -6.3062229, -9.6822376, -6.1973248, -3.3896408, 3.3760147
8: 9.3034248, 11.9475250, 9.2609005, 11.9717941, -2.5420547, 2.5902681
9: -7.8504019, -4.4659004, -7.9129601, -4.4187279, -3.1196637, 3.1407142

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7026344, upper bound: 1.6835629
time: 7.54 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7026354, upper bound: 1.6837426
time: 4.83 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -8.4373474, -5.7615442, -8.4019690, -5.7707810, -2.6665664, 2.6404247
1: -10.8704004, -7.8383551, -10.8594112, -7.8468790, -2.8561640, 2.7709007
2: -5.0780988, -2.3599005, -5.0545835, -2.3715663, -2.5322871, 2.5452492
3: -6.1241555, -2.8833442, -6.0967274, -2.8920715, -3.2320840, 3.2133832
4: -13.4631271, -9.8290863, -13.4336863, -9.8512440, -2.9849105, 2.9209614
5: -3.5623124, -1.4845496, -3.5579381, -1.5256550, -1.6396902, 1.6466274
6: -10.8800812, -8.0327625, -10.8735342, -8.0947018, -2.4698467, 2.4641933
7: -9.6572752, -6.2761574, -9.5977039, -6.2825513, -3.3747239, 3.3215466
8: 9.2811146, 11.9533234, 9.2986794, 11.9428921, -2.5290828, 2.5641856
9: -7.8689613, -4.4308434, -7.8441949, -4.4380422, -3.1209621, 3.0652208

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7005294, upper bound: 1.6927770
time: 5.74 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7005294, upper bound: 1.7010390
time: 5.53 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -8.4375057, -5.7611423, -8.4151936, -5.7622008, -2.6753049, 2.6540513
1: -10.8705521, -7.8380432, -10.8682327, -7.8377113, -2.8656902, 2.7929688
2: -5.0781798, -2.3594866, -5.0682020, -2.3617673, -2.5415545, 2.5590041
3: -6.1251459, -2.8831356, -6.1215820, -2.8708324, -3.2543135, 3.2384465
4: -13.4646597, -9.8287649, -13.4643412, -9.8038130, -3.0088401, 2.9520221
5: -3.5623353, -1.4840019, -3.5709848, -1.5146663, -1.6613426, 1.6614943
6: -10.8802404, -8.0317841, -10.8894730, -8.0706825, -2.4946046, 2.4796989
7: -9.6595364, -6.2759862, -9.6502705, -6.2322745, -3.4272618, 3.3742843
8: 9.2809582, 11.9537754, 9.2805824, 11.9530859, -2.5447350, 2.5827293
9: -7.8700790, -4.4307547, -7.8700328, -4.4248343, -3.1352081, 3.1082315

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7087770, upper bound: 1.6756627
time: 6.67 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7087782, upper bound: 1.6758472
time: 6.93 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -8.4406595, -5.7613516, -8.4465656, -5.7247043, -2.7159553, 2.6852140
1: -10.8712254, -7.8380795, -10.8817387, -7.8284497, -2.9086533, 2.7956872
2: -5.0803356, -2.3594508, -5.0826421, -2.3262205, -2.5769548, 2.5769486
3: -6.1251125, -2.8828335, -6.1358538, -2.8836472, -3.2414653, 3.2530203
4: -13.4632721, -9.8274164, -13.4583836, -9.8228331, -3.0219245, 2.9480150
5: -3.5627418, -1.4813666, -3.6047151, -1.4881656, -1.6750231, 1.6791155
6: -10.8804522, -8.0282974, -10.9443874, -8.0265560, -2.5504036, 2.5200751
7: -9.6591473, -6.2758131, -9.6411314, -6.2452288, -3.4139185, 3.3653183
8: 9.2796135, 11.9535446, 9.2752895, 11.9618416, -2.5505428, 2.5863724
9: -7.8695245, -4.4302788, -7.8885622, -4.4262228, -3.1338778, 3.1278143

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7006671, upper bound: 1.7006693
time: 6.23 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7006671, upper bound: 1.7089309
time: 5.17 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -8.4408169, -5.7609506, -8.4596844, -5.7161283, -2.7246885, 2.6987338
1: -10.8713770, -7.8377647, -10.8905678, -7.8191686, -2.9182358, 2.8177600
2: -5.0804152, -2.3590379, -5.0962415, -2.3163743, -2.5862498, 2.5907049
3: -6.1261015, -2.8826263, -6.1609535, -2.8624218, -3.2636797, 3.2783272
4: -13.4648018, -9.8270969, -13.4890413, -9.7755976, -3.0399725, 2.9790759
5: -3.5627656, -1.4808214, -3.6177502, -1.4771717, -1.6966281, 1.6821320
6: -10.8806124, -8.0273209, -10.9603043, -8.0023861, -2.5753517, 2.5246677
7: -9.6614084, -6.2756443, -9.6937656, -6.1949067, -3.4665017, 3.4181213
8: 9.2794600, 11.9540014, 9.2572079, 11.9720421, -2.5661840, 2.6049070
9: -7.8706398, -4.4301910, -7.9147372, -4.4129963, -3.1481676, 3.1669502

Time for backsubstitution: 12.47 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.5866003036499023
rel_dist={8: [-1.7090580688748975, 1.709057719074428]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4907172, upper bound: 1.4966900
time: 7.25 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969257, upper bound: 1.4969280
time: 5.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.02 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.02
Output dim: 8, lower bound: -1.4907172, upper bound: 1.4966900
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.02
Output dim: 8, lower bound: -1.4969257, upper bound: 1.4969280

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.4049234, -5.7630568, -8.4327374, -5.7614164, -2.5466509, 2.5725741
1: -10.8623381, -7.8408966, -10.8693628, -7.8384266, -2.7060146, 2.7080655
2: -5.0560732, -2.3639569, -5.0749621, -2.3601313, -2.3791418, 2.3944645
3: -6.1155438, -2.8880143, -6.1237812, -2.8838708, -3.2316730, 3.2357669
4: -13.4632463, -9.8448553, -13.4644661, -9.8311605, -2.7458174, 2.7297339
5: -3.5583673, -1.5155169, -3.5617061, -1.4885896, -1.5519714, 1.5290556
6: -10.8766012, -8.0758047, -10.8797121, -8.0382099, -2.3408327, 2.3040793
7: -9.6411133, -6.2792544, -9.6568527, -6.2764831, -3.3646302, 3.3775983
8: 9.2957544, 11.9515572, 9.2831163, 11.9534569, -2.4188805, 2.4288917
9: -7.8644161, -4.4363446, -7.8692684, -4.4315672, -2.9517832, 2.9539819

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709398, upper bound: 1.4966427
time: 7.25 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4907031, upper bound: 1.4966756
time: 5.95 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.4493856, -5.7169747, -8.4408207, -5.7609477, -2.5843654, 2.5996635
1: -10.8846550, -7.8223615, -10.8713779, -7.8377476, -2.7333517, 2.7625532
2: -5.0841165, -2.3185945, -5.0804234, -2.3590331, -2.4072027, 2.4391062
3: -6.1548944, -2.8795891, -6.1261230, -2.8826230, -3.2722714, 3.2465339
4: -13.4879475, -9.8166351, -13.4648170, -9.8270836, -2.7758040, 2.7643149
5: -3.6051393, -1.4780251, -3.5627644, -1.4808145, -1.5864614, 1.5589623
6: -10.9474392, -8.0077410, -10.8806124, -8.0273037, -2.3887944, 2.3767140
7: -9.6843891, -6.2419243, -9.6614246, -6.2756395, -3.4087496, 3.4195004
8: 9.2723780, 11.9705076, 9.2794514, 11.9540062, -2.4399595, 2.4524822
9: -7.9090972, -4.4245591, -7.8706474, -4.4301863, -3.0133805, 2.9690371

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771477, upper bound: 1.4968807
time: 6.23 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969111, upper bound: 1.4969121
time: 5.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.41 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 24.41
Output dim: 8, lower bound: -1.4709398, upper bound: 1.4966427
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 24.41
Output dim: 8, lower bound: -1.4907031, upper bound: 1.4966756
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 24.41
Output dim: 8, lower bound: -1.4771477, upper bound: 1.4968807
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 24.41
Output dim: 8, lower bound: -1.4969111, upper bound: 1.4969121

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -8.3664322, -5.7689085, -8.4267416, -5.7617984, -2.5072594, 2.5549464
1: -10.8445702, -7.9255042, -10.8679171, -7.8515415, -2.6674280, 2.6204195
2: -5.0401192, -2.3712873, -5.0724535, -2.3609877, -2.3655958, 2.3792582
3: -6.0700145, -2.8957264, -6.1168833, -2.8843808, -3.1856337, 3.2211568
4: -13.4615726, -9.8550158, -13.4643335, -9.8325882, -2.7301478, 2.7186596
5: -3.5512605, -1.5657642, -3.5613437, -1.4964254, -1.5242655, 1.4787514
6: -10.8628483, -8.1596642, -10.8788376, -8.0511284, -2.3021278, 2.2234428
7: -9.6205683, -6.2868371, -9.6539087, -6.2773132, -3.3432550, 3.3670716
8: 9.3645554, 11.9426413, 9.2938471, 11.9530602, -2.3490152, 2.4080830
9: -7.8520880, -4.4402094, -7.8678780, -4.4322872, -2.9352179, 2.9474206

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4642542, upper bound: 1.4963764
time: 5.03 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709290, upper bound: 1.4966322
time: 5.99 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -8.4049196, -5.7630558, -8.4327374, -5.7614164, -2.5304298, 2.5725746
1: -10.8623362, -7.8408990, -10.8693628, -7.8384266, -2.7060146, 2.6453424
2: -5.0560713, -2.3639569, -5.0749621, -2.3601313, -2.3791418, 2.3956409
3: -6.1155434, -2.8880131, -6.1237812, -2.8838708, -3.2316725, 3.2357681
4: -13.4632463, -9.8448572, -13.4644661, -9.8311605, -2.7628033, 2.7283764
5: -3.5583677, -1.5155183, -3.5617061, -1.4885896, -1.5519714, 1.5016603
6: -10.8766012, -8.0758085, -10.8797121, -8.0382099, -2.3408327, 2.2430027
7: -9.6411123, -6.2792568, -9.6568527, -6.2764831, -3.3646293, 3.3775959
8: 9.2957573, 11.9515562, 9.2831163, 11.9534569, -2.3767014, 2.4288912
9: -7.8644152, -4.4363451, -7.8692684, -4.4315672, -2.9667921, 2.9529886

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904349, upper bound: 1.4899903
time: 7.15 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906926, upper bound: 1.4966651
time: 6.10 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -8.4104795, -5.7227783, -8.4348125, -5.7613297, -2.5444651, 2.5708523
1: -10.8670712, -7.9061632, -10.8699389, -7.8508558, -2.6902122, 2.6756620
2: -5.0678644, -2.3257790, -5.0779109, -2.3598828, -2.3941197, 2.4177723
3: -6.1102223, -2.8873539, -6.1192312, -2.8831325, -3.2270899, 3.2318773
4: -13.4863272, -9.8274193, -13.4646826, -9.8285618, -2.7602305, 2.7535720
5: -3.5981367, -1.5283635, -3.5624053, -1.4886547, -1.5425466, 1.5088756
6: -10.9338818, -8.0919352, -10.8797407, -8.0402451, -2.3249826, 2.2970655
7: -9.6640568, -6.2492352, -9.6584768, -6.2764673, -3.3875895, 3.4092417
8: 9.3411732, 11.9617395, 9.2901850, 11.9536123, -2.3700628, 2.4185190
9: -7.8970065, -4.4283128, -7.8692703, -4.4309025, -2.9965353, 2.9625592

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4704621, upper bound: 1.4966105
time: 5.88 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771369, upper bound: 1.4968681
time: 6.17 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -8.4493837, -5.7169771, -8.4408207, -5.7609477, -2.5681400, 2.5963972
1: -10.8846531, -7.8223658, -10.8713779, -7.8377476, -2.7333498, 2.6998110
2: -5.0841136, -2.3185945, -5.0804234, -2.3590331, -2.4072027, 2.4389064
3: -6.1548924, -2.8795891, -6.1261230, -2.8826230, -3.2722695, 3.2465339
4: -13.4879465, -9.8166361, -13.4648170, -9.8270836, -2.7928734, 2.7629929
5: -3.6051388, -1.4780277, -3.5627644, -1.4808145, -1.5819737, 1.5315480
6: -10.9474392, -8.0077477, -10.8806124, -8.0273037, -2.3808818, 2.3163638
7: -9.6843872, -6.2419271, -9.6614246, -6.2756395, -3.4087477, 3.4194975
8: 9.2723818, 11.9705086, 9.2794514, 11.9540062, -2.3977823, 2.4524817
9: -7.9090977, -4.4245586, -7.8706474, -4.4301863, -3.0244579, 2.9680448

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966429, upper bound: 1.4902261
time: 7.36 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969006, upper bound: 1.4969007
time: 7.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.00 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 27.00
Output dim: 8, lower bound: -1.4642542, upper bound: 1.4963764
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 27.00
Output dim: 8, lower bound: -1.4709290, upper bound: 1.4966322
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 27.00
Output dim: 8, lower bound: -1.4904349, upper bound: 1.4899903
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 27.00
Output dim: 8, lower bound: -1.4906926, upper bound: 1.4966651
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 27.00
Output dim: 8, lower bound: -1.4704621, upper bound: 1.4966105
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 27.00
Output dim: 8, lower bound: -1.4771369, upper bound: 1.4968681
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 27.00
Output dim: 8, lower bound: -1.4966429, upper bound: 1.4902261
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 27.00
Output dim: 8, lower bound: -1.4969006, upper bound: 1.4969007

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -8.3658371, -5.7704005, -8.4238081, -5.7695246, -2.4967556, 2.5572343
1: -10.8439913, -7.9266872, -10.8649988, -7.8575516, -2.6598415, 2.6039104
2: -5.0398207, -2.3728189, -5.0709677, -2.3686013, -2.3578444, 2.3766842
3: -6.0663199, -2.8965125, -6.0980120, -2.8884392, -3.1778808, 3.2014995
4: -13.4558649, -9.8561974, -13.4347725, -9.8388376, -2.7185068, 2.6864014
5: -3.5511761, -1.5677879, -3.5609176, -1.5065627, -1.5064636, 1.4725327
6: -10.8622437, -8.1633902, -10.8757753, -8.0699310, -2.2810185, 2.2179182
7: -9.6121264, -6.2874775, -9.6105814, -6.2806149, -3.3315115, 3.3231039
8: 9.3651342, 11.9409513, 9.2967691, 11.9443932, -2.3372273, 2.4009681
9: -7.8479800, -4.4405437, -7.8476362, -4.4339647, -2.9225512, 2.9120340

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4593645, upper bound: 1.4963662
time: 6.47 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4642460, upper bound: 1.4963662
time: 5.96 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.3664312, -5.7689085, -8.4370346, -5.7609463, -2.5078955, 2.5575826
1: -10.8445702, -7.9255090, -10.8738012, -7.8483896, -2.6699719, 2.6272202
2: -5.0401182, -2.3712916, -5.0844784, -2.3587809, -2.3665414, 2.3914409
3: -6.0700078, -2.8957283, -6.1228514, -2.8672285, -3.2027793, 3.2271230
4: -13.4615536, -9.8550186, -13.4654207, -9.7915087, -2.7557433, 2.7145522
5: -3.5512586, -1.5657660, -3.5739594, -1.4955751, -1.5232676, 1.4891181
6: -10.8628492, -8.1596718, -10.8917084, -8.0460091, -2.3057532, 2.2364278
7: -9.6205597, -6.2868395, -9.6630449, -6.2302885, -3.3902712, 3.3762054
8: 9.3645554, 11.9426355, 9.2786713, 11.9545794, -2.3515263, 2.4104280
9: -7.8520823, -4.4402089, -7.8734398, -4.4207792, -2.9398937, 2.9544172

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4660394, upper bound: 1.4966239
time: 7.02 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709208, upper bound: 1.4966241
time: 8.92 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -8.4019680, -5.7707820, -8.4321556, -5.7629089, -2.5340996, 2.5620975
1: -10.8594112, -7.8468833, -10.8687935, -7.8395967, -2.6895151, 2.6379490
2: -5.0545840, -2.3715644, -5.0746679, -2.3616672, -2.3765707, 2.3878970
3: -6.0967255, -2.8920732, -6.1201072, -2.8846562, -3.2120693, 3.2280340
4: -13.4336853, -9.8512449, -13.4587545, -9.8323631, -2.7305360, 2.7167387
5: -3.5579381, -1.5256581, -3.5616248, -1.4906125, -1.5457549, 1.4834881
6: -10.8735323, -8.0947075, -10.8791113, -8.0418491, -2.3353281, 2.2216930
7: -9.5977039, -6.2825499, -9.6484566, -6.2771225, -3.3205814, 3.3659067
8: 9.2986813, 11.9428930, 9.2836924, 11.9517670, -2.3695836, 2.4171019
9: -7.8441973, -4.4380426, -7.8651366, -4.4318933, -2.9315734, 2.9402742

Time for backsubstitution: 12.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4855452, upper bound: 1.4899820
time: 6.24 seconds

## Relational analysis of IS_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904267, upper bound: 1.4899841
time: 6.82 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -8.4151926, -5.7622004, -8.4327374, -5.7614188, -2.5397282, 2.5732164
1: -10.8682213, -7.8377123, -10.8693619, -7.8384328, -2.7128344, 2.6480179
2: -5.0680947, -2.3617673, -5.0749612, -2.3601360, -2.3913250, 2.3965740
3: -6.1215801, -2.8708596, -6.1237731, -2.8838727, -3.2377074, 3.2529135
4: -13.4643354, -9.8038139, -13.4644470, -9.8311625, -2.7586803, 2.7556152
5: -3.5709841, -1.5146680, -3.5617065, -1.4885926, -1.5618244, 1.5045164
6: -10.8894711, -8.0706882, -10.8797112, -8.0382166, -2.3538241, 2.2464948
7: -9.6502619, -6.2322774, -9.6568432, -6.2764840, -3.3737779, 3.4245658
8: 9.2805843, 11.9530716, 9.2831173, 11.9534502, -2.3896627, 2.4314079
9: -7.8700323, -4.4248352, -7.8692608, -4.4315667, -2.9737892, 2.9577041

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A1_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4858031, upper bound: 1.4899816
time: 7.01 seconds

## Relational analysis of IS_A1_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906845, upper bound: 1.4966572
time: 4.94 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.4099112, -5.7242727, -8.4318867, -5.7690554, -2.5339890, 2.5731199
1: -10.8664961, -7.9073639, -10.8670244, -7.8568702, -2.6826067, 2.6591339
2: -5.0675697, -2.3273129, -5.0764241, -2.3675008, -2.3863654, 2.4147620
3: -6.1064844, -2.8881392, -6.1003480, -2.8871934, -3.2192910, 3.2122087
4: -13.4806156, -9.8286037, -13.4351225, -9.8347473, -2.7475905, 2.7213089
5: -3.5980549, -1.5303876, -3.5619788, -1.4987879, -1.5247229, 1.5026550
6: -10.9332790, -8.0956230, -10.8766794, -8.0590076, -2.3038726, 2.2914703
7: -9.6556482, -6.2498765, -9.6151772, -6.2797704, -3.3758779, 3.3653007
8: 9.3417501, 11.9600506, 9.2931080, 11.9449444, -2.3582778, 2.4102921
9: -7.8928423, -4.4286404, -7.8490143, -4.4325709, -2.9825397, 2.9271336

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4655724, upper bound: 1.4966022
time: 9.40 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4704539, upper bound: 1.4966043
time: 5.43 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.4104805, -5.7227788, -8.4451132, -5.7604790, -2.5451026, 2.5734997
1: -10.8670692, -7.9061675, -10.8758211, -7.8477135, -2.6927471, 2.6824536
2: -5.0678630, -2.3257833, -5.0899358, -2.3576746, -2.3950696, 2.4207859
3: -6.1102152, -2.8873549, -6.1251764, -2.8659854, -3.2442298, 3.2378216
4: -13.4863081, -9.8274212, -13.4657717, -9.7874269, -2.7669568, 2.7494688
5: -3.5981364, -1.5283656, -3.5750179, -1.4878019, -1.5415469, 1.5192404
6: -10.9338789, -8.0919456, -10.8926039, -8.0351248, -2.3285992, 2.3100483
7: -9.6640463, -6.2492361, -9.6676064, -6.2294426, -3.4346037, 3.4183702
8: 9.3411751, 11.9617329, 9.2750101, 11.9551296, -2.3725753, 2.4194443
9: -7.8969989, -4.4283123, -7.8748078, -4.4193954, -2.9911237, 2.9695578

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6137

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771288, upper bound: 1.4919805
time: 6.91 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771288, upper bound: 1.4968599
time: 6.66 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -8.4465628, -5.7247038, -8.4402409, -5.7624431, -2.5719333, 2.5859845
1: -10.8817387, -7.8284540, -10.8708076, -7.8389168, -2.7168646, 2.6923156
2: -5.0826397, -2.3262215, -5.0801306, -2.3605695, -2.4046183, 2.4311788
3: -6.1358504, -2.8836489, -6.1224504, -2.8834100, -3.2524405, 3.2388015
4: -13.4583855, -9.8228359, -13.4591045, -9.8282852, -2.7606020, 2.7513299
5: -3.6047149, -1.4881693, -3.5626829, -1.4828359, -1.5746955, 1.5134146
6: -10.9443874, -8.0265617, -10.8800097, -8.0309381, -2.3739214, 2.2949338
7: -9.6411295, -6.2452288, -9.6530361, -6.2762814, -3.3648481, 3.4078074
8: 9.2752934, 11.9618397, 9.2800283, 11.9523125, -2.3906622, 2.4406977
9: -7.8885584, -4.4262228, -7.8665104, -4.4305110, -2.9890575, 2.9553766

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4917532, upper bound: 1.4902201
time: 6.47 seconds

## Relational analysis of IS_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966347, upper bound: 1.4902201
time: 5.37 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -8.4596844, -5.7161274, -8.4408197, -5.7609510, -2.5774813, 2.5970535
1: -10.8905582, -7.8191738, -10.8713760, -7.8377519, -2.7401891, 2.7024994
2: -5.0961323, -2.3163743, -5.0804229, -2.3590374, -2.4193754, 2.4398789
3: -6.1609497, -2.8624473, -6.1261148, -2.8826265, -3.2783232, 3.2636676
4: -13.4890375, -9.7755966, -13.4647961, -9.8270864, -2.7887516, 2.7842350
5: -3.6177506, -1.4771750, -3.5627644, -1.4808167, -1.5801282, 1.5343978
6: -10.9603024, -8.0023937, -10.8806105, -8.0273132, -2.3822501, 2.3199015
7: -9.6937551, -6.1949091, -9.6614170, -6.2756433, -3.4181118, 3.4665079
8: 9.2572117, 11.9720268, 9.2794542, 11.9539986, -2.4107351, 2.4549937
9: -7.9147372, -4.4129977, -7.8706388, -4.4301853, -3.0274038, 2.9728546

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A2_A2_A2_A1

### Relational analysis result of IS_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4920111, upper bound: 1.4968950
time: 5.28 seconds

## Relational analysis of IS_A2_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4968924, upper bound: 1.4968929
time: 6.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.46 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4593645, upper bound: 1.4963662
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4642460, upper bound: 1.4963662
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4660394, upper bound: 1.4966239
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4709208, upper bound: 1.4966241
IS_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4855452, upper bound: 1.4899820
IS_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4904267, upper bound: 1.4899841
IS_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4858031, upper bound: 1.4899816
IS_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4906845, upper bound: 1.4966572
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4655724, upper bound: 1.4966022
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4704539, upper bound: 1.4966043
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4771288, upper bound: 1.4919805
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4771288, upper bound: 1.4968599
IS_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4917532, upper bound: 1.4902201
IS_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4966347, upper bound: 1.4902201
IS_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4920111, upper bound: 1.4968950
IS_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.46
Output dim: 8, lower bound: -1.4968924, upper bound: 1.4968929

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.3405790, -5.7852335, -8.4187565, -5.7717113, -2.4705567, 2.5279665
1: -10.8239994, -7.9678273, -10.8634338, -7.8670416, -2.6125360, 2.5594525
2: -4.9834967, -2.3935089, -5.0582132, -2.3703871, -2.2934089, 2.3357460
3: -5.9501181, -2.9273794, -6.0716791, -2.8901551, -3.0599630, 3.1401253
4: -13.4362516, -9.9265614, -13.4335899, -9.8550072, -2.6752737, 2.6135967
5: -3.5425355, -1.5986165, -3.5603540, -1.5133363, -1.4777355, 1.4396873
6: -10.8503599, -8.2117491, -10.8751287, -8.0805988, -2.2421052, 2.1684430
7: -9.5376368, -6.3180237, -9.5944433, -6.2840257, -3.2536111, 3.2764196
8: 9.3889780, 11.9345226, 9.3019180, 11.9440594, -2.3079462, 2.3787289
9: -7.8278360, -4.4762411, -7.8451018, -4.4420061, -2.8917551, 2.8727665

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4593645, upper bound: 1.4903965
time: 4.96 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4593644, upper bound: 1.4963664
time: 6.59 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.3658304, -5.7704024, -8.4238081, -5.7695246, -2.4967403, 2.5838549
1: -10.8439913, -7.9266973, -10.8649988, -7.8575516, -2.6497345, 2.5787220
2: -5.0398102, -2.3728199, -5.0709677, -2.3686013, -2.3311677, 2.3766828
3: -6.0663042, -2.8965149, -6.0980120, -2.8884392, -3.1778650, 3.2014971
4: -13.4558620, -9.8562098, -13.4347725, -9.8388376, -2.7185068, 2.6426296
5: -3.5511758, -1.5677967, -3.5609176, -1.5065627, -1.5011889, 1.4638956
6: -10.8622446, -8.1634045, -10.8757753, -8.0699310, -2.2731364, 2.2042272
7: -9.6121168, -6.2874799, -9.6105814, -6.2806149, -3.3315020, 3.3231015
8: 9.3651381, 11.9409533, 9.2967691, 11.9443932, -2.3303289, 2.3970370
9: -7.8479772, -4.4405503, -7.8476362, -4.4339647, -2.9225483, 2.8980203

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4642460, upper bound: 1.4903944
time: 6.50 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4642460, upper bound: 1.4963665
time: 6.02 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.3411770, -5.7837400, -8.4319897, -5.7631364, -2.4816995, 2.5283992
1: -10.8245811, -7.9666567, -10.8722420, -7.8578892, -2.6226549, 2.5827622
2: -4.9837999, -2.3919830, -5.0717392, -2.3605781, -2.3020997, 2.3417997
3: -5.9538035, -2.9265938, -6.0965090, -2.8689373, -3.0848663, 3.1632361
4: -13.4419413, -9.9253826, -13.4642344, -9.8076649, -2.6944697, 2.6417263
5: -3.5426183, -1.5965922, -3.5733964, -1.5023491, -1.4945979, 1.4562764
6: -10.8509626, -8.2080517, -10.8910542, -8.0566912, -2.2668486, 2.1871686
7: -9.5460634, -6.3173828, -9.6468849, -6.2336917, -3.3123717, 3.3295021
8: 9.3884087, 11.9361992, 9.2838154, 11.9542322, -2.3222275, 2.3878608
9: -7.8319197, -4.4759064, -7.8709450, -4.4288101, -2.9091215, 2.9151134

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4660394, upper bound: 1.4906521
time: 6.93 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4660395, upper bound: 1.4966239
time: 6.97 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.3664274, -5.7689109, -8.4370346, -5.7609463, -2.5078802, 2.5842075
1: -10.8445692, -7.9255209, -10.8738012, -7.8483896, -2.6598639, 2.6020312
2: -5.0401087, -2.3712931, -5.0844784, -2.3587809, -2.3398633, 2.3903897
3: -6.0699902, -2.8957314, -6.1228514, -2.8672285, -3.2027617, 3.2271199
4: -13.4615555, -9.8550291, -13.4654207, -9.7915087, -2.7429802, 2.6707807
5: -3.5512586, -1.5657743, -3.5739594, -1.4955751, -1.5179929, 1.4804807
6: -10.8628473, -8.1596842, -10.8917084, -8.0460091, -2.2978711, 2.2227371
7: -9.6205473, -6.2868438, -9.6630449, -6.2302885, -3.3902588, 3.3762012
8: 9.3645620, 11.9426365, 9.2786713, 11.9545794, -2.3445807, 2.4061894
9: -7.8520794, -4.4402161, -7.8734398, -4.4207792, -2.9398937, 2.9404011

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709208, upper bound: 1.4906543
time: 5.25 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4709209, upper bound: 1.4966245
time: 9.64 seconds

## BFS IS instance: IS_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -8.3768206, -5.7856336, -8.4271059, -5.7650986, -2.5079560, 2.5334017
1: -10.8394451, -7.8879561, -10.8672314, -7.8490934, -2.6559157, 2.5935698
2: -4.9983444, -2.3922343, -5.0619164, -2.3634558, -2.3121986, 2.3493676
3: -5.9806948, -2.9229538, -6.0937672, -2.8863688, -3.0943260, 3.1708133
4: -13.4140759, -9.9217701, -13.4575710, -9.8485317, -2.6888475, 2.6439810
5: -3.5492959, -1.5564768, -3.5610616, -1.4973871, -1.5276589, 1.4504843
6: -10.8616724, -8.1428947, -10.8784618, -8.0525284, -2.3120241, 2.1724017
7: -9.5233269, -6.3131166, -9.6323032, -6.2805319, -3.2427950, 3.3191867
8: 9.3225355, 11.9364357, 9.2888470, 11.9514256, -2.3403273, 2.3994370
9: -7.8238077, -4.4737673, -7.8626795, -4.4399338, -2.9011450, 2.9009485

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A2_A1_A1_B1

### Relational analysis result of IS_A1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4855452, upper bound: 1.4840101
time: 6.60 seconds

## Relational analysis of IS_A1_A2_A1_A1_B2

### Relational analysis result of IS_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4855452, upper bound: 1.4899820
time: 6.14 seconds

## BFS IS instance: IS_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -8.4019613, -5.7707834, -8.4321556, -5.7629089, -2.5340862, 2.5965590
1: -10.8594112, -7.8468947, -10.8687935, -7.8395967, -2.6885643, 2.6128197
2: -5.0545740, -2.3715658, -5.0746679, -2.3616672, -2.3498936, 2.3878956
3: -6.0967102, -2.8920734, -6.1201072, -2.8846562, -3.2120540, 3.2280338
4: -13.4336843, -9.8512583, -13.4587545, -9.8323631, -2.7305350, 2.6729589
5: -3.5579367, -1.5256664, -3.5616248, -1.4906125, -1.5457549, 1.4748461
6: -10.8735352, -8.0947208, -10.8791113, -8.0418491, -2.3353267, 2.2080014
7: -9.5976915, -6.2825541, -9.6484566, -6.2771225, -3.3205690, 3.3659024
8: 9.2986851, 11.9428902, 9.2836924, 11.9517670, -2.3626657, 2.4162335
9: -7.8441939, -4.4380493, -7.8651366, -4.4318933, -2.9315696, 2.9262571

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A2_A1_A2_B1

### Relational analysis result of IS_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904267, upper bound: 1.4840122
time: 9.96 seconds

## Relational analysis of IS_A1_A2_A1_A2_B2

### Relational analysis result of IS_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4904267, upper bound: 1.4899841
time: 6.09 seconds

## BFS IS instance: IS_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -8.3900948, -5.7770576, -8.4276867, -5.7636056, -2.5136385, 2.5445113
1: -10.8482790, -7.8788099, -10.8678007, -7.8479338, -2.6792316, 2.6036119
2: -5.0119247, -2.3824749, -5.0622110, -2.3619246, -2.3270087, 2.3580093
3: -6.0055523, -2.9017096, -6.0974312, -2.8855853, -3.1199670, 3.1957216
4: -13.4447212, -9.8744774, -13.4632607, -9.8473330, -2.7169218, 2.6826818
5: -3.5623422, -1.5454817, -3.5611432, -1.4953671, -1.5331535, 1.4717114
6: -10.8776140, -8.1189060, -10.8790598, -8.0488968, -2.3204417, 2.1971850
7: -9.5758400, -6.2628021, -9.6406832, -6.2798905, -3.2959495, 3.3778811
8: 9.3044214, 11.9465771, 9.2882729, 11.9531116, -2.3604088, 2.4137363
9: -7.8496647, -4.4605064, -7.8668032, -4.4396081, -2.9429169, 2.9184499

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A2_A2_A1_B1

### Relational analysis result of IS_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4858029, upper bound: 1.4840098
time: 7.59 seconds

## Relational analysis of IS_A1_A2_A2_A1_B2

### Relational analysis result of IS_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4858030, upper bound: 1.4899816
time: 9.02 seconds

## BFS IS instance: IS_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -8.4151869, -5.7622027, -8.4327374, -5.7614188, -2.5397148, 2.6076298
1: -10.8682203, -7.8377247, -10.8693619, -7.8384328, -2.7118816, 2.6229367
2: -5.0680847, -2.3617692, -5.0749612, -2.3601360, -2.3646474, 2.3965731
3: -6.1215644, -2.8708615, -6.1237731, -2.8838727, -3.2376916, 3.2529116
4: -13.4643354, -9.8038282, -13.4644470, -9.8311625, -2.7586803, 2.7117941
5: -3.5709834, -1.5146768, -3.5617065, -1.4885926, -1.5565463, 1.4958787
6: -10.8894730, -8.0707006, -10.8797112, -8.0382166, -2.3514605, 2.2328036
7: -9.6502504, -6.2322774, -9.6568432, -6.2764840, -3.3737664, 3.4245658
8: 9.2805901, 11.9530735, 9.2831173, 11.9534502, -2.3827128, 2.4305487
9: -7.8700290, -4.4248390, -7.8692608, -4.4315667, -2.9737873, 2.9436846

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A2_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906844, upper bound: 1.4906854
time: 4.83 seconds

## Relational analysis of IS_A1_A2_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906845, upper bound: 1.4966572
time: 4.96 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.3847399, -5.7390881, -8.4268332, -5.7712421, -2.5078473, 2.5438681
1: -10.8465519, -7.9485149, -10.8654604, -7.8663626, -2.6352415, 2.6146650
2: -5.0112967, -2.3479795, -5.0636702, -2.3692856, -2.3219366, 2.3554590
3: -5.9902792, -2.9190128, -6.0740175, -2.8889081, -3.1013710, 3.1529703
4: -13.4609985, -9.8991575, -13.4339361, -9.8509150, -2.6864834, 2.6484528
5: -3.5894175, -1.5612029, -3.5614157, -1.5055640, -1.4959862, 1.4698298
6: -10.9214125, -8.1439257, -10.8760309, -8.0696793, -2.2649579, 2.2420068
7: -9.5812731, -6.2804451, -9.5990372, -6.2831826, -3.2980905, 3.3185921
8: 9.3656826, 11.9535809, 9.2982588, 11.9446087, -2.3288941, 2.3877046
9: -7.8726830, -4.4642448, -7.8464794, -4.4406118, -2.9395542, 2.8879867

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_A1_B1_A1_A1

### Relational analysis result of IS_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4655724, upper bound: 1.4901861
time: 5.72 seconds

## Relational analysis of IS_A2_A1_B1_A1_A2

### Relational analysis result of IS_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4655724, upper bound: 1.4966022
time: 10.10 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.4099064, -5.7242732, -8.4318867, -5.7690554, -2.5339737, 2.5997419
1: -10.8664951, -7.9073734, -10.8670244, -7.8568702, -2.6724997, 2.6339769
2: -5.0675597, -2.3273149, -5.0764241, -2.3675008, -2.3596878, 2.4040861
3: -6.1064663, -2.8881402, -6.1003480, -2.8871934, -3.2192729, 3.2122078
4: -13.4806156, -9.8286152, -13.4351225, -9.8347473, -2.7348270, 2.6775365
5: -3.5980539, -1.5303954, -3.5619788, -1.4987879, -1.5194463, 1.4940181
6: -10.9332790, -8.0956345, -10.8766794, -8.0590076, -2.2959881, 2.2777796
7: -9.6556358, -6.2498794, -9.6151772, -6.2797704, -3.3758655, 3.3652978
8: 9.3417549, 11.9600487, 9.2931080, 11.9449444, -2.3513670, 2.4060535
9: -7.8928399, -4.4286470, -7.8490143, -4.4325709, -2.9752812, 2.9131188

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_A1_B1_A2_A1

### Relational analysis result of IS_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4642460, upper bound: 1.4901842
time: 6.61 seconds

## Relational analysis of IS_A2_A1_B1_A2_A2

### Relational analysis result of IS_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4704539, upper bound: 1.4966043
time: 5.38 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -8.4054279, -5.7249660, -8.4200439, -5.7753215, -2.5163975, 2.5473835
1: -10.8655033, -7.9156685, -10.8559322, -7.8888221, -2.6481123, 2.6409140
2: -5.0551252, -2.3275757, -5.0337648, -2.3783512, -2.3565545, 2.3562288
3: -6.0838747, -2.8890707, -6.0091009, -2.8968337, -3.1870410, 3.1200302
4: -13.4851217, -9.8435898, -13.4461594, -9.8579550, -2.6940398, 2.7039268
5: -3.5975745, -1.5351344, -3.5663781, -1.5186272, -1.5086615, 1.4969050
6: -10.9332275, -8.1026335, -10.8807535, -8.0833158, -2.2792397, 2.2798905
7: -9.6479549, -6.2526383, -9.5931492, -6.2599978, -3.3879571, 3.3405108
8: 9.3463297, 11.9613876, 9.2989273, 11.9486380, -2.3548803, 2.3893497
9: -7.8945503, -4.4363384, -7.8544178, -4.4550567, -2.9516993, 2.9387159

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771288, upper bound: 1.4722500
time: 7.05 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771288, upper bound: 1.4919805
time: 6.89 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.4104805, -5.7227788, -8.4451094, -5.7604809, -2.5797515, 2.5734987
1: -10.8670692, -7.9061675, -10.8758202, -7.8477240, -2.6654539, 2.6783533
2: -5.0678630, -2.3257833, -5.0899262, -2.3576751, -2.3950686, 2.3940728
3: -6.1102152, -2.8873549, -6.1251607, -2.8659863, -3.2442288, 3.2289042
4: -13.4863081, -9.8274212, -13.4657717, -9.7874393, -2.7231362, 2.7494674
5: -3.5981364, -1.5283656, -3.5750184, -1.4878091, -1.5329006, 1.5192394
6: -10.9338789, -8.0919456, -10.8926039, -8.0351353, -2.3148863, 2.3100474
7: -9.6640463, -6.2492361, -9.6675949, -6.2294455, -3.4346008, 3.4183588
8: 9.3411751, 11.9617329, 9.2750158, 11.9551296, -2.3717179, 2.4112892
9: -7.8969989, -4.4283123, -7.8748055, -4.4193997, -2.9770994, 2.9695549

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771288, upper bound: 1.4771296
time: 7.03 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4771288, upper bound: 1.4968599
time: 6.73 seconds

## BFS IS instance: IS_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -8.4214983, -5.7395363, -8.4351912, -5.7646294, -2.5459228, 2.5567715
1: -10.8618202, -7.8695431, -10.8692493, -7.8484144, -2.6832447, 2.6479225
2: -5.0264492, -2.3468671, -5.0673780, -2.3623548, -2.3402538, 2.3719163
3: -6.0198088, -2.9145374, -6.0961103, -2.8851225, -3.1346862, 3.1815729
4: -13.4387674, -9.8935270, -13.4579201, -9.8444576, -2.7001143, 2.6784713
5: -3.5960763, -1.5189524, -3.5621197, -1.4896123, -1.5460411, 1.4804089
6: -10.9325476, -8.0747328, -10.8793640, -8.0416126, -2.3350284, 2.2456582
7: -9.5668640, -6.2758183, -9.6368809, -6.2796879, -3.2871761, 3.3610625
8: 9.2992296, 11.9553490, 9.2851887, 11.9519749, -2.3613071, 2.4230065
9: -7.8682013, -4.4618568, -7.8640552, -4.4385500, -2.9464421, 2.9161649

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A2_A2_A1_A1_B1

### Relational analysis result of IS_A2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4853351, upper bound: 1.4902184
time: 5.95 seconds

## Relational analysis of IS_A2_A2_A1_A1_B2

### Relational analysis result of IS_A2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4853351, upper bound: 1.4902176
time: 7.10 seconds

## BFS IS instance: IS_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -8.4465570, -5.7247057, -8.4402409, -5.7624431, -2.5719180, 2.6126089
1: -10.8817387, -7.8284645, -10.8708076, -7.8389168, -2.7159204, 2.6672163
2: -5.0826306, -2.3262215, -5.0801306, -2.3605695, -2.3779416, 2.4205031
3: -6.1358356, -2.8836501, -6.1224504, -2.8834100, -3.2524257, 3.2388003
4: -13.4583807, -9.8228474, -13.4591045, -9.8282852, -2.7488141, 2.7075431
5: -3.6047153, -1.4881775, -3.5626829, -1.4828359, -1.5694164, 1.5047722
6: -10.9443855, -8.0265713, -10.8800097, -8.0309381, -2.3660364, 2.2812419
7: -9.6411190, -6.2452316, -9.6530361, -6.2762814, -3.3648376, 3.4078045
8: 9.2752981, 11.9618397, 9.2800283, 11.9523125, -2.3837318, 2.4398394
9: -7.8885603, -4.4262309, -7.8665104, -4.4305110, -2.9817772, 2.9413586

Time for backsubstitution: 12.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A2_A2_A1_A2_B1

### Relational analysis result of IS_A2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4902165, upper bound: 1.4902201
time: 5.61 seconds

## Relational analysis of IS_A2_A2_A1_A2_B2

### Relational analysis result of IS_A2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4902165, upper bound: 1.4902180
time: 7.78 seconds

## BFS IS instance: IS_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -8.4346657, -5.7309699, -8.4357700, -5.7631392, -2.5515184, 2.5678368
1: -10.8706627, -7.8602829, -10.8698215, -7.8472505, -2.7065644, 2.6580791
2: -5.0400109, -2.3370619, -5.0676713, -2.3608260, -2.3550653, 2.3805835
3: -6.0449142, -2.8933043, -6.0997758, -2.8843379, -3.1605763, 3.2064714
4: -13.4694204, -9.8462305, -13.4636106, -9.8432550, -2.7281880, 2.7112577
5: -3.6091125, -1.5079529, -3.5622017, -1.4875900, -1.5514607, 1.5016108
6: -10.9484653, -8.0505981, -10.8799601, -8.0379887, -2.3433475, 2.2706079
7: -9.6194420, -6.2254539, -9.6452560, -6.2790484, -3.3403935, 3.4198022
8: 9.2811337, 11.9654980, 9.2846146, 11.9536572, -2.3813810, 2.4372950
9: -7.8943644, -4.4485779, -7.8681803, -4.4382243, -2.9841690, 2.9336967

Time for backsubstitution: 12.52 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.435220718383789
rel_dist={8: [-1.4969366874007388, 1.496938647628781]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3793761, upper bound: 1.3844870
time: 8.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847595, upper bound: 1.3847602
time: 4.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.02 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.02
Output dim: 8, lower bound: -1.3793761, upper bound: 1.3844870
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.02
Output dim: 8, lower bound: -1.3847595, upper bound: 1.3847602

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.4049234, -5.7630568, -8.4299822, -5.7615762, -2.4608355, 2.4841595
1: -10.8623381, -7.8408966, -10.8686724, -7.8386655, -2.6280470, 2.6298833
2: -5.0560732, -2.3639569, -5.0730987, -2.3605084, -2.2959223, 2.3097391
3: -6.1155438, -2.8880143, -6.1229763, -2.8842952, -3.1821756, 3.1857524
4: -13.4632463, -9.8448553, -13.4643488, -9.8325472, -2.6223531, 2.6078606
5: -3.5583673, -1.5155169, -3.5613351, -1.4912455, -1.4861414, 1.4654243
6: -10.8766012, -8.0758047, -10.8794069, -8.0419302, -2.2440157, 2.2108784
7: -9.6411133, -6.2792544, -9.6552916, -6.2767730, -3.3643403, 3.3760371
8: 9.2957544, 11.9515572, 9.2843657, 11.9532700, -2.3429780, 2.3519969
9: -7.8644161, -4.4363446, -7.8687983, -4.4320426, -2.8716116, 2.8736162

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3624201, upper bound: 1.3844514
time: 6.99 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3793649, upper bound: 1.3844730
time: 11.80 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.4493856, -5.7169747, -8.4408150, -5.7609482, -2.4966607, 2.5133352
1: -10.8846550, -7.8223615, -10.8713779, -7.8377481, -2.6568570, 2.6845570
2: -5.0841165, -2.3185945, -5.0804219, -2.3590331, -2.3222222, 2.3546326
3: -6.1548944, -2.8795891, -6.1261225, -2.8826230, -3.2365446, 3.1993351
4: -13.4879475, -9.8166351, -13.4648151, -9.8270855, -2.6540351, 2.6410170
5: -3.6051393, -1.4780251, -3.5627646, -1.4808174, -1.5218022, 1.4927015
6: -10.9474392, -8.0077410, -10.8806114, -8.0273066, -2.2930889, 2.2795291
7: -9.6843891, -6.2419243, -9.6614237, -6.2756414, -3.4087477, 3.4194994
8: 9.2723780, 11.9705076, 9.2794552, 11.9540043, -2.3635302, 2.3767915
9: -7.9090972, -4.4245591, -7.8706441, -4.4301863, -2.9326262, 2.8899393

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5736

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3678076, upper bound: 1.3847263
time: 5.86 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847477, upper bound: 1.3847484
time: 6.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.00 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 25.00
Output dim: 8, lower bound: -1.3624201, upper bound: 1.3844514
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 25.00
Output dim: 8, lower bound: -1.3793649, upper bound: 1.3844730
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 25.00
Output dim: 8, lower bound: -1.3678076, upper bound: 1.3847263
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 25.00
Output dim: 8, lower bound: -1.3847477, upper bound: 1.3847484

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -8.3664322, -5.7689085, -8.4228020, -5.7620330, -2.4213395, 2.4636135
1: -10.8445702, -7.9255042, -10.8669376, -7.8543873, -2.5833406, 2.5417981
2: -5.0401192, -2.3712873, -5.0700927, -2.3615355, -2.2821531, 2.2938304
3: -6.0700145, -2.8957264, -6.1147051, -2.8849080, -3.1353331, 3.1597304
4: -13.4615726, -9.8550158, -13.4641876, -9.8342085, -2.6063323, 2.5967584
5: -3.5512605, -1.5657642, -3.5609007, -1.5006435, -1.4539018, 1.4149911
6: -10.8628483, -8.1596642, -10.8783560, -8.0574160, -2.1981745, 2.1300311
7: -9.6205683, -6.2868371, -9.6517696, -6.2777715, -3.3427968, 3.3649325
8: 9.3645554, 11.9426413, 9.2972345, 11.9527922, -2.2730083, 2.3285913
9: -7.8520880, -4.4402094, -7.8671207, -4.4329085, -2.8548899, 2.8667016

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3565722, upper bound: 1.3840862
time: 6.88 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3624104, upper bound: 1.3844435
time: 5.10 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -8.4049196, -5.7630558, -8.4299822, -5.7615762, -2.4436584, 2.4841599
1: -10.8623362, -7.8408990, -10.8686724, -7.8386655, -2.6280460, 2.5634637
2: -5.0560713, -2.3639569, -5.0730987, -2.3605084, -2.2959213, 2.3108754
3: -6.1155434, -2.8880131, -6.1229763, -2.8842952, -3.1821728, 3.1938596
4: -13.4632463, -9.8448572, -13.4643488, -9.8325472, -2.6386271, 2.6065028
5: -3.5583677, -1.5155183, -3.5613351, -1.4912455, -1.4861414, 1.4364166
6: -10.8766012, -8.0758085, -10.8794069, -8.0419302, -2.2440157, 2.1461294
7: -9.6411123, -6.2792568, -9.6552916, -6.2767730, -3.3643394, 3.3760347
8: 9.2957573, 11.9515562, 9.2843657, 11.9532700, -2.2983184, 2.3519959
9: -7.8644152, -4.4363451, -7.8687983, -4.4320426, -2.8860049, 2.8726230

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3790029, upper bound: 1.3786175
time: 8.36 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3793556, upper bound: 1.3844636
time: 7.18 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -8.4104795, -5.7227783, -8.4336138, -5.7614050, -2.4566584, 2.4827306
1: -10.8670712, -7.9061632, -10.8696518, -7.8534660, -2.6078663, 2.5972271
2: -5.0678644, -2.3257790, -5.0774093, -2.3600535, -2.3089213, 2.3325105
3: -6.1102223, -2.8873539, -6.1178565, -2.8832366, -3.1906309, 3.1732202
4: -13.4863272, -9.8274193, -13.4646568, -9.8288641, -2.6381247, 2.6302485
5: -3.5981367, -1.5283635, -3.5623331, -1.4902186, -1.4754622, 1.4424860
6: -10.9338818, -8.0919352, -10.8795643, -8.0428276, -2.2258081, 2.1996725
7: -9.6640568, -6.2492352, -9.6578865, -6.2766352, -3.3874216, 3.4086514
8: 9.3411732, 11.9617395, 9.2923241, 11.9535351, -2.2935352, 2.3389695
9: -7.8970065, -4.4283128, -7.8689919, -4.4310455, -2.9156246, 2.8831205

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6109

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3565723, upper bound: 1.3843627
time: 4.69 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3677979, upper bound: 1.3847167
time: 6.62 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -8.4493837, -5.7169771, -8.4408150, -5.7609482, -2.4794817, 2.5100689
1: -10.8846531, -7.8223658, -10.8713779, -7.8377481, -2.6568551, 2.6181202
2: -5.0841136, -2.3185945, -5.0804219, -2.3590331, -2.3222213, 2.3543925
3: -6.1548924, -2.8795891, -6.1261225, -2.8826230, -3.2365427, 3.2074423
4: -13.4879465, -9.8166361, -13.4648151, -9.8270855, -2.6696701, 2.6396952
5: -3.6051388, -1.4780277, -3.5627646, -1.4808174, -1.5173144, 1.4636779
6: -10.9474392, -8.0077477, -10.8806114, -8.0273066, -2.2851763, 2.2155123
7: -9.6843872, -6.2419271, -9.6614237, -6.2756414, -3.4087458, 3.4194965
8: 9.2723818, 11.9705086, 9.2794552, 11.9540043, -2.3188729, 2.3767910
9: -7.9090977, -4.4245586, -7.8706441, -4.4301863, -2.9430962, 2.8889465

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3843816, upper bound: 1.3788926
time: 8.23 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3793555, upper bound: 1.3847390
time: 6.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.55 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 27.55
Output dim: 8, lower bound: -1.3565722, upper bound: 1.3840862
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 27.55
Output dim: 8, lower bound: -1.3624104, upper bound: 1.3844435
IS_A1_A2_A1, status: Status.VERIFIED, split count: 3, time: 27.55
Output dim: 8, lower bound: -1.3790029, upper bound: 1.3786175
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 27.55
Output dim: 8, lower bound: -1.3793556, upper bound: 1.3844636
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 27.55
Output dim: 8, lower bound: -1.3565723, upper bound: 1.3843627
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 27.55
Output dim: 8, lower bound: -1.3677979, upper bound: 1.3847167
IS_A2_A2_A1, status: Status.VERIFIED, split count: 3, time: 27.55
Output dim: 8, lower bound: -1.3843816, upper bound: 1.3788926
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 27.55
Output dim: 8, lower bound: -1.3793555, upper bound: 1.3847390

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.3664303, -5.7689080, -8.4330883, -5.7611856, -2.4218078, 2.4662437
1: -10.8445683, -7.9255090, -10.8728189, -7.8512335, -2.5858879, 2.5485101
2: -5.0401173, -2.3712921, -5.0820622, -2.3593321, -2.2827029, 2.3059530
3: -6.0700064, -2.8957293, -6.1206808, -2.8677716, -3.1526346, 3.1620364
4: -13.4615536, -9.8550186, -13.4652710, -9.7931347, -2.6298842, 2.5907102
5: -3.5512598, -1.5657670, -3.5735168, -1.4997911, -1.4517703, 1.4253578
6: -10.8628473, -8.1596718, -10.8912239, -8.0522938, -2.2015343, 2.1430168
7: -9.6205587, -6.2868409, -9.6609020, -6.2307472, -3.3898115, 3.3740611
8: 9.3645563, 11.9426346, 9.2820606, 11.9543018, -2.2744703, 2.3295135
9: -7.8520784, -4.4402094, -7.8726921, -4.4213996, -2.8595734, 2.8723154

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3582210, upper bound: 1.3844370
time: 4.91 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3624035, upper bound: 1.3844348
time: 11.31 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -8.4151926, -5.7622004, -8.4299822, -5.7615790, -2.4529591, 2.4846306
1: -10.8682194, -7.8377123, -10.8686714, -7.8386703, -2.6347761, 2.5661402
2: -5.0680389, -2.3617678, -5.0730982, -2.3605137, -2.3080440, 2.3114133
3: -6.1215811, -2.8708730, -6.1229677, -2.8842981, -3.1845407, 3.2111616
4: -13.4643326, -9.8038130, -13.4643250, -9.8325529, -2.6325643, 2.6319907
5: -3.5709834, -1.5146694, -3.5613360, -1.4912481, -1.4938828, 1.4381242
6: -10.8894720, -8.0706882, -10.8794060, -8.0419369, -2.2570066, 2.1493607
7: -9.6502600, -6.2322750, -9.6552801, -6.2767744, -3.3734856, 3.4230051
8: 9.2805843, 11.9530640, 9.2843666, 11.9532623, -2.3112788, 2.3534636
9: -7.8700314, -4.4248371, -7.8687859, -4.4320416, -2.8916178, 2.8773360

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A1_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3751659, upper bound: 1.3844569
time: 6.60 seconds

## Relational analysis of IS_A1_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3793486, upper bound: 1.3844573
time: 6.26 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.4104795, -5.7227807, -8.4439144, -5.7605581, -2.4571247, 2.4853754
1: -10.8670692, -7.9061680, -10.8755302, -7.8503246, -2.6104012, 2.6039276
2: -5.0678635, -2.3257852, -5.0893788, -2.3578434, -2.3094749, 2.3354886
3: -6.1102104, -2.8873551, -6.1238036, -2.8660994, -3.2079277, 3.1755052
4: -13.4863052, -9.8274212, -13.4657412, -9.7877264, -2.6431015, 2.6242049
5: -3.5981367, -1.5283660, -3.5749457, -1.4893665, -1.4733284, 1.4528501
6: -10.9338799, -8.0919456, -10.8924332, -8.0377064, -2.2291574, 2.2126539
7: -9.6640472, -6.2492371, -9.6670132, -6.2296095, -3.4344378, 3.4177761
8: 9.3411751, 11.9617290, 9.2771473, 11.9550428, -2.2949996, 2.3398943
9: -7.8969975, -4.4283137, -7.8745337, -4.4195395, -2.9102139, 2.8887367

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6137

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3677909, upper bound: 1.3805327
time: 13.00 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3677910, upper bound: 1.3847100
time: 6.50 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -8.4596844, -5.7161269, -8.4408169, -5.7609510, -2.4888220, 2.5105546
1: -10.8905544, -7.8191752, -10.8713760, -7.8377538, -2.6636038, 2.6208076
2: -5.0960760, -2.3163738, -5.0804219, -2.3590388, -2.3343334, 2.3549683
3: -6.1609507, -2.8624606, -6.1261153, -2.8826251, -3.2390146, 3.2247362
4: -13.4890327, -9.7755966, -13.4647913, -9.8270874, -2.6636662, 2.6591783
5: -3.6177504, -1.4771755, -3.5627642, -1.4808197, -1.5154681, 1.4653780
6: -10.9603024, -8.0023956, -10.8806105, -8.0273161, -2.2865436, 2.2187879
7: -9.6937523, -6.1949072, -9.6614141, -6.2756414, -3.4181108, 3.4665070
8: 9.2572107, 11.9720211, 9.2794552, 11.9539967, -2.3318248, 2.3782535
9: -7.9147387, -4.4129963, -7.8706393, -4.4301853, -2.9446621, 2.8937540

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6137

## Relational analysis of IS_A2_A2_A2_A1

### Relational analysis result of IS_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3805545, upper bound: 1.3847339
time: 5.30 seconds

## Relational analysis of IS_A2_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847314, upper bound: 1.3847322
time: 5.29 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.30 seconds
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 23.30
Output dim: 8, lower bound: -1.3582210, upper bound: 1.3844370
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 23.30
Output dim: 8, lower bound: -1.3624035, upper bound: 1.3844348
IS_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 23.30
Output dim: 8, lower bound: -1.3751659, upper bound: 1.3844569
IS_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.30
Output dim: 8, lower bound: -1.3793486, upper bound: 1.3844573
IS_A2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 23.30
Output dim: 8, lower bound: -1.3677909, upper bound: 1.3805327
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 23.30
Output dim: 8, lower bound: -1.3677910, upper bound: 1.3847100
IS_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 23.30
Output dim: 8, lower bound: -1.3805545, upper bound: 1.3847339
IS_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.30
Output dim: 8, lower bound: -1.3847314, upper bound: 1.3847322

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.3411770, -5.7837415, -8.4271126, -5.7637987, -2.3953304, 2.4364581
1: -10.8245811, -7.9666572, -10.8709650, -7.8624868, -2.5365005, 2.5037231
2: -4.9838004, -2.3919840, -5.0669723, -2.3614669, -2.2177906, 2.2512138
3: -5.9538031, -2.9265945, -6.0894794, -2.8698001, -3.0287962, 3.0815949
4: -13.4419374, -9.9253826, -13.4638624, -9.8122683, -2.5651805, 2.5176618
5: -3.5426178, -1.5965928, -3.5728462, -1.5078132, -1.4216475, 1.3923106
6: -10.8509617, -8.2080517, -10.8904495, -8.0649433, -2.1604147, 2.0936084
7: -9.5460615, -6.3173828, -9.6417608, -6.2347941, -3.3112674, 3.3243780
8: 9.3884096, 11.9362001, 9.2881432, 11.9538908, -2.2450228, 2.3057196
9: -7.8319178, -4.4759064, -7.8697367, -4.4309101, -2.8272686, 2.8324509

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4671

## Relational analysis of IS_A1_A1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3550093, upper bound: 1.3844321
time: 5.32 seconds

## Relational analysis of IS_A1_A1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3582160, upper bound: 1.3844321
time: 4.62 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.3664255, -5.7689114, -8.4330883, -5.7611856, -2.4217916, 2.4915752
1: -10.8445663, -7.9255214, -10.8728189, -7.8512335, -2.5757790, 2.5208464
2: -5.0401082, -2.3712950, -5.0820622, -2.3593321, -2.2544584, 2.3026786
3: -6.0699892, -2.8957319, -6.1206808, -2.8677716, -3.1343727, 3.1620359
4: -13.4615507, -9.8550310, -13.4652710, -9.7931347, -2.6171212, 2.5443659
5: -3.5512590, -1.5657742, -3.5735168, -1.4997911, -1.4464951, 1.4162138
6: -10.8628483, -8.1596870, -10.8912239, -8.0522938, -2.1936512, 2.1285214
7: -9.6205454, -6.2868419, -9.6609020, -6.2307472, -3.3897982, 3.3740602
8: 9.3645611, 11.9426346, 9.2820606, 11.9543018, -2.2664909, 2.3252754
9: -7.8520756, -4.4402156, -7.8726921, -4.4213996, -2.8595715, 2.8574739

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4671

## Relational analysis of IS_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3591799, upper bound: 1.3844321
time: 5.06 seconds

## Relational analysis of IS_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3623985, upper bound: 1.3844318
time: 4.71 seconds

## BFS IS instance: IS_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -8.3900928, -5.7770591, -8.4239979, -5.7641921, -2.4265909, 2.4548812
1: -10.8482761, -7.8788118, -10.8668194, -7.8499198, -2.5988908, 2.5214038
2: -5.0118690, -2.3824759, -5.0579944, -2.3626390, -2.2432599, 2.2702646
3: -6.0055523, -2.9017217, -6.0917687, -2.8863342, -3.0606394, 3.1278419
4: -13.4447193, -9.8744783, -13.4629183, -9.8517056, -2.5853310, 2.5588558
5: -3.5623422, -1.5454816, -3.5606654, -1.4992708, -1.4637574, 1.4051139
6: -10.8776140, -8.1189060, -10.8786287, -8.0545864, -2.2177036, 2.0999024
7: -9.5758343, -6.2628016, -9.6361446, -6.2808275, -3.2950068, 3.3733430
8: 9.3044214, 11.9465714, 9.2904663, 11.9528580, -2.2818756, 2.3343740
9: -7.8496666, -4.4605069, -7.8658710, -4.4415669, -2.8592062, 2.8375196

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A2_A2_A1_B1

### Relational analysis result of IS_A1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3751659, upper bound: 1.3793503
time: 4.77 seconds

## Relational analysis of IS_A1_A2_A2_A1_B2

### Relational analysis result of IS_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3751659, upper bound: 1.3844582
time: 5.14 seconds

## BFS IS instance: IS_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -8.4151878, -5.7622027, -8.4299822, -5.7615790, -2.4529457, 2.5165734
1: -10.8682156, -7.8377247, -10.8686714, -7.8386703, -2.6338224, 2.5385842
2: -5.0680285, -2.3617702, -5.0730982, -2.3605137, -2.2798004, 2.3114114
3: -6.1215653, -2.8708744, -6.1229677, -2.8842981, -3.1662817, 3.2111611
4: -13.4643307, -9.8038254, -13.4643250, -9.8325529, -2.6325638, 2.5855894
5: -3.5709848, -1.5146755, -3.5613360, -1.4912481, -1.4886042, 1.4289796
6: -10.8894720, -8.0707016, -10.8794060, -8.0419369, -2.2509341, 2.1348650
7: -9.6502485, -6.2322779, -9.6552801, -6.2767744, -3.3734741, 3.4230022
8: 9.2805891, 11.9530659, 9.2843666, 11.9532623, -2.3032937, 2.3526030
9: -7.8700299, -4.4248419, -7.8687859, -4.4320416, -2.8916149, 2.8624930

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A2_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3793485, upper bound: 1.3793489
time: 6.60 seconds

## Relational analysis of IS_A1_A2_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3793486, upper bound: 1.3844571
time: 6.51 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -8.4104795, -5.7227807, -8.4439106, -5.7605577, -2.4904861, 2.4853747
1: -10.8670692, -7.9061680, -10.8755293, -7.8503351, -2.5806265, 2.5977321
2: -5.0678635, -2.3257852, -5.0893679, -2.3578448, -2.3094730, 2.3072033
3: -6.1102104, -2.8873551, -6.1237884, -2.8661020, -3.1969519, 3.1572447
4: -13.4863052, -9.8274212, -13.4657421, -9.7877398, -2.5967007, 2.6242042
5: -3.5981367, -1.5283660, -3.5749457, -1.4893742, -1.4641730, 1.4523540
6: -10.9338799, -8.0919456, -10.8924313, -8.0377188, -2.2146373, 2.2106137
7: -9.6640472, -6.2492371, -9.6670027, -6.2296119, -3.4344354, 3.4177656
8: 9.3411751, 11.9617290, 9.2771549, 11.9550438, -2.2941413, 2.3307014
9: -7.8969975, -4.4283137, -7.8745322, -4.4195437, -2.8953619, 2.8887348

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4671

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3645392, upper bound: 1.3847054
time: 6.57 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3677860, upper bound: 1.3847054
time: 7.10 seconds

## BFS IS instance: IS_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -8.4346638, -5.7309713, -8.4348354, -5.7635646, -2.4625826, 2.4807279
1: -10.8706589, -7.8602839, -10.8695288, -7.8490052, -2.6276970, 2.5760593
2: -5.0399542, -2.3370619, -5.0653176, -2.3611617, -2.2695565, 2.2927971
3: -6.0449128, -2.8933175, -6.0949149, -2.8846629, -3.1151352, 3.1422367
4: -13.4694147, -9.8462324, -13.4633837, -9.8462410, -2.5986090, 2.5860000
5: -3.6091120, -1.5079536, -3.5620942, -1.4888420, -1.4853468, 1.4323869
6: -10.9484644, -8.0505981, -10.8798361, -8.0399618, -2.2454276, 2.1693468
7: -9.6194401, -6.2254543, -9.6422720, -6.2796941, -3.3397460, 3.4168177
8: 9.2811327, 11.9654913, 9.2855616, 11.9535923, -2.3023233, 2.3591380
9: -7.8943653, -4.4485779, -7.8677216, -4.4397087, -2.8997092, 2.8540344

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of IS_A2_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3805502, upper bound: 1.3814786
time: 11.89 seconds

## Relational analysis of IS_A2_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3805499, upper bound: 1.3847266
time: 5.94 seconds

## BFS IS instance: IS_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -8.4596777, -5.7161279, -8.4408169, -5.7609510, -2.4888067, 2.5358863
1: -10.8905535, -7.8191862, -10.8713760, -7.8377538, -2.6626577, 2.5932865
2: -5.0960665, -2.3163753, -5.0804219, -2.3590388, -2.3060894, 2.3442950
3: -6.1609359, -2.8624623, -6.1261153, -2.8826251, -3.2207575, 3.2247353
4: -13.4890308, -9.7756100, -13.4647913, -9.8270874, -2.6509037, 2.6127772
5: -3.6177490, -1.4771838, -3.5627642, -1.4808197, -1.5101886, 1.4562337
6: -10.9603052, -8.0024071, -10.8806105, -8.0273161, -2.2786591, 2.2042918
7: -9.6937418, -6.1949110, -9.6614141, -6.2756414, -3.4181004, 3.4665031
8: 9.2572174, 11.9720211, 9.2794552, 11.9539967, -2.3238258, 2.3774028
9: -7.9147377, -4.4130039, -7.8706393, -4.4301853, -2.9374046, 2.8789134

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of IS_A2_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847265, upper bound: 1.3814800
time: 4.95 seconds

## Relational analysis of IS_A2_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847266, upper bound: 1.3847267
time: 7.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.56 seconds
IS_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3550093, upper bound: 1.3844321
IS_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3582160, upper bound: 1.3844321
IS_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3591799, upper bound: 1.3844321
IS_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3623985, upper bound: 1.3844318
IS_A1_A2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3751659, upper bound: 1.3793503
IS_A1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3751659, upper bound: 1.3844582
IS_A1_A2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3793485, upper bound: 1.3793489
IS_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3793486, upper bound: 1.3844571
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3645392, upper bound: 1.3847054
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3677860, upper bound: 1.3847054
IS_A2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3805502, upper bound: 1.3814786
IS_A2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3805499, upper bound: 1.3847266
IS_A2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3847265, upper bound: 1.3814800
IS_A2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 26.56
Output dim: 8, lower bound: -1.3847266, upper bound: 1.3847267

## BFS IS instance: IS_A1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -8.3392696, -5.7991948, -8.4265537, -5.7683487, -2.3891335, 2.4204531
1: -10.8227100, -7.9949317, -10.8704252, -7.8708034, -2.5228896, 2.4748735
2: -4.9828677, -2.4225483, -5.0667019, -2.3704767, -2.2076411, 2.2199130
3: -5.9373350, -2.9279892, -6.0846267, -2.8702049, -3.0120211, 3.0733347
4: -13.4399433, -9.9346075, -13.4632721, -9.8149910, -2.5576315, 2.5070822
5: -3.5327916, -1.5972074, -3.5699561, -1.5079914, -1.4112425, 1.3883801
6: -10.8503532, -8.2190876, -10.8902731, -8.0682449, -2.1528721, 2.0791597
7: -9.5418835, -6.3232431, -9.6405363, -6.2365284, -3.3053551, 3.3172932
8: 9.3952169, 11.9354935, 9.2901497, 11.9536839, -2.2356234, 2.2999275
9: -7.8289261, -4.4776306, -7.8688903, -4.4314165, -2.8234043, 2.8295336

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3550093, upper bound: 1.3793202
time: 5.97 seconds

## Relational analysis of IS_A1_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3550093, upper bound: 1.3844321
time: 5.22 seconds

## BFS IS instance: IS_A1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.3644066, -5.7832317, -8.4271107, -5.7638044, -2.4108744, 2.4307604
1: -10.8614435, -7.9648232, -10.8709660, -7.8624983, -2.5388098, 2.4952383
2: -5.0195274, -2.3906360, -5.0669723, -2.3614788, -2.2429852, 2.2415175
3: -5.9563489, -2.9067862, -6.0894713, -2.8698022, -3.0259933, 3.0824842
4: -13.4512005, -9.9235306, -13.4638624, -9.8122740, -2.5648177, 2.5163398
5: -3.5428658, -1.5840856, -3.5728445, -1.5078132, -1.4182035, 1.3988128
6: -10.8663273, -8.2032566, -10.8904476, -8.0649471, -2.1601777, 2.0990167
7: -9.5545120, -6.3162179, -9.6417599, -6.2347975, -3.3197145, 3.3255420
8: 9.3864555, 11.9447250, 9.2881460, 11.9538898, -2.2462702, 2.3050189
9: -7.8393559, -4.4754577, -7.8697367, -4.4309130, -2.8342700, 2.8328161

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_A1_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3523824, upper bound: 1.3793202
time: 5.91 seconds

## Relational analysis of IS_A1_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3582160, upper bound: 1.3844321
time: 4.54 seconds

## BFS IS instance: IS_A1_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -8.3645229, -5.7843680, -8.4325294, -5.7657337, -2.4155946, 2.4755721
1: -10.8427048, -7.9537888, -10.8722773, -7.8595524, -2.5621309, 2.4919963
2: -5.0391741, -2.4018588, -5.0817914, -2.3683400, -2.2443066, 2.2713766
3: -6.0535345, -2.8971269, -6.1158285, -2.8681738, -3.1176138, 3.1556683
4: -13.4595537, -9.8642464, -13.4646797, -9.7958536, -2.6095715, 2.5337861
5: -3.5414314, -1.5663893, -3.5706270, -1.4999690, -1.4360912, 1.4122837
6: -10.8622389, -8.1707020, -10.8910494, -8.0555935, -2.1861126, 2.1140764
7: -9.6163769, -6.2927012, -9.6596766, -6.2324834, -3.3838935, 3.3669753
8: 9.3713684, 11.9419289, 9.2840662, 11.9540939, -2.2570915, 2.3194816
9: -7.8490710, -4.4419451, -7.8718448, -4.4219046, -2.8556581, 2.8545556

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3591799, upper bound: 1.3793202
time: 4.73 seconds

## Relational analysis of IS_A1_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3591799, upper bound: 1.3844321
time: 5.22 seconds

## BFS IS instance: IS_A1_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.3896723, -5.7684045, -8.4330864, -5.7611895, -2.4373264, 2.4858756
1: -10.8814392, -7.9236836, -10.8728189, -7.8512459, -2.5780797, 2.5123577
2: -5.0758348, -2.3699431, -5.0820613, -2.3593416, -2.2796812, 2.2929850
3: -6.0725360, -2.8759217, -6.1206732, -2.8677707, -3.1315718, 3.1698422
4: -13.4708042, -9.8531742, -13.4652710, -9.7931376, -2.6167545, 2.5430524
5: -3.5515063, -1.5532676, -3.5735140, -1.4997911, -1.4430517, 1.4227406
6: -10.8782148, -8.1548834, -10.8912239, -8.0522966, -2.1934166, 2.1339340
7: -9.6289902, -6.2856760, -9.6609011, -6.2307482, -3.3982420, 3.3752251
8: 9.3626080, 11.9511614, 9.2820625, 11.9543018, -2.2677388, 2.3245747
9: -7.8594995, -4.4397697, -7.8726902, -4.4213991, -2.8667188, 2.8578396

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of IS_A1_A1_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3623985, upper bound: 1.3793198
time: 5.33 seconds

## Relational analysis of IS_A1_A1_B2_A2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3623985, upper bound: 1.3844318
time: 4.78 seconds

## BFS IS instance: IS_A1_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.3900928, -5.7770591, -8.4434137, -5.7195892, -2.4363661, 2.4728351
1: -10.8482761, -7.8788118, -10.8828058, -7.8336210, -2.6042728, 2.5291114
2: -5.0118690, -2.3824759, -5.0690203, -2.3207216, -2.2664270, 2.2742968
3: -6.0055523, -2.9017217, -6.1236992, -2.8816290, -3.0612230, 3.1448030
4: -13.4447193, -9.8744783, -13.4865150, -9.8357868, -2.5953596, 2.5621610
5: -3.5623422, -1.5454816, -3.6044693, -1.4860440, -1.4670136, 1.4222990
6: -10.8776140, -8.1189060, -10.9466667, -8.0203915, -2.2411160, 2.1394312
7: -9.5758343, -6.2628016, -9.6652622, -6.2459793, -3.3298550, 3.4024606
8: 9.3044214, 11.9465714, 9.2784872, 11.9700880, -2.2999783, 2.3465366
9: -7.8496666, -4.4605069, -7.9061756, -4.4340529, -2.8608699, 2.8667691

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of IS_A1_A2_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3751611, upper bound: 1.3812159
time: 6.68 seconds

## Relational analysis of IS_A1_A2_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3751611, upper bound: 1.3844523
time: 7.10 seconds

## BFS IS instance: IS_A1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4151878, -5.7622027, -8.4493847, -5.7169795, -2.4626746, 2.5279279
1: -10.8682156, -7.8377247, -10.8846531, -7.8223662, -2.6392088, 2.5462084
2: -5.0680285, -2.3617702, -5.0841150, -2.3185992, -2.3030667, 2.3211226
3: -6.1215653, -2.8708744, -6.1548834, -2.8795910, -3.1668673, 3.2319908
4: -13.4643307, -9.8038254, -13.4879274, -9.8166409, -2.6452637, 2.5888941
5: -3.5709848, -1.5146755, -3.6051407, -1.4780264, -1.4918427, 1.4461954
6: -10.8894720, -8.0707016, -10.9474382, -8.0077477, -2.2743161, 2.1744418
7: -9.6502485, -6.2322779, -9.6843796, -6.2419291, -3.4083195, 3.4521017
8: 9.2805891, 11.9530659, 9.2723799, 11.9705000, -2.3213606, 2.3647900
9: -7.8700299, -4.4248419, -7.9090862, -4.4245582, -2.8932104, 2.8919406

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of IS_A1_A2_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3793438, upper bound: 1.3812156
time: 6.00 seconds

## Relational analysis of IS_A1_A2_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3793438, upper bound: 1.3844539
time: 6.73 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.4085083, -5.7382407, -8.4433517, -5.7651081, -2.4842200, 2.4693692
1: -10.8651743, -7.9344044, -10.8749886, -7.8586550, -2.5669322, 2.5688646
2: -5.0669489, -2.3563452, -5.0890985, -2.3668537, -2.2993073, 2.2758999
3: -6.0938158, -2.8887398, -6.1189361, -2.8665071, -3.1801596, 3.1508884
4: -13.4843035, -9.8366251, -13.4651489, -9.7904654, -2.5891528, 2.6136482
5: -3.5883117, -1.5289696, -3.5720561, -1.4895530, -1.4537660, 1.4471754
6: -10.9332752, -8.1030159, -10.8922539, -8.0410252, -2.2070975, 2.1959043
7: -9.6599016, -6.2551212, -9.6657772, -6.2313466, -3.4285550, 3.4106560
8: 9.3479862, 11.9610262, 9.2791653, 11.9548359, -2.2847357, 2.3249061
9: -7.8940158, -4.4300346, -7.8736863, -4.4200492, -2.8912826, 2.8858423

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_A1_B2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3645391, upper bound: 1.3677888
time: 5.36 seconds

## Relational analysis of IS_A2_A1_B2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3645391, upper bound: 1.3847053
time: 7.67 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -8.4336576, -5.7222762, -8.4439068, -5.7605634, -2.4976645, 2.4796767
1: -10.9039154, -7.9042978, -10.8755293, -7.8503480, -2.5829253, 2.5892365
2: -5.1035905, -2.3244224, -5.0893679, -2.3578563, -2.3197289, 2.2975175
3: -6.1128306, -2.8675523, -6.1237822, -2.8661015, -3.1940651, 3.1768966
4: -13.4955511, -9.8255510, -13.4657393, -9.7877445, -2.5963359, 2.6229219
5: -3.5983884, -1.5158567, -3.5749414, -1.4893751, -1.4607389, 1.4526050
6: -10.9492464, -8.0874453, -10.8924313, -8.0377245, -2.2144022, 2.2112982
7: -9.6724024, -6.2480779, -9.6670017, -6.2296138, -3.4427886, 3.4189239
8: 9.3392401, 11.9702568, 9.2771578, 11.9550419, -2.2953825, 2.3300014
9: -7.9045839, -4.4278436, -7.8745308, -4.4195452, -2.8991427, 2.8891416

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3623986, upper bound: 1.3677869
time: 7.37 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3677859, upper bound: 1.3847054
time: 6.06 seconds

## BFS IS instance: IS_A2_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.4346638, -5.7309752, -8.4579201, -5.7630610, -2.4568610, 2.4832189
1: -10.8706570, -7.8602934, -10.9063816, -7.8471498, -2.6192307, 2.5894120
2: -5.0399532, -2.3370719, -5.1010494, -2.3597898, -2.2598524, 2.2935221
3: -6.0449076, -2.8933175, -6.0975037, -2.8648531, -3.1347895, 3.1393986
4: -13.4694147, -9.8462353, -13.4726381, -9.8443394, -2.5972714, 2.5856354
5: -3.6091096, -1.5079532, -3.5623546, -1.4763284, -1.4855963, 1.4294231
6: -10.9484644, -8.0506039, -10.8952122, -8.0353680, -2.2463627, 2.1834846
7: -9.6194382, -6.2254572, -9.6506443, -6.2785330, -3.3409052, 3.4251871
8: 9.2811356, 11.9654894, 9.2836285, 11.9621220, -2.3093853, 2.3603523
9: -7.8943639, -4.4485788, -7.8752184, -4.4392376, -2.9000907, 2.8612633

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5832

## Relational analysis of IS_A2_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_A2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3798932, upper bound: 1.3802983
time: 5.70 seconds

## Relational analysis of IS_A2_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3805457, upper bound: 1.3847230
time: 6.50 seconds

## BFS IS instance: IS_A2_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.4591017, -5.7206807, -8.4389076, -5.7764072, -2.4727888, 2.5276995
1: -10.8900051, -7.8274951, -10.8695049, -7.8660488, -2.6337705, 2.5830650
2: -5.0957999, -2.3253832, -5.0794973, -2.3896112, -2.2747693, 2.3305175
3: -6.1560993, -2.8628654, -6.1096120, -2.8840146, -3.2144117, 3.2078867
4: -13.4884405, -9.7783318, -13.4627905, -9.8363180, -2.6401186, 2.6052332
5: -3.6148627, -1.4773606, -3.5529301, -1.4814268, -1.5050120, 1.4458811
6: -10.9601269, -8.0057011, -10.8799953, -8.0385656, -2.2639768, 2.1983695
7: -9.6925201, -6.1966496, -9.6572704, -6.2815185, -3.4110017, 3.4606209
8: 9.2592278, 11.9718113, 9.2862778, 11.9532909, -2.3189945, 2.3680024
9: -7.9138880, -4.4135079, -7.8677368, -4.4319072, -2.9344563, 2.8750706

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5832

## Relational analysis of IS_A2_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3840705, upper bound: 1.3770664
time: 5.08 seconds

## Relational analysis of IS_A2_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847228, upper bound: 1.3814744
time: 6.32 seconds

## BFS IS instance: IS_A2_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.4596758, -5.7161350, -8.4639044, -5.7604485, -2.4830856, 2.5383844
1: -10.8905525, -7.8191972, -10.9082270, -7.8358994, -2.6541901, 2.6047263
2: -5.0960665, -2.3163881, -5.1161509, -2.3576651, -2.2963839, 2.3450165
3: -6.1609278, -2.8624632, -6.1287026, -2.8628180, -3.2404089, 3.2219062
4: -13.4890289, -9.7756128, -13.4740458, -9.8251867, -2.6495681, 2.6124105
5: -3.6177485, -1.4771831, -3.5630252, -1.4683067, -1.5104375, 1.4532707
6: -10.9603043, -8.0024137, -10.8959856, -8.0227213, -2.2795959, 2.2184305
7: -9.6937399, -6.1949129, -9.6697817, -6.2744837, -3.4192562, 3.4748688
8: 9.2572203, 11.9720192, 9.2775230, 11.9625282, -2.3308854, 2.3786201
9: -7.9147348, -4.4130030, -7.8781343, -4.4297166, -2.9377789, 2.8861399

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5832

## Relational analysis of IS_A2_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3840706, upper bound: 1.3802982
time: 6.27 seconds

## Relational analysis of IS_A2_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847228, upper bound: 1.3847226
time: 6.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 25.27 seconds
IS_A1_A1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3550093, upper bound: 1.3793202
IS_A1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3550093, upper bound: 1.3844321
IS_A1_A1_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3523824, upper bound: 1.3793202
IS_A1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3582160, upper bound: 1.3844321
IS_A1_A1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3591799, upper bound: 1.3793202
IS_A1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3591799, upper bound: 1.3844321
IS_A1_A1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3623985, upper bound: 1.3793198
IS_A1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3623985, upper bound: 1.3844318
IS_A1_A2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3751611, upper bound: 1.3812159
IS_A1_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3751611, upper bound: 1.3844523
IS_A1_A2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3793438, upper bound: 1.3812156
IS_A1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3793438, upper bound: 1.3844539
IS_A2_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3645391, upper bound: 1.3677888
IS_A2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3645391, upper bound: 1.3847053
IS_A2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3623986, upper bound: 1.3677869
IS_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3677859, upper bound: 1.3847054
IS_A2_A2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3798932, upper bound: 1.3802983
IS_A2_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3805457, upper bound: 1.3847230
IS_A2_A2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3840705, upper bound: 1.3770664
IS_A2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3847228, upper bound: 1.3814744
IS_A2_A2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3840706, upper bound: 1.3802982
IS_A2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 25.27
Output dim: 8, lower bound: -1.3847228, upper bound: 1.3847226

## BFS IS instance: IS_A1_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.3392696, -5.7991948, -8.4459352, -5.7237458, -2.4032555, 2.4323084
1: -10.8227100, -7.9949317, -10.8864422, -7.8543243, -2.5237131, 2.4826589
2: -4.9828677, -2.4225483, -5.0777259, -2.3285327, -2.2350307, 2.2238579
3: -5.9373350, -2.9279892, -6.1168165, -2.8655169, -3.0125799, 3.0911036
4: -13.4399433, -9.9346075, -13.4868765, -9.7991323, -2.5681891, 2.5308704
5: -3.5327916, -1.5972074, -3.6137674, -1.4947172, -1.4145970, 1.3959756
6: -10.8503532, -8.2190876, -10.9583197, -8.0337753, -2.1770892, 2.0969651
7: -9.5418835, -6.3232431, -9.6700344, -6.2016683, -3.3402152, 3.3467913
8: 9.3952169, 11.9354935, 9.2781725, 11.9709291, -2.2537279, 2.3096416
9: -7.8289261, -4.4776306, -7.9092779, -4.4238424, -2.8252625, 2.8620119

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5736
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 5788
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 5788
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5736

## Relational analysis of IS_A1_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3550091, upper bound: 1.3675117
time: 6.35 seconds

## Relational analysis of IS_A1_A1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3550093, upper bound: 1.3844321
time: 5.53 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 24.66 seconds
IS_A1_A1_B2_A1_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 24.66
Output dim: 8, lower bound: -1.3550091, upper bound: 1.3675117
IS_A1_A1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 24.66
Output dim: 8, lower bound: -1.3550093, upper bound: 1.3844321
IS_A1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.66
Output dim: 8, lower bound: -1.3582160, upper bound: 1.3844321
IS_A1_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 24.66
Output dim: 8, lower bound: -1.3591799, upper bound: 1.3844321
IS_A1_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.66
Output dim: 8, lower bound: -1.3623985, upper bound: 1.3844318
IS_A1_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 24.66
Output dim: 8, lower bound: -1.3751611, upper bound: 1.3844523
IS_A1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 24.66
Output dim: 8, lower bound: -1.3793438, upper bound: 1.3844539
IS_A2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 24.66
Output dim: 8, lower bound: -1.3645391, upper bound: 1.3847053
IS_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 24.66
Output dim: 8, lower bound: -1.3677859, upper bound: 1.3847054
IS_A2_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 24.66
Output dim: 8, lower bound: -1.3805457, upper bound: 1.3847230
IS_A2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 24.66
Output dim: 8, lower bound: -1.3847228, upper bound: 1.3814744
IS_A2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 24.66
Output dim: 8, lower bound: -1.3847228, upper bound: 1.3847226
Binary search (step 2): status=Status.UNKNOWN, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.359529972076416
rel_dist={8: [-1.3847687820365167, 1.3847690605870273]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01953125
execution time: 2418.72 seconds
