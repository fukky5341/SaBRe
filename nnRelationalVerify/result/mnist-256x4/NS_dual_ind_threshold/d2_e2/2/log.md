## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.07289625500000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0557825, 0.0118011, -0.0557825, 0.0118011, -0.0675835, 0.0675835)
1: (0.9177408, 1.0373535, 0.9177408, 1.0373535, -0.1196128, 0.1196128)
2: (-0.0171943, 0.0570202, -0.0171943, 0.0570202, -0.0742145, 0.0742145)
3: (-0.0394948, 0.0086721, -0.0394948, 0.0086721, -0.0481669, 0.0481669)
4: (-0.0429191, 0.0240216, -0.0429191, 0.0240216, -0.0669407, 0.0669407)
5: (-0.0077139, 0.0706782, -0.0077139, 0.0706782, -0.0783920, 0.0783920)
6: (-0.0110202, 0.0225717, -0.0110202, 0.0225717, -0.0335919, 0.0335919)
7: (-0.0116670, 0.1009442, -0.0116670, 0.1009442, -0.1126113, 0.1126113)
8: (-0.0132512, 0.0338270, -0.0132512, 0.0338270, -0.0470782, 0.0470782)
9: (-0.0619663, 0.0257034, -0.0619663, 0.0257034, -0.0876696, 0.0876696)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.76 + 2.85 = 4.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0767329, upper bound: 0.0767329

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0752231, upper bound: 0.0733177
time: 1.57 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0762527, upper bound: 0.0762527
time: 1.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.10 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.10
Output dim: 1, lower bound: -0.0752231, upper bound: 0.0733177
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.10
Output dim: 1, lower bound: -0.0762527, upper bound: 0.0762527

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0401211, 0.0096314, -0.0532233, 0.0098370, -0.0499580, 0.0628547
1: 0.9313526, 1.0035039, 0.9311829, 1.0312945, -0.0999419, 0.0723210
2: -0.0170703, 0.0307478, -0.0171670, 0.0504851, -0.0675554, 0.0479148
3: -0.0253839, 0.0082877, -0.0371831, 0.0083699, -0.0337538, 0.0454708
4: -0.0238668, 0.0235204, -0.0397675, 0.0237752, -0.0476420, 0.0632879
5: -0.0062834, 0.0482794, -0.0074616, 0.0663410, -0.0726244, 0.0557410
6: -0.0106940, 0.0213708, -0.0109587, 0.0223573, -0.0330513, 0.0323295
7: -0.0114504, 0.0670906, -0.0116336, 0.0930230, -0.1044734, 0.0787242
8: -0.0073975, 0.0326692, -0.0113168, 0.0326942, -0.0400917, 0.0439861
9: -0.0321084, 0.0235858, -0.0571436, 0.0237443, -0.0558527, 0.0807294

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0725824, upper bound: 0.0723359
time: 1.56 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0747300, upper bound: 0.0728371
time: 1.96 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0522076, 0.0098212, -0.0557825, 0.0118011, -0.0640087, 0.0656037
1: 0.9312019, 1.0288897, 0.9177408, 1.0373535, -0.1061516, 0.1111489
2: -0.0171565, 0.0489197, -0.0171943, 0.0570202, -0.0741767, 0.0661140
3: -0.0362667, 0.0083609, -0.0394948, 0.0086721, -0.0449387, 0.0478557
4: -0.0385191, 0.0237471, -0.0429191, 0.0240216, -0.0625407, 0.0666661
5: -0.0073601, 0.0649151, -0.0077139, 0.0706782, -0.0780382, 0.0726290
6: -0.0109345, 0.0222706, -0.0110202, 0.0225717, -0.0335062, 0.0332908
7: -0.0116200, 0.0910122, -0.0116670, 0.1009442, -0.1125642, 0.1026793
8: -0.0105514, 0.0326914, -0.0132512, 0.0338270, -0.0443784, 0.0459426
9: -0.0551932, 0.0237266, -0.0619663, 0.0257034, -0.0808965, 0.0856929

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733176, upper bound: 0.0752232
time: 1.41 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0733176, upper bound: 0.0762527
time: 2.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.59 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 5.59
Output dim: 1, lower bound: -0.0725824, upper bound: 0.0723359
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 1, lower bound: -0.0747300, upper bound: 0.0728371
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 1, lower bound: -0.0733176, upper bound: 0.0752232
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.59
Output dim: 1, lower bound: -0.0733176, upper bound: 0.0762527

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0401211, 0.0096314, -0.0490931, 0.0097267, -0.0498478, 0.0587245
1: 0.9313526, 1.0035039, 0.9313483, 1.0215120, -0.0901595, 0.0721557
2: -0.0170703, 0.0307478, -0.0170756, 0.0441124, -0.0611827, 0.0478235
3: -0.0253839, 0.0082877, -0.0334310, 0.0082913, -0.0336752, 0.0417187
4: -0.0238668, 0.0235204, -0.0346704, 0.0235306, -0.0473974, 0.0581907
5: -0.0062834, 0.0482794, -0.0070453, 0.0605077, -0.0667911, 0.0553247
6: -0.0106940, 0.0213708, -0.0107667, 0.0220049, -0.0326989, 0.0321375
7: -0.0114504, 0.0670906, -0.0115405, 0.0848157, -0.0962661, 0.0786311
8: -0.0073975, 0.0326692, -0.0082465, 0.0326699, -0.0400673, 0.0409157
9: -0.0321084, 0.0235858, -0.0491926, 0.0235899, -0.0556984, 0.0727784

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0746578, upper bound: 0.0728224
time: 1.60 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0746463, upper bound: 0.0728177
time: 1.48 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0522076, 0.0098212, -0.0401211, 0.0096314, -0.0618390, 0.0499423
1: 0.9312019, 1.0288897, 0.9313526, 1.0035039, -0.0723020, 0.0975371
2: -0.0171565, 0.0489197, -0.0170703, 0.0307478, -0.0479043, 0.0659899
3: -0.0362667, 0.0083609, -0.0253839, 0.0082877, -0.0445544, 0.0337448
4: -0.0385191, 0.0237471, -0.0238668, 0.0235204, -0.0620394, 0.0476139
5: -0.0073601, 0.0649151, -0.0062834, 0.0482794, -0.0556395, 0.0711985
6: -0.0109345, 0.0222706, -0.0106940, 0.0213708, -0.0323052, 0.0329646
7: -0.0116200, 0.0910122, -0.0114504, 0.0670906, -0.0787105, 0.1024626
8: -0.0105514, 0.0326914, -0.0073975, 0.0326692, -0.0432207, 0.0400889
9: -0.0551932, 0.0237266, -0.0321084, 0.0235858, -0.0787790, 0.0558350

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0723359, upper bound: 0.0725824
time: 2.12 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728371, upper bound: 0.0747300
time: 1.60 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0522076, 0.0098212, -0.0522076, 0.0098212, -0.0620288, 0.0620288
1: 0.9312019, 1.0288897, 0.9312019, 1.0288897, -0.0976877, 0.0976877
2: -0.0171565, 0.0489197, -0.0171565, 0.0489197, -0.0660761, 0.0660761
3: -0.0362667, 0.0083609, -0.0362667, 0.0083609, -0.0446275, 0.0446275
4: -0.0385191, 0.0237471, -0.0385191, 0.0237471, -0.0622661, 0.0622661
5: -0.0073601, 0.0649151, -0.0073601, 0.0649151, -0.0722752, 0.0722752
6: -0.0109345, 0.0222706, -0.0109345, 0.0222706, -0.0332051, 0.0332051
7: -0.0116200, 0.0910122, -0.0116200, 0.0910122, -0.1026322, 0.1026322
8: -0.0105514, 0.0326914, -0.0105514, 0.0326914, -0.0432428, 0.0432428
9: -0.0551932, 0.0237266, -0.0551932, 0.0237266, -0.0789198, 0.0789198

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0723359, upper bound: 0.0738261
time: 2.07 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728371, upper bound: 0.0757006
time: 1.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.43 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 1, lower bound: -0.0746578, upper bound: 0.0728224
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 1, lower bound: -0.0746463, upper bound: 0.0728177
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 5.43
Output dim: 1, lower bound: -0.0723359, upper bound: 0.0725824
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 1, lower bound: -0.0728371, upper bound: 0.0747300
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 1, lower bound: -0.0723359, upper bound: 0.0738261
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.43
Output dim: 1, lower bound: -0.0728371, upper bound: 0.0757006

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0394707, 0.0095301, -0.0490669, 0.0097227, -0.0491933, 0.0585970
1: 0.9315279, 1.0027279, 0.9313549, 1.0214498, -0.0899220, 0.0713729
2: -0.0169739, 0.0297220, -0.0170719, 0.0440707, -0.0610446, 0.0467939
3: -0.0247587, 0.0082047, -0.0334057, 0.0082881, -0.0330469, 0.0416103
4: -0.0228805, 0.0232616, -0.0346295, 0.0235206, -0.0464011, 0.0578912
5: -0.0057086, 0.0475331, -0.0070229, 0.0604787, -0.0661873, 0.0545560
6: -0.0105018, 0.0207460, -0.0107592, 0.0219804, -0.0324822, 0.0315052
7: -0.0113665, 0.0659306, -0.0115372, 0.0847689, -0.0961354, 0.0774678
8: -0.0070815, 0.0326434, -0.0082167, 0.0326689, -0.0397504, 0.0408602
9: -0.0311504, 0.0234222, -0.0491542, 0.0235836, -0.0547341, 0.0725764

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727576, upper bound: 0.0727457
time: 1.77 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727576, upper bound: 0.0727457
time: 2.91 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0403866, 0.0097350, -0.0490625, 0.0097214, -0.0501080, 0.0587975
1: 0.9311408, 1.0032253, 0.9313573, 1.0214392, -0.0902984, 0.0718681
2: -0.0171862, 0.0313582, -0.0170706, 0.0440633, -0.0612495, 0.0484289
3: -0.0256553, 0.0083879, -0.0334014, 0.0082870, -0.0339423, 0.0417893
4: -0.0241808, 0.0238325, -0.0346232, 0.0235172, -0.0476979, 0.0584557
5: -0.0060420, 0.0489379, -0.0070208, 0.0604727, -0.0665146, 0.0559588
6: -0.0109126, 0.0210790, -0.0107567, 0.0219783, -0.0328909, 0.0318357
7: -0.0115338, 0.0677899, -0.0115362, 0.0847604, -0.0962943, 0.0793260
8: -0.0077628, 0.0327004, -0.0082106, 0.0326685, -0.0404314, 0.0409111
9: -0.0329359, 0.0237838, -0.0491465, 0.0235815, -0.0565174, 0.0729302

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 69

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727441, upper bound: 0.0727441
time: 1.53 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727441, upper bound: 0.0727441
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0478819, 0.0097072, -0.0401211, 0.0096314, -0.0575134, 0.0498283
1: 0.9313681, 1.0186549, 0.9313526, 1.0035039, -0.0721358, 0.0873024
2: -0.0170645, 0.0422405, -0.0170703, 0.0307478, -0.0478123, 0.0593108
3: -0.0323250, 0.0082818, -0.0253839, 0.0082877, -0.0406127, 0.0336657
4: -0.0331707, 0.0235009, -0.0238668, 0.0235204, -0.0566911, 0.0473677
5: -0.0069235, 0.0588182, -0.0062834, 0.0482794, -0.0552030, 0.0651016
6: -0.0107390, 0.0219004, -0.0106940, 0.0213708, -0.0321098, 0.0325944
7: -0.0115234, 0.0823833, -0.0114504, 0.0670906, -0.0786140, 0.0938337
8: -0.0074440, 0.0326669, -0.0073975, 0.0326692, -0.0401132, 0.0400644
9: -0.0468369, 0.0235714, -0.0321084, 0.0235858, -0.0704227, 0.0556798

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728224, upper bound: 0.0746578
time: 1.84 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728177, upper bound: 0.0746463
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0385228, 0.0095355, -0.0503510, 0.0097868, -0.0483096, 0.0598865
1: 0.9314978, 1.0031961, 0.9312459, 1.0244930, -0.0929952, 0.0719502
2: -0.0169901, 0.0282610, -0.0171319, 0.0460942, -0.0630843, 0.0453929
3: -0.0239140, 0.0082187, -0.0345840, 0.0083399, -0.0322539, 0.0428027
4: -0.0218875, 0.0233056, -0.0362379, 0.0236817, -0.0455691, 0.0595435
5: -0.0061137, 0.0459847, -0.0071852, 0.0623307, -0.0684444, 0.0531698
6: -0.0105260, 0.0212289, -0.0108796, 0.0221227, -0.0326487, 0.0321085
7: -0.0113696, 0.0638512, -0.0115904, 0.0873347, -0.0987043, 0.0754416
8: -0.0071253, 0.0326479, -0.0092137, 0.0326849, -0.0398102, 0.0418616
9: -0.0289817, 0.0234503, -0.0516488, 0.0236855, -0.0526672, 0.0750990

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0737820, upper bound: 0.0738053
time: 1.40 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0737820, upper bound: 0.0738261
time: 1.62 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0478819, 0.0097072, -0.0522076, 0.0098212, -0.0577032, 0.0619148
1: 0.9313681, 1.0186549, 0.9312019, 1.0288897, -0.0975215, 0.0874530
2: -0.0170645, 0.0422405, -0.0171565, 0.0489197, -0.0659841, 0.0593970
3: -0.0323250, 0.0082818, -0.0362667, 0.0083609, -0.0406859, 0.0445484
4: -0.0331707, 0.0235009, -0.0385191, 0.0237471, -0.0569177, 0.0620200
5: -0.0069235, 0.0588182, -0.0073601, 0.0649151, -0.0718386, 0.0661783
6: -0.0107390, 0.0219004, -0.0109345, 0.0222706, -0.0330097, 0.0328349
7: -0.0115234, 0.0823833, -0.0116200, 0.0910122, -0.1025356, 0.0940033
8: -0.0074440, 0.0326669, -0.0105514, 0.0326914, -0.0401354, 0.0432184
9: -0.0468369, 0.0235714, -0.0551932, 0.0237266, -0.0705635, 0.0787646

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0738005, upper bound: 0.0753705
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0738005, upper bound: 0.0757006
time: 1.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.94 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.94
Output dim: 1, lower bound: -0.0727576, upper bound: 0.0727457
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.94
Output dim: 1, lower bound: -0.0727576, upper bound: 0.0727457
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.94
Output dim: 1, lower bound: -0.0727441, upper bound: 0.0727441
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.94
Output dim: 1, lower bound: -0.0727441, upper bound: 0.0727441
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 1, lower bound: -0.0728224, upper bound: 0.0746578
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 1, lower bound: -0.0728177, upper bound: 0.0746463
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 1, lower bound: -0.0737820, upper bound: 0.0738053
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 1, lower bound: -0.0737820, upper bound: 0.0738261
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 1, lower bound: -0.0738005, upper bound: 0.0753705
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.94
Output dim: 1, lower bound: -0.0738005, upper bound: 0.0757006

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0478557, 0.0097032, -0.0394707, 0.0095301, -0.0573858, 0.0491738
1: 0.9313748, 1.0185921, 0.9315279, 1.0027279, -0.0713530, 0.0870643
2: -0.0170607, 0.0421989, -0.0169739, 0.0297220, -0.0467827, 0.0591728
3: -0.0322996, 0.0082786, -0.0247587, 0.0082047, -0.0405042, 0.0330373
4: -0.0331297, 0.0234909, -0.0228805, 0.0232616, -0.0563914, 0.0463714
5: -0.0069008, 0.0587893, -0.0057086, 0.0475331, -0.0544340, 0.0644979
6: -0.0107315, 0.0218757, -0.0105018, 0.0207460, -0.0314775, 0.0323774
7: -0.0115201, 0.0823365, -0.0113665, 0.0659306, -0.0774507, 0.0937030
8: -0.0074316, 0.0326659, -0.0070815, 0.0326434, -0.0400750, 0.0397475
9: -0.0467987, 0.0235650, -0.0311504, 0.0234222, -0.0702209, 0.0547154

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0721050, upper bound: 0.0720337
time: 1.71 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0724084, upper bound: 0.0743517
time: 2.11 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0478514, 0.0097020, -0.0403866, 0.0097350, -0.0575864, 0.0500886
1: 0.9313772, 1.0185810, 0.9311408, 1.0032253, -0.0718481, 0.0874403
2: -0.0170595, 0.0421917, -0.0171862, 0.0313582, -0.0484177, 0.0593780
3: -0.0322955, 0.0082775, -0.0256553, 0.0083879, -0.0406835, 0.0339328
4: -0.0331239, 0.0234875, -0.0241808, 0.0238325, -0.0569564, 0.0476683
5: -0.0068993, 0.0587832, -0.0060420, 0.0489379, -0.0558373, 0.0648252
6: -0.0107291, 0.0218741, -0.0109126, 0.0210790, -0.0318081, 0.0327866
7: -0.0115191, 0.0823282, -0.0115338, 0.0677899, -0.0793090, 0.0938620
8: -0.0074276, 0.0326656, -0.0077628, 0.0327004, -0.0401280, 0.0404284
9: -0.0467909, 0.0235629, -0.0329359, 0.0237838, -0.0705746, 0.0564988

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0720966, upper bound: 0.0720263
time: 1.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0724027, upper bound: 0.0743396
time: 1.87 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0385228, 0.0095355, -0.0385228, 0.0095355, -0.0480583, 0.0480583
1: 0.9314978, 1.0031961, 0.9314978, 1.0031961, -0.0716984, 0.0716984
2: -0.0169901, 0.0282610, -0.0169901, 0.0282610, -0.0452510, 0.0452510
3: -0.0239140, 0.0082187, -0.0239140, 0.0082187, -0.0321328, 0.0321328
4: -0.0218875, 0.0233056, -0.0218875, 0.0233056, -0.0451931, 0.0451931
5: -0.0061137, 0.0459847, -0.0061137, 0.0459847, -0.0520983, 0.0520983
6: -0.0105260, 0.0212289, -0.0105260, 0.0212289, -0.0317550, 0.0317550
7: -0.0113696, 0.0638512, -0.0113696, 0.0638512, -0.0752207, 0.0752207
8: -0.0071253, 0.0326479, -0.0071253, 0.0326479, -0.0397732, 0.0397732
9: -0.0289817, 0.0234503, -0.0289817, 0.0234503, -0.0524320, 0.0524320

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732607, upper bound: 0.0728197
time: 1.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734320, upper bound: 0.0734531
time: 1.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0385228, 0.0095355, -0.0478819, 0.0097072, -0.0482300, 0.0574175
1: 0.9314978, 1.0031961, 0.9313681, 1.0186549, -0.0871572, 0.0718280
2: -0.0169901, 0.0282610, -0.0170645, 0.0422405, -0.0592305, 0.0453255
3: -0.0239140, 0.0082187, -0.0323250, 0.0082818, -0.0321958, 0.0405437
4: -0.0218875, 0.0233056, -0.0331707, 0.0235009, -0.0453884, 0.0564763
5: -0.0061137, 0.0459847, -0.0069235, 0.0588182, -0.0649319, 0.0529082
6: -0.0105260, 0.0212289, -0.0107390, 0.0219004, -0.0324264, 0.0319680
7: -0.0113696, 0.0638512, -0.0115234, 0.0823833, -0.0937529, 0.0753746
8: -0.0071253, 0.0326479, -0.0074440, 0.0326669, -0.0397922, 0.0400918
9: -0.0289817, 0.0234503, -0.0468369, 0.0235714, -0.0525531, 0.0702871

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732607, upper bound: 0.0728450
time: 1.46 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734320, upper bound: 0.0734753
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0478819, 0.0097072, -0.0385228, 0.0095355, -0.0574175, 0.0482300
1: 0.9313681, 1.0186549, 0.9314978, 1.0031961, -0.0718280, 0.0871572
2: -0.0170645, 0.0422405, -0.0169901, 0.0282610, -0.0453255, 0.0592305
3: -0.0323250, 0.0082818, -0.0239140, 0.0082187, -0.0405437, 0.0321958
4: -0.0331707, 0.0235009, -0.0218875, 0.0233056, -0.0564763, 0.0453884
5: -0.0069235, 0.0588182, -0.0061137, 0.0459847, -0.0529082, 0.0649319
6: -0.0107390, 0.0219004, -0.0105260, 0.0212289, -0.0319680, 0.0324264
7: -0.0115234, 0.0823833, -0.0113696, 0.0638512, -0.0753746, 0.0937529
8: -0.0074440, 0.0326669, -0.0071253, 0.0326479, -0.0400918, 0.0397922
9: -0.0468369, 0.0235714, -0.0289817, 0.0234503, -0.0702871, 0.0525531

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0728044, upper bound: 0.0724433
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0735467, upper bound: 0.0750928
time: 1.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0478819, 0.0097072, -0.0478819, 0.0097072, -0.0575891, 0.0575891
1: 0.9313681, 1.0186549, 0.9313681, 1.0186549, -0.0872868, 0.0872868
2: -0.0170645, 0.0422405, -0.0170645, 0.0422405, -0.0593050, 0.0593050
3: -0.0323250, 0.0082818, -0.0323250, 0.0082818, -0.0406068, 0.0406068
4: -0.0331707, 0.0235009, -0.0331707, 0.0235009, -0.0566716, 0.0566716
5: -0.0069235, 0.0588182, -0.0069235, 0.0588182, -0.0657417, 0.0657417
6: -0.0107390, 0.0219004, -0.0107390, 0.0219004, -0.0326394, 0.0326394
7: -0.0115234, 0.0823833, -0.0115234, 0.0823833, -0.0939067, 0.0939067
8: -0.0074440, 0.0326669, -0.0074440, 0.0326669, -0.0401109, 0.0401109
9: -0.0468369, 0.0235714, -0.0468369, 0.0235714, -0.0704082, 0.0704082

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0728044, upper bound: 0.0728011
time: 2.21 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0735467, upper bound: 0.0754117
time: 1.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.98 seconds
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0721050, upper bound: 0.0720337
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0724084, upper bound: 0.0743517
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0720966, upper bound: 0.0720263
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0724027, upper bound: 0.0743396
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0732607, upper bound: 0.0728197
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0734320, upper bound: 0.0734531
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0732607, upper bound: 0.0728450
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0734320, upper bound: 0.0734753
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0728044, upper bound: 0.0724433
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0735467, upper bound: 0.0750928
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0728044, upper bound: 0.0728011
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.98
Output dim: 1, lower bound: -0.0735467, upper bound: 0.0754117

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0456167, 0.0096314, -0.0394707, 0.0095301, -0.0551468, 0.0491021
1: 0.9314801, 1.0133774, 0.9315279, 1.0027279, -0.0712478, 0.0818496
2: -0.0170025, 0.0387617, -0.0169739, 0.0297220, -0.0467245, 0.0557356
3: -0.0302700, 0.0082285, -0.0247587, 0.0082047, -0.0384746, 0.0329873
4: -0.0303801, 0.0233351, -0.0228805, 0.0232616, -0.0536418, 0.0462157
5: -0.0066731, 0.0556546, -0.0057086, 0.0475331, -0.0542063, 0.0613632
6: -0.0106081, 0.0216820, -0.0105018, 0.0207460, -0.0313541, 0.0321838
7: -0.0114593, 0.0778907, -0.0113665, 0.0659306, -0.0773899, 0.0892572
8: -0.0072323, 0.0326505, -0.0070815, 0.0326434, -0.0398758, 0.0397320
9: -0.0424935, 0.0234668, -0.0311504, 0.0234222, -0.0659157, 0.0546172

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0711650, upper bound: 0.0736771
time: 1.76 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0719482, upper bound: 0.0739882
time: 1.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0456123, 0.0096302, -0.0403866, 0.0097350, -0.0553473, 0.0500168
1: 0.9314824, 1.0133650, 0.9311408, 1.0032253, -0.0717429, 0.0822242
2: -0.0170012, 0.0387541, -0.0171862, 0.0313582, -0.0483594, 0.0559403
3: -0.0302657, 0.0082274, -0.0256553, 0.0083879, -0.0386536, 0.0338827
4: -0.0303737, 0.0233317, -0.0241808, 0.0238325, -0.0542062, 0.0475124
5: -0.0066714, 0.0556482, -0.0060420, 0.0489379, -0.0556093, 0.0616902
6: -0.0106056, 0.0216800, -0.0109126, 0.0210790, -0.0316846, 0.0325926
7: -0.0114583, 0.0778822, -0.0115338, 0.0677899, -0.0792482, 0.0894160
8: -0.0072282, 0.0326501, -0.0077628, 0.0327004, -0.0399286, 0.0404129
9: -0.0424854, 0.0234646, -0.0329359, 0.0237838, -0.0662692, 0.0564005

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0721314, upper bound: 0.0741281
time: 1.56 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0721121, upper bound: 0.0741046
time: 1.53 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0308967, 0.0090834, -0.0377010, 0.0094844, -0.0403811, 0.0467844
1: 0.9711965, 1.0018333, 0.9315735, 1.0030398, -0.0318434, 0.0404911
2: -0.0166483, 0.0174630, -0.0169481, 0.0269851, -0.0436334, 0.0344111
3: -0.0174898, 0.0079252, -0.0231603, 0.0081827, -0.0256724, 0.0310856
4: -0.0133526, 0.0223926, -0.0208789, 0.0231934, -0.0365460, 0.0432715
5: -0.0055930, 0.0353815, -0.0060356, 0.0448020, -0.0503950, 0.0414171
6: -0.0097803, 0.0114231, -0.0104375, 0.0211630, -0.0309433, 0.0218607
7: -0.0109835, 0.0491028, -0.0113263, 0.0621899, -0.0731734, 0.0604291
8: -0.0059309, 0.0168093, -0.0069822, 0.0326367, -0.0249104, 0.0237915
9: -0.0159573, 0.0034817, -0.0273761, 0.0233795, -0.0269075, 0.0308578

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728197
time: 2.22 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728140
time: 2.18 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0355272, 0.0092943, -0.0385228, 0.0095355, -0.0450628, 0.0478170
1: 0.9318772, 1.0025880, 0.9314978, 1.0031961, -0.0713189, 0.0710903
2: -0.0167806, 0.0236038, -0.0169901, 0.0282610, -0.0450415, 0.0405939
3: -0.0211649, 0.0080384, -0.0239140, 0.0082187, -0.0293836, 0.0319525
4: -0.0181956, 0.0227443, -0.0218875, 0.0233056, -0.0415012, 0.0446318
5: -0.0058228, 0.0416854, -0.0061137, 0.0459847, -0.0518074, 0.0477990
6: -0.0100937, 0.0209847, -0.0105260, 0.0212289, -0.0313227, 0.0315108
7: -0.0111671, 0.0578073, -0.0113696, 0.0638512, -0.0750183, 0.0691769
8: -0.0064219, 0.0325920, -0.0071253, 0.0326479, -0.0390697, 0.0397173
9: -0.0231673, 0.0230958, -0.0289817, 0.0234503, -0.0466176, 0.0520775

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728016, upper bound: 0.0733119
time: 1.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728016, upper bound: 0.0734531
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0308967, 0.0090834, -0.0470349, 0.0096622, -0.0405589, 0.0561183
1: 0.9711965, 1.0018333, 0.9314445, 1.0166695, -0.0454730, 0.0427891
2: -0.0166483, 0.0174630, -0.0170224, 0.0409026, -0.0575508, 0.0344854
3: -0.0174898, 0.0079252, -0.0315481, 0.0082455, -0.0257353, 0.0394733
4: -0.0133526, 0.0223926, -0.0321183, 0.0233880, -0.0367406, 0.0545109
5: -0.0055930, 0.0353815, -0.0068384, 0.0575949, -0.0631879, 0.0422200
6: -0.0097803, 0.0114231, -0.0106546, 0.0218288, -0.0316091, 0.0220777
7: -0.0109835, 0.0491028, -0.0114861, 0.0806811, -0.0916646, 0.0605889
8: -0.0059309, 0.0168093, -0.0073055, 0.0326557, -0.0264175, 0.0241147
9: -0.0159573, 0.0034817, -0.0451723, 0.0235000, -0.0270824, 0.0486541

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0718494, upper bound: 0.0719769
time: 1.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0744763, upper bound: 0.0725523
time: 1.88 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0355272, 0.0092943, -0.0478819, 0.0097072, -0.0452345, 0.0571762
1: 0.9318772, 1.0025880, 0.9313681, 1.0186549, -0.0867777, 0.0712199
2: -0.0167806, 0.0236038, -0.0170645, 0.0422405, -0.0590210, 0.0406683
3: -0.0211649, 0.0080384, -0.0323250, 0.0082818, -0.0294467, 0.0403634
4: -0.0181956, 0.0227443, -0.0331707, 0.0235009, -0.0416965, 0.0559150
5: -0.0058228, 0.0416854, -0.0069235, 0.0588182, -0.0646410, 0.0486089
6: -0.0100937, 0.0209847, -0.0107390, 0.0219004, -0.0319941, 0.0317238
7: -0.0111671, 0.0578073, -0.0115234, 0.0823833, -0.0935504, 0.0693308
8: -0.0064219, 0.0325920, -0.0074440, 0.0326669, -0.0390888, 0.0400359
9: -0.0231673, 0.0230958, -0.0468369, 0.0235714, -0.0467387, 0.0699327

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0720221, upper bound: 0.0726077
time: 2.12 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0747150, upper bound: 0.0732191
time: 2.13 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0456429, 0.0096354, -0.0385228, 0.0095355, -0.0551784, 0.0481582
1: 0.9314732, 1.0134418, 0.9314978, 1.0031961, -0.0717229, 0.0819440
2: -0.0170063, 0.0388032, -0.0169901, 0.0282610, -0.0452673, 0.0557932
3: -0.0302953, 0.0082317, -0.0239140, 0.0082187, -0.0385140, 0.0321458
4: -0.0304208, 0.0233452, -0.0218875, 0.0233056, -0.0537264, 0.0452327
5: -0.0066960, 0.0556837, -0.0061137, 0.0459847, -0.0526806, 0.0617973
6: -0.0106157, 0.0217068, -0.0105260, 0.0212289, -0.0318446, 0.0322329
7: -0.0114627, 0.0779374, -0.0113696, 0.0638512, -0.0753139, 0.0893070
8: -0.0072448, 0.0326515, -0.0071253, 0.0326479, -0.0398926, 0.0397768
9: -0.0425316, 0.0234732, -0.0289817, 0.0234503, -0.0659819, 0.0524549

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 191

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725303, upper bound: 0.0744958
time: 2.27 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0731940, upper bound: 0.0747207
time: 1.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0456429, 0.0096354, -0.0478819, 0.0097072, -0.0553501, 0.0575173
1: 0.9314732, 1.0134418, 0.9313681, 1.0186549, -0.0871817, 0.0820737
2: -0.0170063, 0.0388032, -0.0170645, 0.0422405, -0.0592468, 0.0558676
3: -0.0302953, 0.0082317, -0.0323250, 0.0082818, -0.0385770, 0.0405567
4: -0.0304208, 0.0233452, -0.0331707, 0.0235009, -0.0539217, 0.0565159
5: -0.0066960, 0.0556837, -0.0069235, 0.0588182, -0.0655142, 0.0626072
6: -0.0106157, 0.0217068, -0.0107390, 0.0219004, -0.0325161, 0.0324459
7: -0.0114627, 0.0779374, -0.0115234, 0.0823833, -0.0938460, 0.0894609
8: -0.0072448, 0.0326515, -0.0074440, 0.0326669, -0.0399117, 0.0400954
9: -0.0425316, 0.0234732, -0.0468369, 0.0235714, -0.0661030, 0.0703100

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727695, upper bound: 0.0748838
time: 1.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727695, upper bound: 0.0754117
time: 1.78 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.47 seconds
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0711650, upper bound: 0.0736771
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0719482, upper bound: 0.0739882
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0721314, upper bound: 0.0741281
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0721121, upper bound: 0.0741046
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728197
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728140
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0728016, upper bound: 0.0733119
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0728016, upper bound: 0.0734531
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0718494, upper bound: 0.0719769
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0744763, upper bound: 0.0725523
NS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0720221, upper bound: 0.0726077
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0747150, upper bound: 0.0732191
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0725303, upper bound: 0.0744958
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0731940, upper bound: 0.0747207
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0727695, upper bound: 0.0748838
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.47
Output dim: 1, lower bound: -0.0727695, upper bound: 0.0754117

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0447743, 0.0095856, -0.0323475, 0.0090902, -0.0538645, 0.0419330
1: 0.9315562, 1.0114256, 0.9321362, 1.0013850, -0.0698288, 0.0792894
2: -0.0169605, 0.0374313, -0.0166364, 0.0187601, -0.0357206, 0.0540677
3: -0.0294967, 0.0081923, -0.0182519, 0.0079148, -0.0374115, 0.0264442
4: -0.0293353, 0.0232226, -0.0142305, 0.0223598, -0.0516951, 0.0374531
5: -0.0065894, 0.0544394, -0.0051064, 0.0373466, -0.0439360, 0.0595458
6: -0.0105232, 0.0216122, -0.0097700, 0.0202635, -0.0307867, 0.0313822
7: -0.0114211, 0.0761936, -0.0109916, 0.0515266, -0.0629477, 0.0871852
8: -0.0070935, 0.0326392, -0.0059074, 0.0325539, -0.0396473, 0.0385466
9: -0.0408322, 0.0233957, -0.0172842, 0.0228539, -0.0636862, 0.0406798

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0669005, upper bound: 0.0670066
time: 1.97 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0709634, upper bound: 0.0734398
time: 2.10 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0456167, 0.0096314, -0.0366370, 0.0093065, -0.0549233, 0.0462684
1: 0.9314801, 1.0133774, 0.9318932, 1.0021647, -0.0706847, 0.0814843
2: -0.0170025, 0.0387617, -0.0167724, 0.0253097, -0.0423122, 0.0555341
3: -0.0302700, 0.0082285, -0.0221584, 0.0080312, -0.0383011, 0.0303869
4: -0.0303801, 0.0233351, -0.0193833, 0.0227216, -0.0531017, 0.0427184
5: -0.0066731, 0.0556546, -0.0054097, 0.0434593, -0.0501324, 0.0610643
6: -0.0106081, 0.0216820, -0.0100920, 0.0204950, -0.0311031, 0.0317739
7: -0.0114593, 0.0778907, -0.0111799, 0.0602134, -0.0716727, 0.0890706
8: -0.0072323, 0.0326505, -0.0064120, 0.0325896, -0.0398220, 0.0390624
9: -0.0424935, 0.0234668, -0.0256280, 0.0230809, -0.0655744, 0.0490948

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0676797, upper bound: 0.0674330
time: 2.19 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0717438, upper bound: 0.0737480
time: 2.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0456123, 0.0096302, -0.0372067, 0.0094460, -0.0550582, 0.0468369
1: 0.9314824, 1.0133650, 0.9316291, 1.0025569, -0.0710745, 0.0817360
2: -0.0170012, 0.0387541, -0.0169174, 0.0262582, -0.0432594, 0.0556715
3: -0.0302657, 0.0082274, -0.0226989, 0.0081563, -0.0384219, 0.0309263
4: -0.0303737, 0.0233317, -0.0201898, 0.0231112, -0.0534849, 0.0435215
5: -0.0066714, 0.0556482, -0.0056798, 0.0442308, -0.0509022, 0.0613281
6: -0.0106056, 0.0216800, -0.0103720, 0.0207731, -0.0313787, 0.0320520
7: -0.0114583, 0.0778822, -0.0112937, 0.0613004, -0.0727587, 0.0891759
8: -0.0072282, 0.0326501, -0.0068766, 0.0326285, -0.0398567, 0.0395267
9: -0.0424854, 0.0234646, -0.0266162, 0.0233277, -0.0658131, 0.0500808

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0674776, upper bound: 0.0672961
time: 1.37 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0719276, upper bound: 0.0738900
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0452784, 0.0095971, -0.0380705, 0.0094914, -0.0547697, 0.0476676
1: 0.9315428, 1.0125898, 0.9315721, 1.0030279, -0.0714852, 0.0810177
2: -0.0169680, 0.0382096, -0.0169491, 0.0275752, -0.0445432, 0.0551587
3: -0.0299545, 0.0081988, -0.0234963, 0.0081834, -0.0381380, 0.0316951
4: -0.0299500, 0.0232425, -0.0213115, 0.0231957, -0.0531457, 0.0445541
5: -0.0066316, 0.0551511, -0.0060031, 0.0453734, -0.0520050, 0.0611543
6: -0.0105407, 0.0216473, -0.0104436, 0.0211212, -0.0316619, 0.0320909
7: -0.0114312, 0.0772017, -0.0113328, 0.0629680, -0.0743992, 0.0885345
8: -0.0071210, 0.0326412, -0.0069901, 0.0326369, -0.0397579, 0.0396313
9: -0.0418177, 0.0234082, -0.0281614, 0.0233808, -0.0651985, 0.0515696

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0673871, upper bound: 0.0671599
time: 2.36 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0719052, upper bound: 0.0738666
time: 2.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0308200, 0.0090804, -0.0359282, 0.0093590, -0.0401790, 0.0450086
1: 0.9712104, 1.0017998, 0.9317555, 1.0024322, -0.0312218, 0.0399456
2: -0.0166448, 0.0174051, -0.0168473, 0.0241236, -0.0407684, 0.0342524
3: -0.0174550, 0.0079222, -0.0214783, 0.0080960, -0.0255510, 0.0294006
4: -0.0133015, 0.0223833, -0.0186012, 0.0229238, -0.0362252, 0.0409845
5: -0.0055693, 0.0353149, -0.0056693, 0.0421986, -0.0477679, 0.0409842
6: -0.0097738, 0.0113929, -0.0102231, 0.0207883, -0.0305621, 0.0216160
7: -0.0109811, 0.0489951, -0.0112201, 0.0585491, -0.0695302, 0.0602152
8: -0.0059200, 0.0168033, -0.0066364, 0.0326099, -0.0245843, 0.0234397
9: -0.0159334, 0.0034756, -0.0238782, 0.0232095, -0.0266050, 0.0273538

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 230

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728140
time: 1.98 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728140
time: 2.06 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0308967, 0.0090834, -0.0374412, 0.0094539, -0.0403506, 0.0465246
1: 0.9711965, 1.0018333, 0.9316247, 1.0028768, -0.0316803, 0.0404023
2: -0.0166483, 0.0174630, -0.0169199, 0.0265653, -0.0432135, 0.0343829
3: -0.0174898, 0.0079252, -0.0229152, 0.0081584, -0.0256481, 0.0308405
4: -0.0133526, 0.0223926, -0.0205265, 0.0231178, -0.0364703, 0.0429192
5: -0.0055930, 0.0353815, -0.0059239, 0.0444439, -0.0500369, 0.0413054
6: -0.0097803, 0.0114231, -0.0103807, 0.0210430, -0.0308233, 0.0218039
7: -0.0109835, 0.0491028, -0.0113010, 0.0616839, -0.0726675, 0.0604038
8: -0.0059309, 0.0168093, -0.0068892, 0.0326291, -0.0248652, 0.0236984
9: -0.0159573, 0.0034817, -0.0269057, 0.0233317, -0.0267821, 0.0303874

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 230

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728140
time: 1.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728140
time: 2.13 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0355272, 0.0092943, -0.0308967, 0.0090834, -0.0446106, 0.0401910
1: 0.9318772, 1.0025880, 0.9711965, 1.0018333, -0.0398392, 0.0313916
2: -0.0167806, 0.0236038, -0.0166483, 0.0174630, -0.0342435, 0.0402521
3: -0.0211649, 0.0080384, -0.0174898, 0.0079252, -0.0290901, 0.0255282
4: -0.0181956, 0.0227443, -0.0133526, 0.0223926, -0.0405882, 0.0360969
5: -0.0058228, 0.0416854, -0.0055930, 0.0353815, -0.0412043, 0.0472783
6: -0.0100937, 0.0209847, -0.0097803, 0.0114231, -0.0215169, 0.0307650
7: -0.0111671, 0.0578073, -0.0109835, 0.0491028, -0.0602699, 0.0687909
8: -0.0064219, 0.0325920, -0.0059309, 0.0168093, -0.0232312, 0.0245264
9: -0.0231673, 0.0230958, -0.0159573, 0.0034817, -0.0266490, 0.0263874

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728016, upper bound: 0.0733109
time: 2.04 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733109
time: 2.20 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0355272, 0.0092943, -0.0355272, 0.0092943, -0.0448215, 0.0448215
1: 0.9318772, 1.0025880, 0.9318772, 1.0025880, -0.0707108, 0.0707108
2: -0.0167806, 0.0236038, -0.0167806, 0.0236038, -0.0403844, 0.0403844
3: -0.0211649, 0.0080384, -0.0211649, 0.0080384, -0.0292033, 0.0292033
4: -0.0181956, 0.0227443, -0.0181956, 0.0227443, -0.0409399, 0.0409399
5: -0.0058228, 0.0416854, -0.0058228, 0.0416854, -0.0475081, 0.0475081
6: -0.0100937, 0.0209847, -0.0100937, 0.0209847, -0.0310785, 0.0310785
7: -0.0111671, 0.0578073, -0.0111671, 0.0578073, -0.0689745, 0.0689745
8: -0.0064219, 0.0325920, -0.0064219, 0.0325920, -0.0390139, 0.0390139
9: -0.0231673, 0.0230958, -0.0231673, 0.0230958, -0.0462631, 0.0462631

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728016, upper bound: 0.0734433
time: 1.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0734433
time: 3.10 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0308967, 0.0090834, -0.0448003, 0.0095896, -0.0404863, 0.0538837
1: 0.9711965, 1.0018333, 0.9315494, 1.0114902, -0.0402938, 0.0421515
2: -0.0166483, 0.0174630, -0.0169643, 0.0374727, -0.0541209, 0.0344273
3: -0.0174898, 0.0079252, -0.0295220, 0.0081956, -0.0256853, 0.0374472
4: -0.0133526, 0.0223926, -0.0293759, 0.0232327, -0.0365853, 0.0517685
5: -0.0055930, 0.0353815, -0.0066123, 0.0544685, -0.0600615, 0.0419938
6: -0.0097803, 0.0114231, -0.0105308, 0.0216372, -0.0314175, 0.0219539
7: -0.0109835, 0.0491028, -0.0114245, 0.0762401, -0.0872236, 0.0605273
8: -0.0059309, 0.0168093, -0.0071059, 0.0326402, -0.0260408, 0.0239151
9: -0.0159573, 0.0034817, -0.0408703, 0.0234020, -0.0269004, 0.0443520

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 230

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0744704, upper bound: 0.0725485
time: 3.50 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0744704, upper bound: 0.0725485
time: 2.10 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0355272, 0.0092943, -0.0456429, 0.0096354, -0.0451627, 0.0549371
1: 0.9318772, 1.0025880, 0.9314732, 1.0134418, -0.0815646, 0.0711148
2: -0.0167806, 0.0236038, -0.0170063, 0.0388032, -0.0555837, 0.0406101
3: -0.0211649, 0.0080384, -0.0302953, 0.0082317, -0.0293966, 0.0383337
4: -0.0181956, 0.0227443, -0.0304208, 0.0233452, -0.0415408, 0.0531651
5: -0.0058228, 0.0416854, -0.0066960, 0.0556837, -0.0615064, 0.0483813
6: -0.0100937, 0.0209847, -0.0106157, 0.0217068, -0.0318006, 0.0316004
7: -0.0111671, 0.0578073, -0.0114627, 0.0779374, -0.0891046, 0.0692700
8: -0.0064219, 0.0325920, -0.0072448, 0.0326515, -0.0390734, 0.0398367
9: -0.0231673, 0.0230958, -0.0425316, 0.0234732, -0.0466405, 0.0656274

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0746981, upper bound: 0.0732108
time: 2.11 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0746859, upper bound: 0.0732108
time: 1.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0448003, 0.0095896, -0.0308967, 0.0090834, -0.0538837, 0.0404863
1: 0.9315494, 1.0114902, 0.9711965, 1.0018333, -0.0421515, 0.0402938
2: -0.0169643, 0.0374727, -0.0166483, 0.0174630, -0.0344273, 0.0541209
3: -0.0295220, 0.0081956, -0.0174898, 0.0079252, -0.0374472, 0.0256853
4: -0.0293759, 0.0232327, -0.0133526, 0.0223926, -0.0517685, 0.0365853
5: -0.0066123, 0.0544685, -0.0055930, 0.0353815, -0.0419938, 0.0600615
6: -0.0105308, 0.0216372, -0.0097803, 0.0114231, -0.0219539, 0.0314175
7: -0.0114245, 0.0762401, -0.0109835, 0.0491028, -0.0605273, 0.0872236
8: -0.0071059, 0.0326402, -0.0059309, 0.0168093, -0.0239151, 0.0260408
9: -0.0408703, 0.0234020, -0.0159573, 0.0034817, -0.0443520, 0.0269004

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0688964, upper bound: 0.0687889
time: 2.15 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0722642, upper bound: 0.0742553
time: 1.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0456429, 0.0096354, -0.0355272, 0.0092943, -0.0549371, 0.0451627
1: 0.9314732, 1.0134418, 0.9318772, 1.0025880, -0.0711148, 0.0815646
2: -0.0170063, 0.0388032, -0.0167806, 0.0236038, -0.0406101, 0.0555837
3: -0.0302953, 0.0082317, -0.0211649, 0.0080384, -0.0383337, 0.0293966
4: -0.0304208, 0.0233452, -0.0181956, 0.0227443, -0.0531651, 0.0415408
5: -0.0066960, 0.0556837, -0.0058228, 0.0416854, -0.0483813, 0.0615064
6: -0.0106157, 0.0217068, -0.0100937, 0.0209847, -0.0316004, 0.0318006
7: -0.0114627, 0.0779374, -0.0111671, 0.0578073, -0.0692700, 0.0891046
8: -0.0072448, 0.0326515, -0.0064219, 0.0325920, -0.0398367, 0.0390734
9: -0.0425316, 0.0234732, -0.0231673, 0.0230958, -0.0656274, 0.0466405

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0697065, upper bound: 0.0691646
time: 2.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0729579, upper bound: 0.0744779
time: 1.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0456429, 0.0096354, -0.0388256, 0.0094859, -0.0551287, 0.0484611
1: 0.9314732, 1.0134418, 0.9316083, 1.0031979, -0.0717247, 0.0818335
2: -0.0170063, 0.0388032, -0.0169296, 0.0285527, -0.0455590, 0.0557328
3: -0.0302953, 0.0082317, -0.0241355, 0.0081665, -0.0384618, 0.0323673
4: -0.0304208, 0.0233452, -0.0221673, 0.0231428, -0.0535636, 0.0455125
5: -0.0066960, 0.0556837, -0.0061065, 0.0462690, -0.0529649, 0.0617901
6: -0.0106157, 0.0217068, -0.0104151, 0.0212217, -0.0318373, 0.0321220
7: -0.0114627, 0.0779374, -0.0113301, 0.0643545, -0.0758172, 0.0892676
8: -0.0072448, 0.0326515, -0.0069383, 0.0326316, -0.0398763, 0.0395898
9: -0.0425316, 0.0234732, -0.0294291, 0.0233469, -0.0658786, 0.0529022

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0690823, upper bound: 0.0690681
time: 1.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725571, upper bound: 0.0746474
time: 1.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0456429, 0.0096354, -0.0456429, 0.0096354, -0.0552783, 0.0552783
1: 0.9314732, 1.0134418, 0.9314732, 1.0134418, -0.0819686, 0.0819686
2: -0.0170063, 0.0388032, -0.0170063, 0.0388032, -0.0558094, 0.0558094
3: -0.0302953, 0.0082317, -0.0302953, 0.0082317, -0.0385270, 0.0385270
4: -0.0304208, 0.0233452, -0.0304208, 0.0233452, -0.0537660, 0.0537660
5: -0.0066960, 0.0556837, -0.0066960, 0.0556837, -0.0623796, 0.0623796
6: -0.0106157, 0.0217068, -0.0106157, 0.0217068, -0.0323225, 0.0323225
7: -0.0114627, 0.0779374, -0.0114627, 0.0779374, -0.0894001, 0.0894001
8: -0.0072448, 0.0326515, -0.0072448, 0.0326515, -0.0398962, 0.0398962
9: -0.0425316, 0.0234732, -0.0425316, 0.0234732, -0.0660048, 0.0660048

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 50

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0690823, upper bound: 0.0699224
time: 7.07 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725571, upper bound: 0.0751631
time: 1.95 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 11.23 seconds
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0669005, upper bound: 0.0670066
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0709634, upper bound: 0.0734398
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0676797, upper bound: 0.0674330
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0717438, upper bound: 0.0737480
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0674776, upper bound: 0.0672961
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0719276, upper bound: 0.0738900
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0673871, upper bound: 0.0671599
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0719052, upper bound: 0.0738666
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728140
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728140
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728140
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0732602, upper bound: 0.0728140
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0728016, upper bound: 0.0733109
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733109
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0728016, upper bound: 0.0734433
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0734433
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0744704, upper bound: 0.0725485
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0744704, upper bound: 0.0725485
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0746981, upper bound: 0.0732108
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0746859, upper bound: 0.0732108
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0688964, upper bound: 0.0687889
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0722642, upper bound: 0.0742553
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0697065, upper bound: 0.0691646
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0729579, upper bound: 0.0744779
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0690823, upper bound: 0.0690681
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0725571, upper bound: 0.0746474
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0690823, upper bound: 0.0699224
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 11.23
Output dim: 1, lower bound: -0.0725571, upper bound: 0.0751631

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0427394, 0.0095295, -0.0323475, 0.0090902, -0.0518297, 0.0418770
1: 0.9316259, 1.0069466, 0.9321362, 1.0013850, -0.0697591, 0.0748104
2: -0.0169217, 0.0343506, -0.0166364, 0.0187601, -0.0356818, 0.0509869
3: -0.0276545, 0.0081591, -0.0182519, 0.0079148, -0.0355693, 0.0264109
4: -0.0268527, 0.0231191, -0.0142305, 0.0223598, -0.0492125, 0.0373496
5: -0.0063942, 0.0516389, -0.0051064, 0.0373466, -0.0437409, 0.0567453
6: -0.0104353, 0.0214483, -0.0097700, 0.0202635, -0.0306988, 0.0312182
7: -0.0113728, 0.0721451, -0.0109916, 0.0515266, -0.0628994, 0.0831367
8: -0.0069540, 0.0326290, -0.0059074, 0.0325539, -0.0395079, 0.0385364
9: -0.0369283, 0.0233306, -0.0172842, 0.0228539, -0.0597823, 0.0406148

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0685123, upper bound: 0.0724002
time: 1.74 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0685123, upper bound: 0.0734398
time: 1.57 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0435790, 0.0095765, -0.0366370, 0.0093065, -0.0528855, 0.0462135
1: 0.9315496, 1.0087266, 0.9318932, 1.0021647, -0.0706151, 0.0768334
2: -0.0169637, 0.0356732, -0.0167724, 0.0253097, -0.0422734, 0.0524457
3: -0.0284241, 0.0081952, -0.0221584, 0.0080312, -0.0364553, 0.0303536
4: -0.0278910, 0.0232317, -0.0193833, 0.0227216, -0.0506126, 0.0426150
5: -0.0064749, 0.0528509, -0.0054097, 0.0434593, -0.0499342, 0.0582606
6: -0.0105210, 0.0215157, -0.0100920, 0.0204950, -0.0310159, 0.0316076
7: -0.0114120, 0.0738372, -0.0111799, 0.0602134, -0.0716254, 0.0850171
8: -0.0070939, 0.0326402, -0.0064120, 0.0325896, -0.0396835, 0.0390522
9: -0.0385843, 0.0234018, -0.0256280, 0.0230809, -0.0616652, 0.0490298

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0691901, upper bound: 0.0727561
time: 1.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0691901, upper bound: 0.0737480
time: 1.68 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0435745, 0.0095752, -0.0372067, 0.0094460, -0.0530204, 0.0467819
1: 0.9315521, 1.0087132, 0.9316291, 1.0025569, -0.0710049, 0.0770842
2: -0.0169624, 0.0356655, -0.0169174, 0.0262582, -0.0432206, 0.0525829
3: -0.0284197, 0.0081941, -0.0226989, 0.0081563, -0.0365760, 0.0308930
4: -0.0278844, 0.0232282, -0.0201898, 0.0231112, -0.0509956, 0.0434180
5: -0.0064731, 0.0528444, -0.0056798, 0.0442308, -0.0507039, 0.0585242
6: -0.0105184, 0.0215137, -0.0103720, 0.0207731, -0.0312916, 0.0318857
7: -0.0114110, 0.0738284, -0.0112937, 0.0613004, -0.0727114, 0.0851221
8: -0.0070897, 0.0326398, -0.0068766, 0.0326285, -0.0397183, 0.0395164
9: -0.0385761, 0.0233995, -0.0266162, 0.0233277, -0.0619038, 0.0500157

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0692856, upper bound: 0.0728726
time: 1.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0692856, upper bound: 0.0738900
time: 2.54 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0432447, 0.0095417, -0.0380705, 0.0094914, -0.0527361, 0.0476122
1: 0.9316127, 1.0079864, 0.9315721, 1.0030279, -0.0714152, 0.0764143
2: -0.0169291, 0.0351261, -0.0169491, 0.0275752, -0.0445043, 0.0520751
3: -0.0281128, 0.0081654, -0.0234963, 0.0081834, -0.0362962, 0.0316617
4: -0.0274663, 0.0231388, -0.0213115, 0.0231957, -0.0506620, 0.0444504
5: -0.0064352, 0.0523513, -0.0060031, 0.0453734, -0.0518086, 0.0583544
6: -0.0104531, 0.0214823, -0.0104436, 0.0211212, -0.0315743, 0.0319258
7: -0.0113835, 0.0731557, -0.0113328, 0.0629680, -0.0743515, 0.0844885
8: -0.0069819, 0.0326309, -0.0069901, 0.0326369, -0.0396188, 0.0396211
9: -0.0379149, 0.0233429, -0.0281614, 0.0233808, -0.0612957, 0.0515043

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0691559, upper bound: 0.0727898
time: 1.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0691559, upper bound: 0.0738666
time: 1.79 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0269109, 0.0089970, -0.0359282, 0.0093590, -0.0362699, 0.0449253
1: 0.9717264, 1.0011108, 0.9321359, 1.0024322, -0.0307059, 0.0395904
2: -0.0165492, 0.0147392, -0.0168473, 0.0241236, -0.0406728, 0.0315865
3: -0.0158362, 0.0078397, -0.0214783, 0.0080960, -0.0239322, 0.0293180
4: -0.0111144, 0.0221261, -0.0186012, 0.0229238, -0.0340382, 0.0407273
5: -0.0052198, 0.0318536, -0.0056693, 0.0421986, -0.0474184, 0.0375229
6: -0.0095951, 0.0103872, -0.0102231, 0.0207883, -0.0303834, 0.0206104
7: -0.0109142, 0.0434356, -0.0112201, 0.0585491, -0.0694633, 0.0546557
8: -0.0056206, 0.0165341, -0.0066364, 0.0326099, -0.0242685, 0.0231705
9: -0.0148018, 0.0033068, -0.0238782, 0.0232095, -0.0254371, 0.0271850

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0727955
time: 3.25 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0728197
time: 1.66 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0303371, 0.0090587, -0.0359282, 0.0093590, -0.0396961, 0.0449870
1: 0.9713005, 1.0016538, 0.9317555, 1.0024322, -0.0311317, 0.0397426
2: -0.0166200, 0.0170489, -0.0168473, 0.0241236, -0.0407436, 0.0338962
3: -0.0172441, 0.0079008, -0.0214783, 0.0080960, -0.0253401, 0.0293791
4: -0.0129984, 0.0223165, -0.0186012, 0.0229238, -0.0359222, 0.0409177
5: -0.0054733, 0.0348890, -0.0056693, 0.0421986, -0.0476719, 0.0405583
6: -0.0097274, 0.0112310, -0.0102231, 0.0207883, -0.0305157, 0.0214541
7: -0.0109637, 0.0483148, -0.0112201, 0.0585491, -0.0695128, 0.0595348
8: -0.0058423, 0.0167663, -0.0066364, 0.0326099, -0.0244919, 0.0234026
9: -0.0157837, 0.0034317, -0.0238782, 0.0232095, -0.0264561, 0.0273100

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0727955
time: 4.21 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0728197
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0269109, 0.0089970, -0.0374412, 0.0094539, -0.0363648, 0.0464382
1: 0.9717264, 1.0011108, 0.9317011, 1.0028768, -0.0311504, 0.0400253
2: -0.0165492, 0.0147392, -0.0169199, 0.0265653, -0.0431145, 0.0316591
3: -0.0158362, 0.0078397, -0.0229152, 0.0081584, -0.0239945, 0.0307549
4: -0.0111144, 0.0221261, -0.0205265, 0.0231178, -0.0342322, 0.0426526
5: -0.0052198, 0.0318536, -0.0059239, 0.0444439, -0.0496638, 0.0377775
6: -0.0095951, 0.0103872, -0.0103807, 0.0210430, -0.0306381, 0.0207680
7: -0.0109142, 0.0434356, -0.0113010, 0.0616839, -0.0725982, 0.0547366
8: -0.0056206, 0.0165341, -0.0068892, 0.0326291, -0.0245374, 0.0234233
9: -0.0148018, 0.0033068, -0.0269057, 0.0233317, -0.0255728, 0.0302125

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0727929
time: 1.51 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0728140
time: 1.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0303371, 0.0090587, -0.0374412, 0.0094539, -0.0397910, 0.0464999
1: 0.9713005, 1.0016538, 0.9316247, 1.0028768, -0.0315763, 0.0402014
2: -0.0166200, 0.0170489, -0.0169199, 0.0265653, -0.0431853, 0.0339688
3: -0.0172441, 0.0079008, -0.0229152, 0.0081584, -0.0254025, 0.0308160
4: -0.0129984, 0.0223165, -0.0205265, 0.0231178, -0.0361162, 0.0428430
5: -0.0054733, 0.0348890, -0.0059239, 0.0444439, -0.0499172, 0.0408129
6: -0.0097274, 0.0112310, -0.0103807, 0.0210430, -0.0307704, 0.0216117
7: -0.0109637, 0.0483148, -0.0113010, 0.0616839, -0.0726477, 0.0596157
8: -0.0058423, 0.0167663, -0.0068892, 0.0326291, -0.0247656, 0.0236554
9: -0.0157837, 0.0034317, -0.0269057, 0.0233317, -0.0266103, 0.0303375

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0727929
time: 2.43 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0728140
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0336734, 0.0091617, -0.0308200, 0.0090804, -0.0427538, 0.0399817
1: 0.9320573, 1.0019317, 0.9712104, 1.0017998, -0.0392802, 0.0307212
2: -0.0166806, 0.0206440, -0.0166448, 0.0174051, -0.0340857, 0.0372888
3: -0.0194114, 0.0079526, -0.0174550, 0.0079222, -0.0273337, 0.0254075
4: -0.0158272, 0.0224773, -0.0133015, 0.0223833, -0.0382105, 0.0357787
5: -0.0054440, 0.0390039, -0.0055693, 0.0353149, -0.0407589, 0.0445732
6: -0.0098754, 0.0205992, -0.0097738, 0.0113929, -0.0212683, 0.0303730
7: -0.0110539, 0.0540043, -0.0109811, 0.0489951, -0.0600490, 0.0649854
8: -0.0060723, 0.0325655, -0.0059200, 0.0168033, -0.0228756, 0.0241873
9: -0.0195398, 0.0229276, -0.0159334, 0.0034756, -0.0230154, 0.0260841

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 230

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733109
time: 1.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733109
time: 1.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0352645, 0.0092633, -0.0308967, 0.0090834, -0.0443479, 0.0401600
1: 0.9319286, 1.0024198, 0.9711965, 1.0018333, -0.0397495, 0.0312234
2: -0.0167522, 0.0231811, -0.0166483, 0.0174630, -0.0342152, 0.0398294
3: -0.0209170, 0.0080140, -0.0174898, 0.0079252, -0.0288422, 0.0255038
4: -0.0178405, 0.0226684, -0.0133526, 0.0223926, -0.0402331, 0.0360210
5: -0.0057081, 0.0413246, -0.0055930, 0.0353815, -0.0410896, 0.0469176
6: -0.0100365, 0.0208619, -0.0097803, 0.0114231, -0.0214596, 0.0306422
7: -0.0111413, 0.0572950, -0.0109835, 0.0491028, -0.0602441, 0.0682785
8: -0.0063281, 0.0325844, -0.0059309, 0.0168093, -0.0231374, 0.0244808
9: -0.0226925, 0.0230478, -0.0159573, 0.0034817, -0.0261742, 0.0262619

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 230

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733109
time: 2.06 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733108
time: 3.19 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0336734, 0.0091617, -0.0354909, 0.0092903, -0.0429637, 0.0446526
1: 0.9320573, 1.0019317, 0.9318835, 1.0025562, -0.0704989, 0.0700482
2: -0.0166806, 0.0206440, -0.0167771, 0.0235456, -0.0402262, 0.0374211
3: -0.0194114, 0.0079526, -0.0211304, 0.0080354, -0.0274469, 0.0290830
4: -0.0158272, 0.0224773, -0.0181449, 0.0227349, -0.0385622, 0.0406222
5: -0.0054440, 0.0390039, -0.0058003, 0.0416374, -0.0470815, 0.0448042
6: -0.0098754, 0.0205992, -0.0100866, 0.0209598, -0.0308352, 0.0306858
7: -0.0110539, 0.0540043, -0.0111638, 0.0577381, -0.0687920, 0.0651681
8: -0.0060723, 0.0325655, -0.0064102, 0.0325910, -0.0386633, 0.0389757
9: -0.0195398, 0.0229276, -0.0231046, 0.0230899, -0.0426297, 0.0460322

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734162, upper bound: 0.0734433
time: 1.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734162, upper bound: 0.0734433
time: 1.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0352645, 0.0092633, -0.0355272, 0.0092943, -0.0445587, 0.0447906
1: 0.9319286, 1.0024198, 0.9318772, 1.0025880, -0.0706594, 0.0705426
2: -0.0167522, 0.0231811, -0.0167806, 0.0236038, -0.0403561, 0.0399617
3: -0.0209170, 0.0080140, -0.0211649, 0.0080384, -0.0289554, 0.0291789
4: -0.0178405, 0.0226684, -0.0181956, 0.0227443, -0.0405848, 0.0408639
5: -0.0057081, 0.0413246, -0.0058228, 0.0416854, -0.0473934, 0.0471474
6: -0.0100365, 0.0208619, -0.0100937, 0.0209847, -0.0310212, 0.0309557
7: -0.0111413, 0.0572950, -0.0111671, 0.0578073, -0.0689487, 0.0684621
8: -0.0063281, 0.0325844, -0.0064219, 0.0325920, -0.0389201, 0.0390063
9: -0.0226925, 0.0230478, -0.0231673, 0.0230958, -0.0457883, 0.0462151

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734162, upper bound: 0.0734433
time: 1.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0734162, upper bound: 0.0734433
time: 2.36 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0269109, 0.0089970, -0.0447648, 0.0095857, -0.0364965, 0.0537618
1: 0.9717264, 1.0011108, 0.9315557, 1.0114057, -0.0396793, 0.0417619
2: -0.0165492, 0.0147392, -0.0169608, 0.0374144, -0.0539637, 0.0317000
3: -0.0158362, 0.0078397, -0.0294880, 0.0081926, -0.0240287, 0.0373277
4: -0.0111144, 0.0221261, -0.0293253, 0.0232233, -0.0343378, 0.0514514
5: -0.0052198, 0.0318536, -0.0065932, 0.0544202, -0.0596400, 0.0384468
6: -0.0095951, 0.0103872, -0.0105237, 0.0216150, -0.0312102, 0.0209109
7: -0.0109142, 0.0434356, -0.0114212, 0.0761719, -0.0870861, 0.0548568
8: -0.0056206, 0.0165341, -0.0070942, 0.0326393, -0.0257070, 0.0236283
9: -0.0148018, 0.0033068, -0.0408078, 0.0233961, -0.0256950, 0.0441146

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0685655, upper bound: 0.0688760
time: 1.62 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0742333, upper bound: 0.0722830
time: 1.99 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0303371, 0.0090587, -0.0448003, 0.0095896, -0.0399267, 0.0538591
1: 0.9713005, 1.0016538, 0.9315494, 1.0114902, -0.0401897, 0.0419562
2: -0.0166200, 0.0170489, -0.0169643, 0.0374727, -0.0540927, 0.0340132
3: -0.0172441, 0.0079008, -0.0295220, 0.0081956, -0.0254397, 0.0374228
4: -0.0129984, 0.0223165, -0.0293759, 0.0232327, -0.0362311, 0.0516923
5: -0.0054733, 0.0348890, -0.0066123, 0.0544685, -0.0599417, 0.0415013
6: -0.0097274, 0.0112310, -0.0105308, 0.0216372, -0.0313646, 0.0217618
7: -0.0109637, 0.0483148, -0.0114245, 0.0762401, -0.0872038, 0.0597392
8: -0.0058423, 0.0167663, -0.0071059, 0.0326402, -0.0259414, 0.0238721
9: -0.0157837, 0.0034317, -0.0408703, 0.0234020, -0.0267286, 0.0443020

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0685634, upper bound: 0.0688760
time: 1.42 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0742333, upper bound: 0.0722830
time: 1.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0336734, 0.0091617, -0.0456074, 0.0096315, -0.0433049, 0.0547691
1: 0.9320573, 1.0019317, 0.9314796, 1.0133573, -0.0813000, 0.0704520
2: -0.0166806, 0.0206440, -0.0170028, 0.0387449, -0.0554255, 0.0376468
3: -0.0194114, 0.0079526, -0.0302614, 0.0082287, -0.0276402, 0.0382139
4: -0.0158272, 0.0224773, -0.0303703, 0.0233359, -0.0391631, 0.0528476
5: -0.0054440, 0.0390039, -0.0066772, 0.0556353, -0.0610793, 0.0456811
6: -0.0098754, 0.0205992, -0.0106085, 0.0216849, -0.0315603, 0.0312077
7: -0.0110539, 0.0540043, -0.0114594, 0.0778692, -0.0889232, 0.0654636
8: -0.0060723, 0.0325655, -0.0072331, 0.0326505, -0.0387228, 0.0397986
9: -0.0195398, 0.0229276, -0.0424690, 0.0234673, -0.0430071, 0.0653966

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0689538, upper bound: 0.0696279
time: 1.98 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0744586, upper bound: 0.0729750
time: 1.36 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0352645, 0.0092633, -0.0456429, 0.0096354, -0.0448999, 0.0549062
1: 0.9319286, 1.0024198, 0.9314732, 1.0134418, -0.0815132, 0.0709466
2: -0.0167522, 0.0231811, -0.0170063, 0.0388032, -0.0555554, 0.0401874
3: -0.0209170, 0.0080140, -0.0302953, 0.0082317, -0.0291488, 0.0383093
4: -0.0178405, 0.0226684, -0.0304208, 0.0233452, -0.0411857, 0.0530892
5: -0.0057081, 0.0413246, -0.0066960, 0.0556837, -0.0613917, 0.0480206
6: -0.0100365, 0.0208619, -0.0106157, 0.0217068, -0.0317433, 0.0314776
7: -0.0111413, 0.0572950, -0.0114627, 0.0779374, -0.0890788, 0.0687576
8: -0.0063281, 0.0325844, -0.0072448, 0.0326515, -0.0389796, 0.0398292
9: -0.0226925, 0.0230478, -0.0425316, 0.0234732, -0.0461657, 0.0655794

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0689318, upper bound: 0.0696231
time: 2.02 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0744462, upper bound: 0.0729750
time: 1.84 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0427655, 0.0095335, -0.0308967, 0.0090834, -0.0518489, 0.0404302
1: 0.9316190, 1.0070096, 0.9711965, 1.0018333, -0.0415879, 0.0358132
2: -0.0169255, 0.0343917, -0.0166483, 0.0174630, -0.0343885, 0.0510400
3: -0.0276797, 0.0081623, -0.0174898, 0.0079252, -0.0356049, 0.0256521
4: -0.0268931, 0.0231293, -0.0133526, 0.0223926, -0.0492857, 0.0364818
5: -0.0064173, 0.0516682, -0.0055930, 0.0353815, -0.0417989, 0.0572612
6: -0.0104429, 0.0214734, -0.0097803, 0.0114231, -0.0218660, 0.0312537
7: -0.0113761, 0.0721917, -0.0109835, 0.0491028, -0.0604789, 0.0831752
8: -0.0069664, 0.0326300, -0.0059309, 0.0168093, -0.0237757, 0.0257085
9: -0.0369665, 0.0233370, -0.0159573, 0.0034817, -0.0404482, 0.0267841

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 230

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0722587, upper bound: 0.0742508
time: 1.49 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0722587, upper bound: 0.0742507
time: 3.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0436051, 0.0095805, -0.0355272, 0.0092943, -0.0528993, 0.0451077
1: 0.9315428, 1.0087911, 0.9318772, 1.0025880, -0.0710452, 0.0769139
2: -0.0169675, 0.0357144, -0.0167806, 0.0236038, -0.0405713, 0.0524950
3: -0.0284492, 0.0081985, -0.0211649, 0.0080384, -0.0364877, 0.0293634
4: -0.0279314, 0.0232419, -0.0181956, 0.0227443, -0.0506757, 0.0414374
5: -0.0064979, 0.0528800, -0.0058228, 0.0416854, -0.0481833, 0.0587028
6: -0.0105285, 0.0215407, -0.0100937, 0.0209847, -0.0315133, 0.0316345
7: -0.0114154, 0.0738837, -0.0111671, 0.0578073, -0.0692227, 0.0850508
8: -0.0071063, 0.0326412, -0.0064219, 0.0325920, -0.0396983, 0.0390631
9: -0.0386224, 0.0234081, -0.0231673, 0.0230958, -0.0617182, 0.0465754

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0729461, upper bound: 0.0744647
time: 1.39 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0729461, upper bound: 0.0744539
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0436051, 0.0095805, -0.0388256, 0.0094859, -0.0530909, 0.0484061
1: 0.9315428, 1.0087911, 0.9316083, 1.0031979, -0.0716551, 0.0771828
2: -0.0169675, 0.0357144, -0.0169296, 0.0285527, -0.0455202, 0.0526441
3: -0.0284492, 0.0081985, -0.0241355, 0.0081665, -0.0366157, 0.0323340
4: -0.0279314, 0.0232419, -0.0221673, 0.0231428, -0.0510742, 0.0454091
5: -0.0064979, 0.0528800, -0.0061065, 0.0462690, -0.0527669, 0.0589865
6: -0.0105285, 0.0215407, -0.0104151, 0.0212217, -0.0317502, 0.0319558
7: -0.0114154, 0.0738837, -0.0113301, 0.0643545, -0.0757699, 0.0852138
8: -0.0071063, 0.0326412, -0.0069383, 0.0326316, -0.0397379, 0.0395795
9: -0.0386224, 0.0234081, -0.0294291, 0.0233469, -0.0619693, 0.0528372

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725197, upper bound: 0.0745831
time: 2.12 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0724428, upper bound: 0.0745738
time: 1.45 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0436051, 0.0095805, -0.0456429, 0.0096354, -0.0532405, 0.0552234
1: 0.9315428, 1.0087911, 0.9314732, 1.0134418, -0.0818990, 0.0773179
2: -0.0169675, 0.0357144, -0.0170063, 0.0388032, -0.0557707, 0.0527207
3: -0.0284492, 0.0081985, -0.0302953, 0.0082317, -0.0366810, 0.0384938
4: -0.0279314, 0.0232419, -0.0304208, 0.0233452, -0.0512767, 0.0536626
5: -0.0064979, 0.0528800, -0.0066960, 0.0556837, -0.0621816, 0.0595760
6: -0.0105285, 0.0215407, -0.0106157, 0.0217068, -0.0322354, 0.0321564
7: -0.0114154, 0.0738837, -0.0114627, 0.0779374, -0.0893528, 0.0853463
8: -0.0071063, 0.0326412, -0.0072448, 0.0326515, -0.0397578, 0.0398860
9: -0.0386224, 0.0234081, -0.0425316, 0.0234732, -0.0620956, 0.0659398

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0698233, upper bound: 0.0724945
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0698233, upper bound: 0.0751631
time: 1.63 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 4.69 seconds
NS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0685123, upper bound: 0.0724002
NS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0685123, upper bound: 0.0734398
NS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0691901, upper bound: 0.0727561
NS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0691901, upper bound: 0.0737480
NS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0692856, upper bound: 0.0728726
NS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0692856, upper bound: 0.0738900
NS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0691559, upper bound: 0.0727898
NS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0691559, upper bound: 0.0738666
NS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0727955
NS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0728197
NS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0727955
NS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0728197
NS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0727929
NS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0728140
NS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0727929
NS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727598, upper bound: 0.0728140
NS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733109
NS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733109
NS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733109
NS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0727931, upper bound: 0.0733108
NS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0734162, upper bound: 0.0734433
NS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0734162, upper bound: 0.0734433
NS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0734162, upper bound: 0.0734433
NS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0734162, upper bound: 0.0734433
NS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0685655, upper bound: 0.0688760
NS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0742333, upper bound: 0.0722830
NS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0685634, upper bound: 0.0688760
NS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0742333, upper bound: 0.0722830
NS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0689538, upper bound: 0.0696279
NS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0744586, upper bound: 0.0729750
NS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0689318, upper bound: 0.0696231
NS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0744462, upper bound: 0.0729750
NS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0722587, upper bound: 0.0742508
NS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0722587, upper bound: 0.0742507
NS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0729461, upper bound: 0.0744647
NS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0729461, upper bound: 0.0744539
NS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0725197, upper bound: 0.0745831
NS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0724428, upper bound: 0.0745738
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0698233, upper bound: 0.0724945
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.69
Output dim: 1, lower bound: -0.0698233, upper bound: 0.0751631

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0427394, 0.0095295, -0.0241894, 0.0089902, -0.0517296, 0.0337189
1: 0.9316259, 1.0069466, 0.9719079, 1.0003613, -0.0405000, 0.0350387
2: -0.0169217, 0.0343506, -0.0165414, 0.0131121, -0.0300338, 0.0508920
3: -0.0276545, 0.0081591, -0.0147422, 0.0078329, -0.0354874, 0.0229012
4: -0.0268527, 0.0231191, -0.0095935, 0.0221050, -0.0489577, 0.0327127
5: -0.0063942, 0.0516389, -0.0047720, 0.0297267, -0.0361209, 0.0564109
6: -0.0104353, 0.0214483, -0.0095805, 0.0094690, -0.0199043, 0.0310288
7: -0.0113728, 0.0721451, -0.0109087, 0.0397796, -0.0511524, 0.0830538
8: -0.0069540, 0.0326290, -0.0055961, 0.0163843, -0.0233384, 0.0253029
9: -0.0369283, 0.0233306, -0.0141641, 0.0032930, -0.0402213, 0.0248574

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0682410, upper bound: 0.0727889
time: 1.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0684574, upper bound: 0.0733530
time: 1.45 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0435790, 0.0095765, -0.0328123, 0.0091396, -0.0527187, 0.0423887
1: 0.9315496, 1.0087266, 0.9320630, 1.0013683, -0.0698187, 0.0766636
2: -0.0169637, 0.0356732, -0.0166768, 0.0195860, -0.0365497, 0.0523500
3: -0.0284241, 0.0081952, -0.0186993, 0.0079495, -0.0363736, 0.0268945
4: -0.0278910, 0.0232317, -0.0148039, 0.0224680, -0.0503590, 0.0380357
5: -0.0064749, 0.0528509, -0.0050383, 0.0381370, -0.0446120, 0.0578892
6: -0.0105210, 0.0215157, -0.0098554, 0.0201788, -0.0306998, 0.0313710
7: -0.0114120, 0.0738372, -0.0110334, 0.0525543, -0.0639663, 0.0848706
8: -0.0070939, 0.0326402, -0.0060454, 0.0325646, -0.0396585, 0.0386856
9: -0.0385843, 0.0234018, -0.0183132, 0.0229222, -0.0615065, 0.0417149

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0689395, upper bound: 0.0732479
time: 1.49 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0691346, upper bound: 0.0736857
time: 1.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0435745, 0.0095752, -0.0333982, 0.0092794, -0.0528539, 0.0429734
1: 0.9315521, 1.0087132, 0.9318015, 1.0017757, -0.0702237, 0.0769117
2: -0.0169624, 0.0356655, -0.0168203, 0.0205473, -0.0375098, 0.0524858
3: -0.0284197, 0.0081941, -0.0192583, 0.0080734, -0.0364931, 0.0274524
4: -0.0278844, 0.0232282, -0.0156310, 0.0228537, -0.0507381, 0.0388592
5: -0.0064731, 0.0528444, -0.0053172, 0.0389260, -0.0453991, 0.0581616
6: -0.0105184, 0.0215137, -0.0101339, 0.0204658, -0.0309843, 0.0316475
7: -0.0114110, 0.0738284, -0.0111477, 0.0536812, -0.0650922, 0.0849761
8: -0.0070897, 0.0326398, -0.0065069, 0.0326031, -0.0396928, 0.0391467
9: -0.0385761, 0.0233995, -0.0193263, 0.0231665, -0.0617426, 0.0427259

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0690316, upper bound: 0.0721906
time: 1.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0692299, upper bound: 0.0738251
time: 1.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0432447, 0.0095417, -0.0342160, 0.0093283, -0.0525730, 0.0437577
1: 0.9316127, 1.0079864, 0.9317431, 1.0022607, -0.0706480, 0.0762433
2: -0.0169291, 0.0351261, -0.0168530, 0.0217711, -0.0387003, 0.0519790
3: -0.0281128, 0.0081654, -0.0200045, 0.0081013, -0.0362141, 0.0281699
4: -0.0274663, 0.0231388, -0.0166769, 0.0229406, -0.0504069, 0.0398157
5: -0.0064352, 0.0523513, -0.0056413, 0.0399819, -0.0464171, 0.0579925
6: -0.0104531, 0.0214823, -0.0102090, 0.0208153, -0.0312684, 0.0316913
7: -0.0113835, 0.0731557, -0.0111900, 0.0552474, -0.0666308, 0.0843457
8: -0.0069819, 0.0326309, -0.0066255, 0.0326117, -0.0395936, 0.0392564
9: -0.0379149, 0.0233429, -0.0207529, 0.0232211, -0.0611360, 0.0440958

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0689119, upper bound: 0.0733455
time: 2.25 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0690998, upper bound: 0.0738000
time: 2.45 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0336734, 0.0091617, -0.0269109, 0.0089970, -0.0426704, 0.0360726
1: 0.9328014, 1.0019317, 0.9717264, 1.0011108, -0.0389250, 0.0302053
2: -0.0166806, 0.0206440, -0.0165492, 0.0147392, -0.0314197, 0.0371932
3: -0.0194114, 0.0079526, -0.0158362, 0.0078397, -0.0272511, 0.0237887
4: -0.0158272, 0.0224773, -0.0111144, 0.0221261, -0.0379533, 0.0335917
5: -0.0054440, 0.0390039, -0.0052198, 0.0318536, -0.0372976, 0.0442238
6: -0.0098754, 0.0205992, -0.0095951, 0.0103872, -0.0202626, 0.0301943
7: -0.0110539, 0.0540043, -0.0109142, 0.0434356, -0.0544896, 0.0649185
8: -0.0060723, 0.0325655, -0.0056206, 0.0165341, -0.0226064, 0.0238715
9: -0.0195398, 0.0229276, -0.0148018, 0.0033068, -0.0228466, 0.0249161

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717606, upper bound: 0.0701243
time: 2.14 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725097, upper bound: 0.0730516
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0336734, 0.0091617, -0.0303371, 0.0090587, -0.0427321, 0.0394988
1: 0.9322233, 1.0019317, 0.9713005, 1.0016538, -0.0390772, 0.0306312
2: -0.0166806, 0.0206440, -0.0166200, 0.0170489, -0.0337295, 0.0372640
3: -0.0194114, 0.0079526, -0.0172441, 0.0079008, -0.0273122, 0.0251967
4: -0.0158272, 0.0224773, -0.0129984, 0.0223165, -0.0381437, 0.0354757
5: -0.0054440, 0.0390039, -0.0054733, 0.0348890, -0.0403330, 0.0444772
6: -0.0098754, 0.0205992, -0.0097274, 0.0112310, -0.0211064, 0.0303266
7: -0.0110539, 0.0540043, -0.0109637, 0.0483148, -0.0593687, 0.0649680
8: -0.0060723, 0.0325655, -0.0058423, 0.0167663, -0.0228386, 0.0240949
9: -0.0195398, 0.0229276, -0.0157837, 0.0034317, -0.0229715, 0.0259352

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717606, upper bound: 0.0701243
time: 2.64 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725097, upper bound: 0.0730516
time: 1.40 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0352645, 0.0092633, -0.0269109, 0.0089970, -0.0442615, 0.0361742
1: 0.9323539, 1.0024198, 0.9717264, 1.0011108, -0.0393725, 0.0306935
2: -0.0167522, 0.0231811, -0.0165492, 0.0147392, -0.0314914, 0.0397304
3: -0.0209170, 0.0080140, -0.0158362, 0.0078397, -0.0287567, 0.0238502
4: -0.0178405, 0.0226684, -0.0111144, 0.0221261, -0.0399666, 0.0337828
5: -0.0057081, 0.0413246, -0.0052198, 0.0318536, -0.0375617, 0.0465444
6: -0.0100365, 0.0208619, -0.0095951, 0.0103872, -0.0204237, 0.0304571
7: -0.0111413, 0.0572950, -0.0109142, 0.0434356, -0.0545770, 0.0682092
8: -0.0063281, 0.0325844, -0.0056206, 0.0165341, -0.0228623, 0.0241530
9: -0.0226925, 0.0230478, -0.0148018, 0.0033068, -0.0259993, 0.0250525

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717606, upper bound: 0.0701243
time: 1.94 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725029, upper bound: 0.0730516
time: 1.61 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0352645, 0.0092633, -0.0303371, 0.0090587, -0.0443232, 0.0396004
1: 0.9319286, 1.0024198, 0.9713005, 1.0016538, -0.0395546, 0.0311193
2: -0.0167522, 0.0231811, -0.0166200, 0.0170489, -0.0338011, 0.0398011
3: -0.0209170, 0.0080140, -0.0172441, 0.0079008, -0.0288178, 0.0252581
4: -0.0178405, 0.0226684, -0.0129984, 0.0223165, -0.0401570, 0.0356668
5: -0.0057081, 0.0413246, -0.0054733, 0.0348890, -0.0405971, 0.0467979
6: -0.0100365, 0.0208619, -0.0097274, 0.0112310, -0.0212674, 0.0305893
7: -0.0111413, 0.0572950, -0.0109637, 0.0483148, -0.0594561, 0.0682587
8: -0.0063281, 0.0325844, -0.0058423, 0.0167663, -0.0230944, 0.0243811
9: -0.0226925, 0.0230478, -0.0157837, 0.0034317, -0.0261242, 0.0260901

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717606, upper bound: 0.0701243
time: 1.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0725029, upper bound: 0.0730516
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0336734, 0.0091617, -0.0336734, 0.0091617, -0.0428351, 0.0428351
1: 0.9320573, 1.0019317, 0.9320573, 1.0019317, -0.0698743, 0.0698743
2: -0.0166806, 0.0206440, -0.0166806, 0.0206440, -0.0373246, 0.0373246
3: -0.0194114, 0.0079526, -0.0194114, 0.0079526, -0.0273640, 0.0273640
4: -0.0158272, 0.0224773, -0.0158272, 0.0224773, -0.0383045, 0.0383045
5: -0.0054440, 0.0390039, -0.0054440, 0.0390039, -0.0444480, 0.0444480
6: -0.0098754, 0.0205992, -0.0098754, 0.0205992, -0.0304746, 0.0304746
7: -0.0110539, 0.0540043, -0.0110539, 0.0540043, -0.0650582, 0.0650582
8: -0.0060723, 0.0325655, -0.0060723, 0.0325655, -0.0386377, 0.0386377
9: -0.0195398, 0.0229276, -0.0195398, 0.0229276, -0.0424674, 0.0424674

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0723066, upper bound: 0.0702436
time: 1.80 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0731746, upper bound: 0.0731873
time: 1.85 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0336734, 0.0091617, -0.0352645, 0.0092633, -0.0429367, 0.0444262
1: 0.9320573, 1.0019317, 0.9319286, 1.0024198, -0.0703625, 0.0700030
2: -0.0166806, 0.0206440, -0.0167522, 0.0231811, -0.0398617, 0.0373962
3: -0.0194114, 0.0079526, -0.0209170, 0.0080140, -0.0274254, 0.0288696
4: -0.0158272, 0.0224773, -0.0178405, 0.0226684, -0.0384956, 0.0403178
5: -0.0054440, 0.0390039, -0.0057081, 0.0413246, -0.0467686, 0.0447120
6: -0.0098754, 0.0205992, -0.0100365, 0.0208619, -0.0307373, 0.0306357
7: -0.0110539, 0.0540043, -0.0111413, 0.0572950, -0.0683489, 0.0651456
8: -0.0060723, 0.0325655, -0.0063281, 0.0325844, -0.0386567, 0.0388936
9: -0.0195398, 0.0229276, -0.0226925, 0.0230478, -0.0425876, 0.0456201

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0723066, upper bound: 0.0702436
time: 2.34 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0731746, upper bound: 0.0731873
time: 1.88 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0352645, 0.0092633, -0.0336734, 0.0091617, -0.0444262, 0.0429367
1: 0.9319286, 1.0024198, 0.9320573, 1.0019317, -0.0700030, 0.0703625
2: -0.0167522, 0.0231811, -0.0166806, 0.0206440, -0.0373962, 0.0398617
3: -0.0209170, 0.0080140, -0.0194114, 0.0079526, -0.0288696, 0.0274254
4: -0.0178405, 0.0226684, -0.0158272, 0.0224773, -0.0403178, 0.0384956
5: -0.0057081, 0.0413246, -0.0054440, 0.0390039, -0.0447120, 0.0467686
6: -0.0100365, 0.0208619, -0.0098754, 0.0205992, -0.0306357, 0.0307373
7: -0.0111413, 0.0572950, -0.0110539, 0.0540043, -0.0651456, 0.0683489
8: -0.0063281, 0.0325844, -0.0060723, 0.0325655, -0.0388936, 0.0386567
9: -0.0226925, 0.0230478, -0.0195398, 0.0229276, -0.0456201, 0.0425876

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0723065, upper bound: 0.0702436
time: 1.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0731596, upper bound: 0.0731873
time: 1.60 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0352645, 0.0092633, -0.0352645, 0.0092633, -0.0445278, 0.0445278
1: 0.9319286, 1.0024198, 0.9319286, 1.0024198, -0.0704912, 0.0704912
2: -0.0167522, 0.0231811, -0.0167522, 0.0231811, -0.0399333, 0.0399333
3: -0.0209170, 0.0080140, -0.0209170, 0.0080140, -0.0289310, 0.0289310
4: -0.0178405, 0.0226684, -0.0178405, 0.0226684, -0.0405089, 0.0405089
5: -0.0057081, 0.0413246, -0.0057081, 0.0413246, -0.0470327, 0.0470327
6: -0.0100365, 0.0208619, -0.0100365, 0.0208619, -0.0308984, 0.0308984
7: -0.0111413, 0.0572950, -0.0111413, 0.0572950, -0.0684363, 0.0684363
8: -0.0063281, 0.0325844, -0.0063281, 0.0325844, -0.0389125, 0.0389125
9: -0.0226925, 0.0230478, -0.0226925, 0.0230478, -0.0457403, 0.0457403

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0723065, upper bound: 0.0702436
time: 1.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0731596, upper bound: 0.0731873
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0269109, 0.0089970, -0.0427296, 0.0095296, -0.0364404, 0.0517266
1: 0.9717264, 1.0011108, 0.9316254, 1.0069294, -0.0352030, 0.0412024
2: -0.0165492, 0.0147392, -0.0169220, 0.0343331, -0.0508823, 0.0316612
3: -0.0158362, 0.0078397, -0.0276454, 0.0081593, -0.0239955, 0.0354851
4: -0.0111144, 0.0221261, -0.0268419, 0.0231199, -0.0342343, 0.0489679
5: -0.0052198, 0.0318536, -0.0063974, 0.0516199, -0.0568398, 0.0382510
6: -0.0095951, 0.0103872, -0.0104357, 0.0214505, -0.0310457, 0.0208229
7: -0.0109142, 0.0434356, -0.0113728, 0.0721229, -0.0830371, 0.0548085
8: -0.0056206, 0.0165341, -0.0069547, 0.0326291, -0.0253751, 0.0234889
9: -0.0148018, 0.0033068, -0.0369036, 0.0233311, -0.0255787, 0.0402104

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0715777, upper bound: 0.0695446
time: 1.94 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0715777, upper bound: 0.0722830
time: 1.81 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0303371, 0.0090587, -0.0427655, 0.0095335, -0.0398706, 0.0518242
1: 0.9713005, 1.0016538, 0.9316190, 1.0070096, -0.0357091, 0.0413842
2: -0.0166200, 0.0170489, -0.0169255, 0.0343917, -0.0510117, 0.0339744
3: -0.0172441, 0.0079008, -0.0276797, 0.0081623, -0.0254064, 0.0355805
4: -0.0129984, 0.0223165, -0.0268931, 0.0231293, -0.0361277, 0.0492096
5: -0.0054733, 0.0348890, -0.0064173, 0.0516682, -0.0571415, 0.0413063
6: -0.0097274, 0.0112310, -0.0104429, 0.0214734, -0.0312008, 0.0216738
7: -0.0109637, 0.0483148, -0.0113761, 0.0721917, -0.0831554, 0.0596909
8: -0.0058423, 0.0167663, -0.0069664, 0.0326300, -0.0256089, 0.0237327
9: -0.0157837, 0.0034317, -0.0369665, 0.0233370, -0.0266123, 0.0403982

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0715777, upper bound: 0.0695451
time: 2.48 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0715777, upper bound: 0.0722830
time: 1.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0336734, 0.0091617, -0.0435692, 0.0095765, -0.0432499, 0.0527310
1: 0.9320573, 1.0019317, 0.9315492, 1.0087072, -0.0766498, 0.0703825
2: -0.0166806, 0.0206440, -0.0169640, 0.0356558, -0.0523364, 0.0376080
3: -0.0194114, 0.0079526, -0.0284151, 0.0081955, -0.0276069, 0.0363676
4: -0.0158272, 0.0224773, -0.0278803, 0.0232324, -0.0390597, 0.0503576
5: -0.0054440, 0.0390039, -0.0064783, 0.0528317, -0.0582758, 0.0454822
6: -0.0098754, 0.0205992, -0.0105213, 0.0215180, -0.0313934, 0.0311205
7: -0.0110539, 0.0540043, -0.0114121, 0.0738151, -0.0848690, 0.0654163
8: -0.0060723, 0.0325655, -0.0070946, 0.0326403, -0.0387126, 0.0396601
9: -0.0195398, 0.0229276, -0.0385596, 0.0234022, -0.0429420, 0.0614872

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717263, upper bound: 0.0700264
time: 1.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0717263, upper bound: 0.0729750
time: 1.70 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0352645, 0.0092633, -0.0436051, 0.0095805, -0.0448449, 0.0528684
1: 0.9319286, 1.0024198, 0.9315428, 1.0087911, -0.0768625, 0.0708770
2: -0.0167522, 0.0231811, -0.0169675, 0.0357144, -0.0524667, 0.0401486
3: -0.0209170, 0.0080140, -0.0284492, 0.0081985, -0.0291155, 0.0364632
4: -0.0178405, 0.0226684, -0.0279314, 0.0232419, -0.0410823, 0.0505998
5: -0.0057081, 0.0413246, -0.0064979, 0.0528800, -0.0585881, 0.0478225
6: -0.0100365, 0.0208619, -0.0105285, 0.0215407, -0.0315772, 0.0313904
7: -0.0111413, 0.0572950, -0.0114154, 0.0738837, -0.0850250, 0.0687103
8: -0.0063281, 0.0325844, -0.0071063, 0.0326412, -0.0389693, 0.0396907
9: -0.0226925, 0.0230478, -0.0386224, 0.0234081, -0.0461006, 0.0616702

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0717249, upper bound: 0.0700264
time: 2.04 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0717249, upper bound: 0.0729750
time: 2.46 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0427296, 0.0095296, -0.0269109, 0.0089970, -0.0517266, 0.0364404
1: 0.9316254, 1.0069294, 0.9717264, 1.0011108, -0.0412024, 0.0352030
2: -0.0169220, 0.0343331, -0.0165492, 0.0147392, -0.0316612, 0.0508823
3: -0.0276454, 0.0081593, -0.0158362, 0.0078397, -0.0354851, 0.0239955
4: -0.0268419, 0.0231199, -0.0111144, 0.0221261, -0.0489679, 0.0342343
5: -0.0063974, 0.0516199, -0.0052198, 0.0318536, -0.0382510, 0.0568398
6: -0.0104357, 0.0214505, -0.0095951, 0.0103872, -0.0208229, 0.0310457
7: -0.0113728, 0.0721229, -0.0109142, 0.0434356, -0.0548085, 0.0830371
8: -0.0069547, 0.0326291, -0.0056206, 0.0165341, -0.0234889, 0.0253751
9: -0.0369036, 0.0233311, -0.0148018, 0.0033068, -0.0402104, 0.0255788

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0720481, upper bound: 0.0738833
time: 1.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0722010, upper bound: 0.0741906
time: 1.36 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0427655, 0.0095335, -0.0303371, 0.0090587, -0.0518242, 0.0398706
1: 0.9316190, 1.0070096, 0.9713005, 1.0016538, -0.0413842, 0.0357091
2: -0.0169255, 0.0343917, -0.0166200, 0.0170489, -0.0339744, 0.0510117
3: -0.0276797, 0.0081623, -0.0172441, 0.0079008, -0.0355805, 0.0254064
4: -0.0268931, 0.0231293, -0.0129984, 0.0223165, -0.0492096, 0.0361277
5: -0.0064173, 0.0516682, -0.0054733, 0.0348890, -0.0413063, 0.0571415
6: -0.0104429, 0.0214734, -0.0097274, 0.0112310, -0.0216738, 0.0312008
7: -0.0113761, 0.0721917, -0.0109637, 0.0483148, -0.0596909, 0.0831554
8: -0.0069664, 0.0326300, -0.0058423, 0.0167663, -0.0237327, 0.0256090
9: -0.0369665, 0.0233370, -0.0157837, 0.0034317, -0.0403982, 0.0266123

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 54

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0720481, upper bound: 0.0738833
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0722010, upper bound: 0.0741906
time: 1.98 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0435692, 0.0095765, -0.0336734, 0.0091617, -0.0527310, 0.0432499
1: 0.9315492, 1.0087072, 0.9320573, 1.0019317, -0.0703825, 0.0766498
2: -0.0169640, 0.0356558, -0.0166806, 0.0206440, -0.0376080, 0.0523364
3: -0.0284151, 0.0081955, -0.0194114, 0.0079526, -0.0363676, 0.0276069
4: -0.0278803, 0.0232324, -0.0158272, 0.0224773, -0.0503576, 0.0390597
5: -0.0064783, 0.0528317, -0.0054440, 0.0390039, -0.0454822, 0.0582758
6: -0.0105213, 0.0215180, -0.0098754, 0.0205992, -0.0311205, 0.0313934
7: -0.0114121, 0.0738151, -0.0110539, 0.0540043, -0.0654163, 0.0848690
8: -0.0070946, 0.0326403, -0.0060723, 0.0325655, -0.0396601, 0.0387126
9: -0.0385596, 0.0234022, -0.0195398, 0.0229276, -0.0614872, 0.0429420

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727443, upper bound: 0.0741347
time: 1.51 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728868, upper bound: 0.0744087
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0436051, 0.0095805, -0.0352645, 0.0092633, -0.0528684, 0.0448449
1: 0.9315428, 1.0087911, 0.9319286, 1.0024198, -0.0708770, 0.0768625
2: -0.0169675, 0.0357144, -0.0167522, 0.0231811, -0.0401486, 0.0524667
3: -0.0284492, 0.0081985, -0.0209170, 0.0080140, -0.0364632, 0.0291155
4: -0.0279314, 0.0232419, -0.0178405, 0.0226684, -0.0505998, 0.0410823
5: -0.0064979, 0.0528800, -0.0057081, 0.0413246, -0.0478225, 0.0585881
6: -0.0105285, 0.0215407, -0.0100365, 0.0208619, -0.0313904, 0.0315772
7: -0.0114154, 0.0738837, -0.0111413, 0.0572950, -0.0687103, 0.0850250
8: -0.0071063, 0.0326412, -0.0063281, 0.0325844, -0.0396907, 0.0389693
9: -0.0386224, 0.0234081, -0.0226925, 0.0230478, -0.0616702, 0.0461006

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0727443, upper bound: 0.0741228
time: 1.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0728868, upper bound: 0.0743980
time: 1.47 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0435790, 0.0095765, -0.0381604, 0.0093847, -0.0529638, 0.0477368
1: 0.9315496, 1.0087266, 0.9317837, 1.0024081, -0.0708585, 0.0769429
2: -0.0169637, 0.0356732, -0.0168332, 0.0275149, -0.0444786, 0.0525064
3: -0.0284241, 0.0081952, -0.0234963, 0.0080834, -0.0365075, 0.0316916
4: -0.0278910, 0.0232317, -0.0211653, 0.0228839, -0.0507749, 0.0443970
5: -0.0064749, 0.0528509, -0.0055177, 0.0455080, -0.0519829, 0.0583686
6: -0.0105210, 0.0215157, -0.0102230, 0.0205831, -0.0311041, 0.0317386
7: -0.0114120, 0.0738372, -0.0112465, 0.0631668, -0.0745788, 0.0850836
8: -0.0070939, 0.0326402, -0.0066225, 0.0326058, -0.0396997, 0.0392627
9: -0.0385843, 0.0234018, -0.0284502, 0.0231832, -0.0617675, 0.0518520

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0723495, upper bound: 0.0742485
time: 1.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0724601, upper bound: 0.0745197
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0435745, 0.0095752, -0.0390663, 0.0095847, -0.0531592, 0.0486415
1: 0.9315521, 1.0087132, 0.9314071, 1.0029097, -0.0713576, 0.0773062
2: -0.0169624, 0.0356655, -0.0170398, 0.0291378, -0.0461003, 0.0527053
3: -0.0284197, 0.0081941, -0.0243843, 0.0082617, -0.0366814, 0.0325784
4: -0.0278844, 0.0232282, -0.0224576, 0.0234393, -0.0513237, 0.0456858
5: -0.0064731, 0.0528444, -0.0058557, 0.0468986, -0.0533717, 0.0587001
6: -0.0105184, 0.0215137, -0.0106230, 0.0209217, -0.0314402, 0.0321367
7: -0.0114110, 0.0738284, -0.0114098, 0.0650045, -0.0764155, 0.0852382
8: -0.0070897, 0.0326398, -0.0072858, 0.0326612, -0.0397509, 0.0399257
9: -0.0385761, 0.0233995, -0.0302176, 0.0235349, -0.0621111, 0.0536172

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0722785, upper bound: 0.0742424
time: 2.20 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0723828, upper bound: 0.0745112
time: 1.58 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0436051, 0.0095805, -0.0436051, 0.0095805, -0.0531855, 0.0531855
1: 0.9315428, 1.0087911, 0.9315428, 1.0087911, -0.0772483, 0.0772483
2: -0.0169675, 0.0357144, -0.0169675, 0.0357144, -0.0526819, 0.0526819
3: -0.0284492, 0.0081985, -0.0284492, 0.0081985, -0.0366477, 0.0366477
4: -0.0279314, 0.0232419, -0.0279314, 0.0232419, -0.0511733, 0.0511733
5: -0.0064979, 0.0528800, -0.0064979, 0.0528800, -0.0593779, 0.0593779
6: -0.0105285, 0.0215407, -0.0105285, 0.0215407, -0.0320693, 0.0320693
7: -0.0114154, 0.0738837, -0.0114154, 0.0738837, -0.0852990, 0.0852990
8: -0.0071063, 0.0326412, -0.0071063, 0.0326412, -0.0397475, 0.0397475
9: -0.0386224, 0.0234081, -0.0386224, 0.0234081, -0.0620305, 0.0620305

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 121
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 244
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 50

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0696447, upper bound: 0.0748015
time: 1.55 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0697603, upper bound: 0.0751031
time: 1.37 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 4.59 seconds
NS_A2_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0682410, upper bound: 0.0727889
NS_A2_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0684574, upper bound: 0.0733530
NS_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0689395, upper bound: 0.0732479
NS_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0691346, upper bound: 0.0736857
NS_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0690316, upper bound: 0.0721906
NS_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0692299, upper bound: 0.0738251
NS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0689119, upper bound: 0.0733455
NS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0690998, upper bound: 0.0738000
NS_A2_B2_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0717606, upper bound: 0.0701243
NS_A2_B2_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0725097, upper bound: 0.0730516
NS_A2_B2_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0717606, upper bound: 0.0701243
NS_A2_B2_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0725097, upper bound: 0.0730516
NS_A2_B2_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0717606, upper bound: 0.0701243
NS_A2_B2_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0725029, upper bound: 0.0730516
NS_A2_B2_A1_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0717606, upper bound: 0.0701243
NS_A2_B2_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0725029, upper bound: 0.0730516
NS_A2_B2_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0723066, upper bound: 0.0702436
NS_A2_B2_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0731746, upper bound: 0.0731873
NS_A2_B2_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0723066, upper bound: 0.0702436
NS_A2_B2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0731746, upper bound: 0.0731873
NS_A2_B2_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0723065, upper bound: 0.0702436
NS_A2_B2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0731596, upper bound: 0.0731873
NS_A2_B2_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0723065, upper bound: 0.0702436
NS_A2_B2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0731596, upper bound: 0.0731873
NS_A2_B2_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0715777, upper bound: 0.0695446
NS_A2_B2_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0715777, upper bound: 0.0722830
NS_A2_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0715777, upper bound: 0.0695451
NS_A2_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0715777, upper bound: 0.0722830
NS_A2_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0717263, upper bound: 0.0700264
NS_A2_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0717263, upper bound: 0.0729750
NS_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0717249, upper bound: 0.0700264
NS_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0717249, upper bound: 0.0729750
NS_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0720481, upper bound: 0.0738833
NS_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0722010, upper bound: 0.0741906
NS_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0720481, upper bound: 0.0738833
NS_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0722010, upper bound: 0.0741906
NS_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0727443, upper bound: 0.0741347
NS_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0728868, upper bound: 0.0744087
NS_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0727443, upper bound: 0.0741228
NS_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0728868, upper bound: 0.0743980
NS_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0723495, upper bound: 0.0742485
NS_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0724601, upper bound: 0.0745197
NS_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0722785, upper bound: 0.0742424
NS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0723828, upper bound: 0.0745112
NS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0696447, upper bound: 0.0748015
NS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 4.59
Output dim: 1, lower bound: -0.0697603, upper bound: 0.0751031

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0424620, 0.0094137, -0.0241894, 0.0089902, -0.0514522, 0.0336031
1: 0.9318555, 1.0064081, 0.9719079, 1.0003613, -0.0401415, 0.0345002
2: -0.0167959, 0.0338953, -0.0165414, 0.0131121, -0.0299080, 0.0504368
3: -0.0273915, 0.0080504, -0.0147422, 0.0078329, -0.0352244, 0.0227926
4: -0.0264672, 0.0227806, -0.0095935, 0.0221050, -0.0485722, 0.0323741
5: -0.0063135, 0.0512757, -0.0047720, 0.0297267, -0.0360402, 0.0560477
6: -0.0101958, 0.0213816, -0.0095805, 0.0094690, -0.0196648, 0.0309621
7: -0.0112790, 0.0716086, -0.0109087, 0.0397796, -0.0510586, 0.0825173
8: -0.0065548, 0.0325952, -0.0055961, 0.0163843, -0.0229392, 0.0252361
9: -0.0364450, 0.0231160, -0.0141641, 0.0032930, -0.0397380, 0.0245012

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0708669, upper bound: 0.0733376
time: 1.52 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0708669, upper bound: 0.0733396
time: 1.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0446047, 0.0093730, -0.0326948, 0.0091049, -0.0537096, 0.0420678
1: 0.9319888, 1.0110630, 0.9321296, 1.0013233, -0.0693346, 0.0789334
2: -0.0167237, 0.0370424, -0.0166403, 0.0193993, -0.0361230, 0.0536827
3: -0.0293139, 0.0079878, -0.0185889, 0.0079181, -0.0372319, 0.0265768
4: -0.0290554, 0.0225853, -0.0146457, 0.0223699, -0.0514253, 0.0372310
5: -0.0065436, 0.0541230, -0.0050140, 0.0379813, -0.0445249, 0.0591370
6: -0.0100762, 0.0216058, -0.0097851, 0.0201575, -0.0302337, 0.0313909
7: -0.0112497, 0.0758303, -0.0110050, 0.0523237, -0.0635734, 0.0868353
8: -0.0063466, 0.0325755, -0.0059288, 0.0325548, -0.0389014, 0.0385043
9: -0.0404770, 0.0229916, -0.0181069, 0.0228601, -0.0633371, 0.0410985

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0714298, upper bound: 0.0732390
time: 1.50 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0714278, upper bound: 0.0732391
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0433011, 0.0094603, -0.0328123, 0.0091396, -0.0524408, 0.0422725
1: 0.9317802, 1.0081288, 0.9320630, 1.0013683, -0.0695881, 0.0760657
2: -0.0168375, 0.0352177, -0.0166768, 0.0195860, -0.0364235, 0.0518945
3: -0.0281609, 0.0080862, -0.0186993, 0.0079495, -0.0361104, 0.0267855
4: -0.0275052, 0.0228921, -0.0148039, 0.0224680, -0.0499731, 0.0376960
5: -0.0063929, 0.0524870, -0.0050383, 0.0381370, -0.0445299, 0.0575253
6: -0.0102807, 0.0214480, -0.0098554, 0.0201788, -0.0304595, 0.0313033
7: -0.0113179, 0.0733001, -0.0110334, 0.0525543, -0.0638722, 0.0843335
8: -0.0066934, 0.0326063, -0.0060454, 0.0325646, -0.0392580, 0.0386517
9: -0.0380998, 0.0231865, -0.0183132, 0.0229222, -0.0610220, 0.0414997

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0716385, upper bound: 0.0736765
time: 1.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0716362, upper bound: 0.0736765
time: 2.01 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0432968, 0.0094590, -0.0333982, 0.0092794, -0.0525762, 0.0428572
1: 0.9317825, 1.0081160, 0.9318015, 1.0017757, -0.0699933, 0.0763145
2: -0.0168361, 0.0352104, -0.0168203, 0.0205473, -0.0373835, 0.0520307
3: -0.0281568, 0.0080851, -0.0192583, 0.0080734, -0.0362301, 0.0273434
4: -0.0274990, 0.0228886, -0.0156310, 0.0228537, -0.0503526, 0.0385196
5: -0.0063911, 0.0524809, -0.0053172, 0.0389260, -0.0453171, 0.0577981
6: -0.0102781, 0.0214460, -0.0101339, 0.0204658, -0.0307440, 0.0315799
7: -0.0113169, 0.0732918, -0.0111477, 0.0536812, -0.0649981, 0.0844395
8: -0.0066892, 0.0326059, -0.0065069, 0.0326031, -0.0392923, 0.0391128
9: -0.0380921, 0.0231842, -0.0193263, 0.0231665, -0.0612586, 0.0425106

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0718254, upper bound: 0.0738181
time: 1.81 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0718246, upper bound: 0.0738194
time: 1.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0442861, 0.0093412, -0.0341047, 0.0092969, -0.0535830, 0.0434458
1: 0.9320469, 1.0103315, 0.9318030, 1.0022182, -0.0701713, 0.0785285
2: -0.0166918, 0.0365224, -0.0168202, 0.0215951, -0.0382869, 0.0533426
3: -0.0290164, 0.0079603, -0.0199004, 0.0080730, -0.0370894, 0.0278607
4: -0.0286496, 0.0224995, -0.0165273, 0.0228524, -0.0515020, 0.0390268
5: -0.0065057, 0.0536476, -0.0056178, 0.0398366, -0.0463423, 0.0592654
6: -0.0100137, 0.0215739, -0.0101458, 0.0207946, -0.0308083, 0.0317197
7: -0.0112235, 0.0751812, -0.0111645, 0.0550301, -0.0662536, 0.0863457
8: -0.0062432, 0.0325670, -0.0065205, 0.0326029, -0.0388461, 0.0390875
9: -0.0398414, 0.0229373, -0.0205584, 0.0231652, -0.0630066, 0.0434957

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0715909, upper bound: 0.0733345
time: 1.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0715901, upper bound: 0.0733345
time: 1.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0429667, 0.0094256, -0.0342160, 0.0093283, -0.0522950, 0.0436416
1: 0.9318431, 1.0074153, 0.9317431, 1.0022607, -0.0704176, 0.0756722
2: -0.0168029, 0.0346704, -0.0168530, 0.0217711, -0.0385740, 0.0515233
3: -0.0278495, 0.0080564, -0.0200045, 0.0081013, -0.0359508, 0.0280609
4: -0.0270803, 0.0227992, -0.0166769, 0.0229406, -0.0500209, 0.0394761
5: -0.0063536, 0.0519875, -0.0056413, 0.0399819, -0.0463355, 0.0576287
6: -0.0102129, 0.0214149, -0.0102090, 0.0208153, -0.0310282, 0.0316239
7: -0.0112894, 0.0726186, -0.0111900, 0.0552474, -0.0665368, 0.0838086
8: -0.0065814, 0.0325970, -0.0066255, 0.0326117, -0.0391932, 0.0392225
9: -0.0374307, 0.0231277, -0.0207529, 0.0232211, -0.0606518, 0.0438806

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0717978, upper bound: 0.0737907
time: 1.60 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0717924, upper bound: 0.0737909
time: 1.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0306500, 0.0090634, -0.0269109, 0.0089970, -0.0396470, 0.0359742
1: 0.9712740, 1.0013918, 0.9717264, 1.0011108, -0.0298368, 0.0296654
2: -0.0166253, 0.0172083, -0.0165492, 0.0147392, -0.0313644, 0.0337575
3: -0.0173188, 0.0079054, -0.0158362, 0.0078397, -0.0251585, 0.0237415
4: -0.0130660, 0.0223308, -0.0111144, 0.0221261, -0.0351921, 0.0334452
5: -0.0052338, 0.0351842, -0.0052198, 0.0318536, -0.0370874, 0.0404041
6: -0.0097374, 0.0111863, -0.0095951, 0.0103872, -0.0201246, 0.0207815
7: -0.0109674, 0.0487447, -0.0109142, 0.0434356, -0.0544031, 0.0596589
8: -0.0058589, 0.0167831, -0.0056206, 0.0165341, -0.0223931, 0.0224037
9: -0.0158738, 0.0034411, -0.0148018, 0.0033068, -0.0191806, 0.0182429

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0697697, upper bound: 0.0723933
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0697697, upper bound: 0.0730519
time: 2.26 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0306500, 0.0090634, -0.0303371, 0.0090587, -0.0397087, 0.0394004
1: 0.9712740, 1.0013918, 0.9713005, 1.0016538, -0.0303798, 0.0300913
2: -0.0166253, 0.0172083, -0.0166200, 0.0170489, -0.0336742, 0.0338283
3: -0.0173188, 0.0079054, -0.0172441, 0.0079008, -0.0252196, 0.0251495
4: -0.0130660, 0.0223308, -0.0129984, 0.0223165, -0.0353825, 0.0353292
5: -0.0052338, 0.0351842, -0.0054733, 0.0348890, -0.0401228, 0.0406575
6: -0.0097374, 0.0111863, -0.0097274, 0.0112310, -0.0209683, 0.0209138
7: -0.0109674, 0.0487447, -0.0109637, 0.0483148, -0.0592822, 0.0597084
8: -0.0058589, 0.0167831, -0.0058423, 0.0167663, -0.0226252, 0.0226253
9: -0.0158738, 0.0034411, -0.0157837, 0.0034317, -0.0193055, 0.0192248

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 104

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0697698, upper bound: 0.0723958
time: 1.56 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0697698, upper bound: 0.0730516
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0330830, 0.0091618, -0.0269109, 0.0089970, -0.0420800, 0.0360727
1: 0.9329535, 1.0019526, 0.9717264, 1.0011108, -0.0387728, 0.0302263
2: -0.0166970, 0.0199214, -0.0165492, 0.0147392, -0.0314361, 0.0364707
3: -0.0189381, 0.0079669, -0.0158362, 0.0078397, -0.0267778, 0.0238030
4: -0.0152234, 0.0225221, -0.0111144, 0.0221261, -0.0373494, 0.0336365
5: -0.0055111, 0.0383119, -0.0052198, 0.0318536, -0.0373648, 0.0435317
6: -0.0098963, 0.0206932, -0.0095951, 0.0103872, -0.0202835, 0.0299006
7: -0.0110519, 0.0529142, -0.0109142, 0.0434356, -0.0544875, 0.0638284
8: -0.0061123, 0.0325700, -0.0056206, 0.0165341, -0.0226464, 0.0237813
9: -0.0185360, 0.0229564, -0.0148018, 0.0033068, -0.0218428, 0.0248847

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0697576, upper bound: 0.0723933
time: 1.75 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0697576, upper bound: 0.0730519
time: 1.51 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0330830, 0.0091618, -0.0303371, 0.0090587, -0.0421417, 0.0394989
1: 0.9323453, 1.0019526, 0.9713005, 1.0016538, -0.0389552, 0.0306522
2: -0.0166970, 0.0199214, -0.0166200, 0.0170489, -0.0337458, 0.0365414
3: -0.0189381, 0.0079669, -0.0172441, 0.0079008, -0.0268389, 0.0252110
4: -0.0152234, 0.0225221, -0.0129984, 0.0223165, -0.0375398, 0.0355205
5: -0.0055111, 0.0383119, -0.0054733, 0.0348890, -0.0404001, 0.0437851
6: -0.0098963, 0.0206932, -0.0097274, 0.0112310, -0.0211273, 0.0300511
7: -0.0110519, 0.0529142, -0.0109637, 0.0483148, -0.0593666, 0.0638779
8: -0.0061123, 0.0325700, -0.0058423, 0.0167663, -0.0228786, 0.0240095
9: -0.0185360, 0.0229564, -0.0157837, 0.0034317, -0.0219678, 0.0259229

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0697576, upper bound: 0.0723933
time: 2.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0697576, upper bound: 0.0730516
time: 1.62 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0306500, 0.0090634, -0.0336734, 0.0091617, -0.0398117, 0.0427367
1: 0.9712740, 1.0013918, 0.9323050, 1.0019317, -0.0306576, 0.0389691
2: -0.0166253, 0.0172083, -0.0166806, 0.0206440, -0.0372693, 0.0338889
3: -0.0173188, 0.0079054, -0.0194114, 0.0079526, -0.0252714, 0.0273168
4: -0.0130660, 0.0223308, -0.0158272, 0.0224773, -0.0355433, 0.0381580
5: -0.0052338, 0.0351842, -0.0054440, 0.0390039, -0.0442377, 0.0406283
6: -0.0097374, 0.0111863, -0.0098754, 0.0205992, -0.0303365, 0.0210617
7: -0.0109674, 0.0487447, -0.0110539, 0.0540043, -0.0649717, 0.0597986
8: -0.0058589, 0.0167831, -0.0060723, 0.0325655, -0.0240986, 0.0228554
9: -0.0158738, 0.0034411, -0.0195398, 0.0229276, -0.0261227, 0.0229809

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0702539, upper bound: 0.0725398
time: 2.07 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0702539, upper bound: 0.0731961
time: 2.01 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0306500, 0.0090634, -0.0352645, 0.0092633, -0.0399133, 0.0443278
1: 0.9712740, 1.0013918, 0.9319286, 1.0024198, -0.0311458, 0.0393939
2: -0.0166253, 0.0172083, -0.0167522, 0.0231811, -0.0398064, 0.0339605
3: -0.0173188, 0.0079054, -0.0209170, 0.0080140, -0.0253328, 0.0288224
4: -0.0130660, 0.0223308, -0.0178405, 0.0226684, -0.0357344, 0.0401713
5: -0.0052338, 0.0351842, -0.0057081, 0.0413246, -0.0465584, 0.0408923
6: -0.0097374, 0.0111863, -0.0100365, 0.0208619, -0.0305993, 0.0212228
7: -0.0109674, 0.0487447, -0.0111413, 0.0572950, -0.0682624, 0.0598860
8: -0.0058589, 0.0167831, -0.0063281, 0.0325844, -0.0243790, 0.0231112
9: -0.0158738, 0.0034411, -0.0226925, 0.0230478, -0.0262618, 0.0261336

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0702539, upper bound: 0.0725398
time: 1.84 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0702539, upper bound: 0.0731873
time: 1.42 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0330830, 0.0091618, -0.0336734, 0.0091617, -0.0422447, 0.0428352
1: 0.9320264, 1.0019526, 0.9320573, 1.0019317, -0.0699052, 0.0698953
2: -0.0166970, 0.0199214, -0.0166806, 0.0206440, -0.0373409, 0.0366020
3: -0.0189381, 0.0079669, -0.0194114, 0.0079526, -0.0268907, 0.0273783
4: -0.0152234, 0.0225221, -0.0158272, 0.0224773, -0.0377006, 0.0383493
5: -0.0055111, 0.0383119, -0.0054440, 0.0390039, -0.0445151, 0.0437559
6: -0.0098963, 0.0206932, -0.0098754, 0.0205992, -0.0304955, 0.0305686
7: -0.0110519, 0.0529142, -0.0110539, 0.0540043, -0.0650562, 0.0639681
8: -0.0061123, 0.0325700, -0.0060723, 0.0325655, -0.0386778, 0.0386423
9: -0.0185360, 0.0229564, -0.0195398, 0.0229276, -0.0414636, 0.0424962

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0702415, upper bound: 0.0725398
time: 2.24 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0702415, upper bound: 0.0731961
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0330830, 0.0091618, -0.0352645, 0.0092633, -0.0423463, 0.0444263
1: 0.9320264, 1.0019526, 0.9319286, 1.0024198, -0.0703934, 0.0700240
2: -0.0166970, 0.0199214, -0.0167522, 0.0231811, -0.0398781, 0.0366737
3: -0.0189381, 0.0079669, -0.0209170, 0.0080140, -0.0269521, 0.0288839
4: -0.0152234, 0.0225221, -0.0178405, 0.0226684, -0.0378917, 0.0403626
5: -0.0055111, 0.0383119, -0.0057081, 0.0413246, -0.0468357, 0.0440200
6: -0.0098963, 0.0206932, -0.0100365, 0.0208619, -0.0307582, 0.0307297
7: -0.0110519, 0.0529142, -0.0111413, 0.0572950, -0.0683469, 0.0640555
8: -0.0061123, 0.0325700, -0.0063281, 0.0325844, -0.0386967, 0.0388981
9: -0.0185360, 0.0229564, -0.0226925, 0.0230478, -0.0415838, 0.0456489

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0702415, upper bound: 0.0725398
time: 1.31 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0702415, upper bound: 0.0731873
time: 2.11 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0306500, 0.0090634, -0.0435692, 0.0095765, -0.0402265, 0.0526326
1: 0.9712740, 1.0013918, 0.9315492, 1.0087072, -0.0374331, 0.0415557
2: -0.0166253, 0.0172083, -0.0169640, 0.0356558, -0.0522811, 0.0341723
3: -0.0173188, 0.0079054, -0.0284151, 0.0081955, -0.0255143, 0.0363204
4: -0.0130660, 0.0223308, -0.0278803, 0.0232324, -0.0362984, 0.0502111
5: -0.0052338, 0.0351842, -0.0064783, 0.0528317, -0.0580655, 0.0416625
6: -0.0097374, 0.0111863, -0.0105213, 0.0215180, -0.0312554, 0.0217077
7: -0.0109674, 0.0487447, -0.0114121, 0.0738151, -0.0847825, 0.0601567
8: -0.0058589, 0.0167831, -0.0070946, 0.0326403, -0.0257569, 0.0238777
9: -0.0158738, 0.0034411, -0.0385596, 0.0234022, -0.0268562, 0.0420007

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0716033, upper bound: 0.0727737
time: 2.07 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0716633, upper bound: 0.0729158
time: 1.49 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0330830, 0.0091618, -0.0436051, 0.0095805, -0.0426635, 0.0527669
1: 0.9320264, 1.0019526, 0.9315428, 1.0087911, -0.0767646, 0.0704098
2: -0.0166970, 0.0199214, -0.0169675, 0.0357144, -0.0524114, 0.0368889
3: -0.0189381, 0.0079669, -0.0284492, 0.0081985, -0.0271366, 0.0364161
4: -0.0152234, 0.0225221, -0.0279314, 0.0232419, -0.0384652, 0.0504535
5: -0.0055111, 0.0383119, -0.0064979, 0.0528800, -0.0583912, 0.0448098
6: -0.0098963, 0.0206932, -0.0105285, 0.0215407, -0.0314370, 0.0312217
7: -0.0110519, 0.0529142, -0.0114154, 0.0738837, -0.0849356, 0.0643295
8: -0.0061123, 0.0325700, -0.0071063, 0.0326412, -0.0387535, 0.0396763
9: -0.0185360, 0.0229564, -0.0386224, 0.0234081, -0.0419442, 0.0615788

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 50

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 196

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0716021, upper bound: 0.0727738
time: 1.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0716619, upper bound: 0.0699244
time: 1.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0437627, 0.0093281, -0.0266405, 0.0089642, -0.0527269, 0.0359686
1: 0.9320623, 1.0091690, 0.9718144, 1.0010608, -0.0406662, 0.0373546
2: -0.0166832, 0.0357119, -0.0165116, 0.0145412, -0.0312244, 0.0522235
3: -0.0285414, 0.0079529, -0.0157182, 0.0078072, -0.0363486, 0.0236711
4: -0.0280123, 0.0224766, -0.0109428, 0.0220247, -0.0500371, 0.0334194
5: -0.0064680, 0.0529053, -0.0051964, 0.0316260, -0.0380941, 0.0581018
6: -0.0099938, 0.0215416, -0.0095247, 0.0103061, -0.0203000, 0.0310663
7: -0.0112123, 0.0741327, -0.0108879, 0.0430549, -0.0542672, 0.0850206
8: -0.0062119, 0.0325647, -0.0055027, 0.0165128, -0.0227247, 0.0253542
9: -0.0388154, 0.0229229, -0.0147185, 0.0032403, -0.0420557, 0.0250881

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0693745, upper bound: 0.0731340
time: 1.41 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0693745, upper bound: 0.0738833
time: 2.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0424518, 0.0094138, -0.0269109, 0.0089970, -0.0514489, 0.0363247
1: 0.9318550, 1.0063912, 0.9717264, 1.0011108, -0.0408613, 0.0346648
2: -0.0167962, 0.0338773, -0.0165492, 0.0147392, -0.0315353, 0.0504266
3: -0.0273821, 0.0080507, -0.0158362, 0.0078397, -0.0352217, 0.0238868
4: -0.0264559, 0.0227814, -0.0111144, 0.0221261, -0.0485820, 0.0338959
5: -0.0063165, 0.0512564, -0.0052198, 0.0318536, -0.0381701, 0.0564762
6: -0.0101962, 0.0213837, -0.0095951, 0.0103872, -0.0205834, 0.0309789
7: -0.0112791, 0.0715858, -0.0109142, 0.0434356, -0.0547147, 0.0825000
8: -0.0065557, 0.0325952, -0.0056206, 0.0165341, -0.0230898, 0.0253081
9: -0.0364197, 0.0231166, -0.0148018, 0.0033068, -0.0397265, 0.0252220

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694808, upper bound: 0.0733700
time: 1.53 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694808, upper bound: 0.0741906
time: 1.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0437911, 0.0093312, -0.0300807, 0.0090268, -0.0528179, 0.0394119
1: 0.9320576, 1.0092365, 0.9713852, 1.0016074, -0.0408533, 0.0378512
2: -0.0166858, 0.0357579, -0.0165834, 0.0168597, -0.0335456, 0.0523413
3: -0.0285686, 0.0079552, -0.0171321, 0.0078692, -0.0364378, 0.0250873
4: -0.0280537, 0.0224837, -0.0128341, 0.0222180, -0.0502717, 0.0353178
5: -0.0064872, 0.0529425, -0.0054502, 0.0346725, -0.0411597, 0.0583927
6: -0.0099993, 0.0215638, -0.0096590, 0.0111530, -0.0211522, 0.0312228
7: -0.0112149, 0.0741862, -0.0109381, 0.0479544, -0.0591693, 0.0851244
8: -0.0062208, 0.0325654, -0.0057277, 0.0167462, -0.0229669, 0.0255918
9: -0.0388631, 0.0229273, -0.0157057, 0.0033671, -0.0422302, 0.0261176

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 230

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0693753, upper bound: 0.0731388
time: 1.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0693753, upper bound: 0.0738833
time: 1.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0424881, 0.0094177, -0.0303371, 0.0090587, -0.0515468, 0.0397548
1: 0.9318488, 1.0064712, 0.9713005, 1.0016538, -0.0410379, 0.0351707
2: -0.0167996, 0.0339365, -0.0166200, 0.0170489, -0.0338485, 0.0505565
3: -0.0274166, 0.0080536, -0.0172441, 0.0079008, -0.0353174, 0.0252977
4: -0.0265075, 0.0227907, -0.0129984, 0.0223165, -0.0488240, 0.0357891
5: -0.0063367, 0.0513052, -0.0054733, 0.0348890, -0.0412257, 0.0567785
6: -0.0102033, 0.0214068, -0.0097274, 0.0112310, -0.0214343, 0.0311342
7: -0.0112823, 0.0716553, -0.0109637, 0.0483148, -0.0595971, 0.0826190
8: -0.0065672, 0.0325962, -0.0058423, 0.0167663, -0.0233335, 0.0255407
9: -0.0364833, 0.0231224, -0.0157837, 0.0034317, -0.0399150, 0.0262553

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 104

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694811, upper bound: 0.0733770
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0694811, upper bound: 0.0741906
time: 1.62 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0446026, 0.0093740, -0.0335523, 0.0091260, -0.0537286, 0.0429263
1: 0.9319866, 1.0110602, 0.9321256, 1.0018851, -0.0698985, 0.0789346
2: -0.0167249, 0.0370380, -0.0166431, 0.0204503, -0.0371752, 0.0536811
3: -0.0293121, 0.0079888, -0.0192973, 0.0079203, -0.0372323, 0.0272861
4: -0.0290547, 0.0225885, -0.0156630, 0.0223766, -0.0514312, 0.0382515
5: -0.0065478, 0.0541155, -0.0054180, 0.0388450, -0.0453928, 0.0595336
6: -0.0100784, 0.0216089, -0.0098033, 0.0205764, -0.0306548, 0.0314121
7: -0.0112505, 0.0758240, -0.0110248, 0.0537662, -0.0650167, 0.0868488
8: -0.0063502, 0.0325759, -0.0059525, 0.0325554, -0.0389056, 0.0385283
9: -0.0404681, 0.0229936, -0.0193278, 0.0228638, -0.0633319, 0.0423214

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0698757, upper bound: 0.0733718
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0698757, upper bound: 0.0741347
time: 1.45 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0432911, 0.0094604, -0.0336734, 0.0091617, -0.0524528, 0.0431338
1: 0.9317796, 1.0081095, 0.9320573, 1.0019317, -0.0701521, 0.0760521
2: -0.0168378, 0.0352000, -0.0166806, 0.0206440, -0.0374818, 0.0518805
3: -0.0281517, 0.0080865, -0.0194114, 0.0079526, -0.0361042, 0.0274979
4: -0.0274941, 0.0228929, -0.0158272, 0.0224773, -0.0499714, 0.0387202
5: -0.0063961, 0.0524676, -0.0054440, 0.0390039, -0.0454000, 0.0579116
6: -0.0102811, 0.0214502, -0.0098754, 0.0205992, -0.0308803, 0.0313256
7: -0.0113180, 0.0732775, -0.0110539, 0.0540043, -0.0653223, 0.0843314
8: -0.0066943, 0.0326063, -0.0060723, 0.0325655, -0.0392597, 0.0386786
9: -0.0380747, 0.0231870, -0.0195398, 0.0229276, -0.0610023, 0.0427268

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0699605, upper bound: 0.0735649
time: 1.78 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0699605, upper bound: 0.0744087
time: 1.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0446311, 0.0093771, -0.0351456, 0.0092286, -0.0538597, 0.0445227
1: 0.9319819, 1.0111285, 0.9319955, 1.0023748, -0.0703929, 0.0791331
2: -0.0167275, 0.0370841, -0.0167156, 0.0229917, -0.0397192, 0.0537997
3: -0.0293393, 0.0079911, -0.0208054, 0.0079824, -0.0373217, 0.0287965
4: -0.0290960, 0.0225955, -0.0176798, 0.0225699, -0.0516659, 0.0402753
5: -0.0065667, 0.0541528, -0.0056824, 0.0411683, -0.0477349, 0.0598352
6: -0.0100838, 0.0216309, -0.0099660, 0.0208396, -0.0309234, 0.0315969
7: -0.0112531, 0.0758776, -0.0111131, 0.0570625, -0.0683156, 0.0869906
8: -0.0063590, 0.0325766, -0.0062112, 0.0325746, -0.0389336, 0.0387877
9: -0.0405159, 0.0229980, -0.0224830, 0.0229854, -0.0635013, 0.0454810

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0698757, upper bound: 0.0733718
time: 1.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0698757, upper bound: 0.0741228
time: 1.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0433273, 0.0094643, -0.0352645, 0.0092633, -0.0525906, 0.0447288
1: 0.9317733, 1.0081939, 0.9319286, 1.0024198, -0.0706465, 0.0762652
2: -0.0168412, 0.0352591, -0.0167522, 0.0231811, -0.0400224, 0.0520113
3: -0.0281862, 0.0080895, -0.0209170, 0.0080140, -0.0362002, 0.0290065
4: -0.0275456, 0.0229022, -0.0178405, 0.0226684, -0.0502140, 0.0407427
5: -0.0064159, 0.0525165, -0.0057081, 0.0413246, -0.0477405, 0.0582246
6: -0.0102882, 0.0214731, -0.0100365, 0.0208619, -0.0311501, 0.0315096
7: -0.0113213, 0.0733469, -0.0111413, 0.0572950, -0.0686162, 0.0844883
8: -0.0067058, 0.0326073, -0.0063281, 0.0325844, -0.0392902, 0.0389354
9: -0.0381383, 0.0231929, -0.0226925, 0.0230478, -0.0611861, 0.0458854

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0699605, upper bound: 0.0735684
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0699605, upper bound: 0.0735684
time: 4.00 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0446047, 0.0093730, -0.0380427, 0.0093499, -0.0539547, 0.0474157
1: 0.9319888, 1.0110630, 0.9318516, 1.0023661, -0.0703773, 0.0792114
2: -0.0167237, 0.0370424, -0.0167960, 0.0273272, -0.0440509, 0.0538383
3: -0.0293139, 0.0079878, -0.0233864, 0.0080512, -0.0373651, 0.0313742
4: -0.0290554, 0.0225853, -0.0210080, 0.0227838, -0.0518393, 0.0435933
5: -0.0065436, 0.0541230, -0.0054926, 0.0453508, -0.0518944, 0.0596157
6: -0.0100762, 0.0216058, -0.0101518, 0.0205622, -0.0306384, 0.0317575
7: -0.0112497, 0.0758303, -0.0112182, 0.0629361, -0.0741859, 0.0870485
8: -0.0063466, 0.0325755, -0.0065040, 0.0325958, -0.0389423, 0.0390795
9: -0.0404770, 0.0229916, -0.0282395, 0.0231198, -0.0635968, 0.0512311

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 121
type: B, layer: 1, pos: 244
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 104

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0714365, upper bound: 0.0736116
time: 1.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0719595, upper bound: 0.0738859
time: 1.57 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.61 + 595.67 = 600.28 seconds
