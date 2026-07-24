## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.301048902


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8750806, 0.8750806)
1: (2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7192923, 0.7192924)
2: (-5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5813267, 0.5813267)
3: (-9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5080827, 0.5080827)
4: (-4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5576992, 0.5576992)
5: (-8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5290227, 0.5290227)
6: (-5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.9008565, 0.9008565)
7: (-3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7421594, 0.7421596)
8: (-3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4909875, 0.4909875)
9: (-10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8214734, 0.8214734)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.26 + 33.65 = 56.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.3040898, upper bound: 0.3040897

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 5830

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3024644, upper bound: 0.3035691
time: 2.95 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3040873, upper bound: 0.3040874
time: 3.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.42 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.42
Output dim: 1, lower bound: -0.3024644, upper bound: 0.3035691
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.42
Output dim: 1, lower bound: -0.3040873, upper bound: 0.3040874

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.4382954, -7.1752834, -8.4389114, -7.1738777, -0.8666420, 0.8659070
1: 2.3819056, 3.4039402, 2.3793869, 3.4045324, -0.7095212, 0.7112626
2: -5.0931301, -4.1387110, -5.0983739, -4.1378646, -0.5674740, 0.5717989
3: -9.8947601, -8.8094254, -9.8949184, -8.8077908, -0.5043954, 0.5034128
4: -4.4496756, -3.5807076, -4.4516602, -3.5805569, -0.5479462, 0.5493047
5: -8.1135330, -7.1427937, -8.1137352, -7.1412468, -0.5259295, 0.5247908
6: -5.5508862, -4.3392334, -5.5517235, -4.3350158, -0.8916821, 0.8884528
7: -3.9329557, -3.0853076, -3.9359465, -3.0850723, -0.7326467, 0.7341402
8: -3.4815402, -2.6167841, -3.4824743, -2.6138263, -0.4811343, 0.4799333
9: -10.7408276, -9.5616779, -10.7419491, -9.5596008, -0.8143659, 0.8139479

Time for backsubstitution: 22.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 5830

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3024644, upper bound: 0.3024642
time: 3.08 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3024644, upper bound: 0.3035691
time: 3.09 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.4438829, -7.1706562, -8.4397469, -7.1719217, -0.8759170, 0.8783095
1: 2.3737564, 3.4104815, 2.3759069, 3.4053383, -0.7198297, 0.7235858
2: -5.1078491, -4.1213388, -5.1055918, -4.1367168, -0.5783567, 0.5900027
3: -9.8966331, -8.8031025, -9.8951368, -8.8055763, -0.5082897, 0.5114066
4: -4.4560604, -3.5768299, -4.4544044, -3.5803545, -0.5595769, 0.5589397
5: -8.1190290, -7.1383047, -8.1140079, -7.1391029, -0.5340312, 0.5282943
6: -5.5649462, -4.3279800, -5.5528631, -4.3291779, -0.9120903, 0.8994408
7: -3.9421129, -3.0781431, -3.9400744, -3.0847542, -0.7406816, 0.7459872
8: -3.4926944, -2.6087871, -3.4837394, -2.6097507, -0.4941974, 0.4885275
9: -10.7508068, -9.5561171, -10.7434740, -9.5567427, -0.8282075, 0.8195248

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5830
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 5830

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3035693, upper bound: 0.3024643
time: 3.12 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3035693, upper bound: 0.3024661
time: 3.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.26 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.26
Output dim: 1, lower bound: -0.3024644, upper bound: 0.3024642
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.26
Output dim: 1, lower bound: -0.3024644, upper bound: 0.3035691
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.26
Output dim: 1, lower bound: -0.3035693, upper bound: 0.3024643
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.26
Output dim: 1, lower bound: -0.3035693, upper bound: 0.3024661

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -8.4382954, -7.1752834, -8.4382954, -7.1752834, -0.8639235, 0.8639238
1: 2.3819056, 3.4039402, 2.3819056, 3.4039402, -0.7080201, 0.7080202
2: -5.0931301, -4.1387110, -5.0931301, -4.1387110, -0.5665212, 0.5665213
3: -9.8947601, -8.8094254, -9.8947601, -8.8094254, -0.5027859, 0.5027859
4: -4.4496756, -3.5807076, -4.4496756, -3.5807076, -0.5462091, 0.5462091
5: -8.1135330, -7.1427937, -8.1135330, -7.1427937, -0.5243878, 0.5243878
6: -5.5508862, -4.3392334, -5.5508862, -4.3392334, -0.8872390, 0.8872390
7: -3.9329557, -3.0853076, -3.9329557, -3.0853076, -0.7310619, 0.7310617
8: -3.4815402, -2.6167841, -3.4815402, -2.6167841, -0.4777547, 0.4777547
9: -10.7408276, -9.5616779, -10.7408276, -9.5616779, -0.8122168, 0.8122168

Time for backsubstitution: 22.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3017221, upper bound: 0.3024451
time: 3.20 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3024433, upper bound: 0.3024443
time: 3.16 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.4382954, -7.1752834, -8.4438829, -7.1706562, -0.8689704, 0.8694842
1: 2.3819056, 3.4039402, 2.3737564, 3.4104815, -0.7159057, 0.7176225
2: -5.0931301, -4.1387110, -5.1078491, -4.1213388, -0.5774479, 0.5816410
3: -9.8947601, -8.8094254, -9.8966331, -8.8031025, -0.5084524, 0.5044919
4: -4.4496756, -3.5807076, -4.4560604, -3.5768299, -0.5515949, 0.5528193
5: -8.1135330, -7.1427937, -8.1190290, -7.1383047, -0.5289540, 0.5303491
6: -5.5508862, -4.3392334, -5.5649462, -4.3279800, -0.8988309, 0.9013674
7: -3.9329557, -3.0853076, -3.9421129, -3.0781431, -0.7386806, 0.7402840
8: -3.4815402, -2.6167841, -3.4926944, -2.6087871, -0.4856567, 0.4861515
9: -10.7408276, -9.5616779, -10.7508068, -9.5561171, -0.8180547, 0.8230989

Time for backsubstitution: 22.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3017221, upper bound: 0.3035491
time: 3.01 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3024433, upper bound: 0.3035483
time: 3.17 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.4438829, -7.1706562, -8.4382954, -7.1752834, -0.8694844, 0.8689704
1: 2.3737564, 3.4104815, 2.3819056, 3.4039402, -0.7176224, 0.7159057
2: -5.1078491, -4.1213388, -5.0931301, -4.1387110, -0.5816410, 0.5774478
3: -9.8966331, -8.8031025, -9.8947601, -8.8094254, -0.5044919, 0.5084524
4: -4.4560604, -3.5768299, -4.4496756, -3.5807076, -0.5528193, 0.5515947
5: -8.1190290, -7.1383047, -8.1135330, -7.1427937, -0.5303493, 0.5289543
6: -5.5649462, -4.3279800, -5.5508862, -4.3392334, -0.9013672, 0.8988309
7: -3.9421129, -3.0781431, -3.9329557, -3.0853076, -0.7402840, 0.7386806
8: -3.4926944, -2.6087871, -3.4815402, -2.6167841, -0.4861515, 0.4856567
9: -10.7508068, -9.5561171, -10.7408276, -9.5616779, -0.8230987, 0.8180544

Time for backsubstitution: 22.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3028267, upper bound: 0.3024444
time: 3.18 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3035480, upper bound: 0.3024436
time: 3.11 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.4438829, -7.1706562, -8.4438829, -7.1706562, -0.8799076, 0.8799074
1: 2.3737564, 3.4104815, 2.3737564, 3.4104815, -0.7209649, 0.7209649
2: -5.1078491, -4.1213388, -5.1078491, -4.1213388, -0.5831349, 0.5831350
3: -9.8966331, -8.8031025, -9.8966331, -8.8031025, -0.5117500, 0.5117499
4: -4.4560604, -3.5768299, -4.4560604, -3.5768299, -0.5600783, 0.5600781
5: -8.1190290, -7.1383047, -8.1190290, -7.1383047, -0.5302271, 0.5302272
6: -5.5649462, -4.3279800, -5.5649462, -4.3279800, -0.9027653, 0.9027653
7: -3.9421129, -3.0781431, -3.9421129, -3.0781431, -0.7416475, 0.7416475
8: -3.4926944, -2.6087871, -3.4926944, -2.6087871, -0.4893515, 0.4893515
9: -10.7508068, -9.5561171, -10.7508068, -9.5561171, -0.8216655, 0.8216655

Time for backsubstitution: 22.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4612
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3028270, upper bound: 0.3030202
time: 3.18 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3035482, upper bound: 0.3030213
time: 3.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.51 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 1, lower bound: -0.3017221, upper bound: 0.3024451
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 1, lower bound: -0.3024433, upper bound: 0.3024443
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 1, lower bound: -0.3017221, upper bound: 0.3035491
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 1, lower bound: -0.3024433, upper bound: 0.3035483
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 1, lower bound: -0.3028267, upper bound: 0.3024444
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 1, lower bound: -0.3035480, upper bound: 0.3024436
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 1, lower bound: -0.3028270, upper bound: 0.3030202
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.51
Output dim: 1, lower bound: -0.3035482, upper bound: 0.3030213

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.4363737, -7.1753426, -8.4382954, -7.1752834, -0.8619862, 0.8638639
1: 2.3827143, 3.4039025, 2.3819056, 3.4039402, -0.7069417, 0.7078170
2: -5.0930705, -4.1388593, -5.0931301, -4.1387110, -0.5663991, 0.5662239
3: -9.8947420, -8.8102312, -9.8947601, -8.8094254, -0.5027654, 0.5019170
4: -4.4495420, -3.5841985, -4.4496756, -3.5807076, -0.5461516, 0.5427203
5: -8.1131649, -7.1428618, -8.1135330, -7.1427937, -0.5238686, 0.5242369
6: -5.5507784, -4.3399501, -5.5508862, -4.3392334, -0.8871343, 0.8864920
7: -3.9321468, -3.0853522, -3.9329557, -3.0853076, -0.7302461, 0.7310238
8: -3.4814396, -2.6168127, -3.4815402, -2.6167841, -0.4776219, 0.4777251
9: -10.7407312, -9.5626783, -10.7408276, -9.5616779, -0.8121276, 0.8111861

Time for backsubstitution: 22.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3017229, upper bound: 0.3017228
time: 3.19 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3017229, upper bound: 0.3024442
time: 3.18 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.4400244, -7.1403809, -8.4382753, -7.1752834, -0.8667426, 0.8728340
1: 2.3780823, 3.4185543, 2.3819094, 3.4039402, -0.7187558, 0.7166615
2: -5.0977449, -4.1371560, -5.0931292, -4.1387124, -0.5712957, 0.5713726
3: -9.9091234, -8.8073149, -9.8947592, -8.8094339, -0.5072722, 0.5065590
4: -4.5083599, -3.5806336, -4.4496722, -3.5807242, -0.5574462, 0.5502954
5: -8.1151543, -7.1341281, -8.1135292, -7.1427946, -0.5292084, 0.5294564
6: -5.5681925, -4.3384624, -5.5508862, -4.3392434, -0.9001775, 0.8890686
7: -3.9344769, -3.0713518, -3.9329460, -3.0853078, -0.7332315, 0.7374687
8: -3.4825525, -2.6130266, -3.4815388, -2.6167836, -0.4792539, 0.4813540
9: -10.7599936, -9.5605202, -10.7408247, -9.5616827, -0.8251708, 0.8142967

Time for backsubstitution: 22.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3024445, upper bound: 0.3017225
time: 3.14 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3024445, upper bound: 0.3024442
time: 3.20 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.4363737, -7.1753426, -8.4438829, -7.1706562, -0.8670325, 0.8694243
1: 2.3827143, 3.4039025, 2.3737564, 3.4104815, -0.7148273, 0.7174194
2: -5.0930705, -4.1388593, -5.1078491, -4.1213388, -0.5773253, 0.5813437
3: -9.8947420, -8.8102312, -9.8966331, -8.8031025, -0.5084318, 0.5036230
4: -4.4495420, -3.5841985, -4.4560604, -3.5768299, -0.5515374, 0.5493305
5: -8.1131649, -7.1428618, -8.1190290, -7.1383047, -0.5284350, 0.5301982
6: -5.5507784, -4.3399501, -5.5649462, -4.3279800, -0.8987262, 0.9006186
7: -3.9321468, -3.0853522, -3.9421129, -3.0781431, -0.7378647, 0.7402458
8: -3.4814396, -2.6168127, -3.4926944, -2.6087871, -0.4855198, 0.4861219
9: -10.7407312, -9.5626783, -10.7508068, -9.5561171, -0.8179650, 0.8220682

Time for backsubstitution: 22.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3017221, upper bound: 0.3028266
time: 3.17 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3017221, upper bound: 0.3035480
time: 3.19 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.4400244, -7.1403809, -8.4438648, -7.1706567, -0.8717890, 0.8770919
1: 2.3780823, 3.4185543, 2.3737588, 3.4104815, -0.7246008, 0.7238435
2: -5.0977449, -4.1371560, -5.1078482, -4.1213403, -0.5799952, 0.5840050
3: -9.9091234, -8.8073149, -9.8966331, -8.8031101, -0.5126858, 0.5082648
4: -4.5083599, -3.5806336, -4.4560580, -3.5768456, -0.5590748, 0.5569055
5: -8.1151543, -7.1341281, -8.1190243, -7.1383047, -0.5335257, 0.5318738
6: -5.5681925, -4.3384624, -5.5649447, -4.3279896, -0.9074731, 0.9026229
7: -3.9344769, -3.0713518, -3.9421024, -3.0781417, -0.7408500, 0.7429266
8: -3.4825525, -2.6130266, -3.4926934, -2.6087890, -0.4869685, 0.4876581
9: -10.7599936, -9.5605202, -10.7508049, -9.5561237, -0.8278763, 0.8251784

Time for backsubstitution: 22.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3024437, upper bound: 0.3028263
time: 3.20 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3024437, upper bound: 0.3035480
time: 3.19 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.4419622, -7.1707163, -8.4382954, -7.1752834, -0.8675470, 0.8689091
1: 2.3745656, 3.4104443, 2.3819056, 3.4039402, -0.7165428, 0.7157018
2: -5.1077886, -4.1214867, -5.0931301, -4.1387110, -0.5815196, 0.5771493
3: -9.8966160, -8.8039112, -9.8947601, -8.8094254, -0.5044715, 0.5075828
4: -4.4559250, -3.5803204, -4.4496756, -3.5807076, -0.5527601, 0.5481057
5: -8.1186619, -7.1383724, -8.1135330, -7.1427937, -0.5298307, 0.5288031
6: -5.5648351, -4.3286977, -5.5508862, -4.3392334, -0.9012570, 0.8980868
7: -3.9413059, -3.0781868, -3.9329557, -3.0853076, -0.7394695, 0.7386429
8: -3.4925928, -2.6088181, -3.4815402, -2.6167841, -0.4860146, 0.4856271
9: -10.7507076, -9.5571203, -10.7408276, -9.5616779, -0.8230071, 0.8170238

Time for backsubstitution: 22.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3028267, upper bound: 0.3017219
time: 3.12 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3028267, upper bound: 0.3024435
time: 3.21 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.4456215, -7.1357532, -8.4382753, -7.1752834, -0.8723230, 0.8778443
1: 2.3699341, 3.4250832, 2.3819094, 3.4039402, -0.7279794, 0.7203619
2: -5.1124411, -4.1198139, -5.0931292, -4.1387124, -0.5845563, 0.5794090
3: -9.9109955, -8.8010159, -9.8947592, -8.8094339, -0.5085623, 0.5122252
4: -4.5147324, -3.5767553, -4.4496722, -3.5807242, -0.5621600, 0.5556786
5: -8.1206493, -7.1296282, -8.1135292, -7.1427946, -0.5330256, 0.5323696
6: -5.5822797, -4.3271990, -5.5508862, -4.3392434, -0.9056373, 0.9006767
7: -3.9436777, -3.0641797, -3.9329460, -3.0853078, -0.7425146, 0.7402091
8: -3.4937057, -2.6050100, -3.4815388, -2.6167836, -0.4874631, 0.4871749
9: -10.7699337, -9.5549631, -10.7408247, -9.5616827, -0.8312774, 0.8201342

Time for backsubstitution: 23.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3035483, upper bound: 0.3017219
time: 3.17 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3035483, upper bound: 0.3024435
time: 3.18 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.4419622, -7.1707163, -8.4438829, -7.1706562, -0.8779702, 0.8798466
1: 2.3745656, 3.4104443, 2.3737564, 3.4104815, -0.7198851, 0.7207618
2: -5.1077886, -4.1214867, -5.1078491, -4.1213388, -0.5830138, 0.5828369
3: -9.8966160, -8.8039112, -9.8966331, -8.8031025, -0.5117296, 0.5108803
4: -4.4559250, -3.5803204, -4.4560604, -3.5768299, -0.5600219, 0.5565896
5: -8.1186619, -7.1383724, -8.1190290, -7.1383047, -0.5297083, 0.5300760
6: -5.5648351, -4.3286977, -5.5649462, -4.3279800, -0.9026563, 0.9020188
7: -3.9413059, -3.0781868, -3.9421129, -3.0781431, -0.7408307, 0.7416093
8: -3.4925928, -2.6088181, -3.4926944, -2.6087871, -0.4892194, 0.4893212
9: -10.7507076, -9.5571203, -10.7508068, -9.5561171, -0.8215773, 0.8206348

Time for backsubstitution: 22.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3033449, upper bound: 0.3022978
time: 3.25 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3033449, upper bound: 0.3030194
time: 3.30 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.4456215, -7.1357532, -8.4438648, -7.1706567, -0.8827205, 0.8851106
1: 2.3699341, 3.4250832, 2.3737588, 3.4104815, -0.7316623, 0.7300066
2: -5.1124411, -4.1198139, -5.1078482, -4.1213403, -0.5878775, 0.5879633
3: -9.9109955, -8.8010159, -9.8966331, -8.8031101, -0.5154161, 0.5155226
4: -4.5147324, -3.5767553, -4.4560580, -3.5768456, -0.5675896, 0.5641649
5: -8.1206493, -7.1296282, -8.1190243, -7.1383047, -0.5350503, 0.5348346
6: -5.5822797, -4.3271990, -5.5649447, -4.3279896, -0.9161267, 0.9045873
7: -3.9436777, -3.0641797, -3.9421024, -3.0781417, -0.7438049, 0.7485101
8: -3.4937057, -2.6050100, -3.4926934, -2.6087890, -0.4908513, 0.4929590
9: -10.7699337, -9.5549631, -10.7508049, -9.5561237, -0.8347461, 0.8237448

Time for backsubstitution: 23.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4612
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4612

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3040666, upper bound: 0.3022978
time: 3.35 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3040666, upper bound: 0.3030194
time: 3.39 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.08 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3017229, upper bound: 0.3017228
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3017229, upper bound: 0.3024442
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3024445, upper bound: 0.3017225
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3024445, upper bound: 0.3024442
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3017221, upper bound: 0.3028266
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3017221, upper bound: 0.3035480
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3024437, upper bound: 0.3028263
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3024437, upper bound: 0.3035480
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3028267, upper bound: 0.3017219
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3028267, upper bound: 0.3024435
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3035483, upper bound: 0.3017219
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3035483, upper bound: 0.3024435
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3033449, upper bound: 0.3022978
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3033449, upper bound: 0.3030194
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3040666, upper bound: 0.3022978
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.08
Output dim: 1, lower bound: -0.3040666, upper bound: 0.3030194

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.4363737, -7.1753426, -8.4363737, -7.1753426, -0.8619266, 0.8619261
1: 2.3827143, 3.4039025, 2.3827143, 3.4039025, -0.7067386, 0.7067385
2: -5.0930705, -4.1388593, -5.0930705, -4.1388593, -0.5661018, 0.5661017
3: -9.8947420, -8.8102312, -9.8947420, -8.8102312, -0.5018965, 0.5018964
4: -4.4495420, -3.5841985, -4.4495420, -3.5841985, -0.5426629, 0.5426630
5: -8.1131649, -7.1428618, -8.1131649, -7.1428618, -0.5237176, 0.5237178
6: -5.5507784, -4.3399501, -5.5507784, -4.3399501, -0.8863873, 0.8863873
7: -3.9321468, -3.0853522, -3.9321468, -3.0853522, -0.7302084, 0.7302079
8: -3.4814396, -2.6168127, -3.4814396, -2.6168127, -0.4775923, 0.4775922
9: -10.7407312, -9.5626783, -10.7407312, -9.5626783, -0.8110964, 0.8110967

Time for backsubstitution: 22.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 37

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3008144, upper bound: 0.2948372
time: 3.18 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3017225, upper bound: 0.3017237
time: 3.25 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.4363737, -7.1753426, -8.4400244, -7.1403809, -0.8708835, 0.8649054
1: 2.3827143, 3.4039025, 2.3780823, 3.4185543, -0.7155811, 0.7122463
2: -5.0930705, -4.1388593, -5.0977449, -4.1371560, -0.5679983, 0.5710003
3: -9.8947420, -8.8102312, -9.9091234, -8.8073149, -0.5046215, 0.5064059
4: -4.4495420, -3.5841985, -4.5083599, -3.5806336, -0.5462331, 0.5539520
5: -8.1131649, -7.1428618, -8.1151543, -7.1341281, -0.5289365, 0.5260043
6: -5.5507784, -4.3399501, -5.5681925, -4.3384624, -0.8881347, 0.8994293
7: -3.9321468, -3.0853522, -3.9344769, -3.0713518, -0.7366509, 0.7324612
8: -3.4814396, -2.6168127, -3.4825525, -2.6130266, -0.4812227, 0.4789295
9: -10.7407312, -9.5626783, -10.7599936, -9.5605202, -0.8130531, 0.8241386

Time for backsubstitution: 22.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.3008144, upper bound: 0.2955581
time: 3.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3017225, upper bound: 0.3024447
time: 3.13 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.4400244, -7.1403809, -8.4363737, -7.1753426, -0.8649054, 0.8708839
1: 2.3780823, 3.4185543, 2.3827143, 3.4039025, -0.7122463, 0.7155812
2: -5.0977449, -4.1371560, -5.0930705, -4.1388593, -0.5710003, 0.5679982
3: -9.9091234, -8.8073149, -9.8947420, -8.8102312, -0.5064059, 0.5046215
4: -4.5083599, -3.5806336, -4.4495420, -3.5841985, -0.5539521, 0.5462331
5: -8.1151543, -7.1341281, -8.1131649, -7.1428618, -0.5260043, 0.5289365
6: -5.5681925, -4.3384624, -5.5507784, -4.3399501, -0.8994293, 0.8881347
7: -3.9344769, -3.0713518, -3.9321468, -3.0853522, -0.7324610, 0.7366509
8: -3.4825525, -2.6130266, -3.4814396, -2.6168127, -0.4789295, 0.4812227
9: -10.7599936, -9.5605202, -10.7407312, -9.5626783, -0.8241386, 0.8130531

Time for backsubstitution: 22.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 37

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 916

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3015356, upper bound: 0.2948358
time: 3.25 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3024437, upper bound: 0.3017223
time: 3.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.4400244, -7.1403809, -8.4400244, -7.1403809, -0.8739123, 0.8756647
1: 2.3780823, 3.4185543, 2.3780823, 3.4185543, -0.7199962, 0.7199963
2: -5.0977449, -4.1371560, -5.0977449, -4.1371560, -0.5739927, 0.5739926
3: -9.9091234, -8.8073149, -9.9091234, -8.8073149, -0.5071785, 0.5071785
4: -4.5083599, -3.5806336, -4.5083599, -3.5806336, -0.5556062, 0.5556062
5: -8.1151543, -7.1341281, -8.1151543, -7.1341281, -0.5301942, 0.5301942
6: -5.5681925, -4.3384624, -5.5681925, -4.3384624, -0.8944099, 0.8944099
7: -3.9344769, -3.0713518, -3.9344769, -3.0713518, -0.7375228, 0.7375231
8: -3.4825525, -2.6130266, -3.4825525, -2.6130266, -0.4818724, 0.4818724
9: -10.7599936, -9.5605202, -10.7599936, -9.5605202, -0.8186293, 0.8186297

Time for backsubstitution: 22.67 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.91 + 546.20 = 603.11 seconds
