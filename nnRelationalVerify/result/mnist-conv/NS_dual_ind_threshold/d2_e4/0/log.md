## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.276117024


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6436245, 0.6436245)
1: (-2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7583723, 0.7583723)
2: (-3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8736706, 0.8736706)
3: (-12.2057600, -10.5023746, -12.2057600, -10.5023746, -1.0068245, 1.0068247)
4: (-5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8551059, 0.8551056)
5: (-2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7331014, 0.7331014)
6: (2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7932694, 0.7932692)
7: (-9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8936715, 0.8936715)
8: (-1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0782151, 1.0782149)
9: (-8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6931071, 0.6931071)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.15 + 33.32 = 55.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.2876219, upper bound: 0.2876191

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 430

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 430

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2872995, upper bound: 0.2800688
time: 3.34 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2876152, upper bound: 0.2876110
time: 3.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.02 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.02
Output dim: 6, lower bound: -0.2872995, upper bound: 0.2800688
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.02
Output dim: 6, lower bound: -0.2876152, upper bound: 0.2876110

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.4763131, -3.4187107, -4.4791183, -3.4178977, -0.6271455, 0.6292549
1: -2.4403651, -1.2178694, -2.4426517, -1.2172081, -0.7418343, 0.7429466
2: -3.8639922, -2.7615929, -3.8676326, -2.7597961, -0.8639030, 0.8654759
3: -12.1947575, -10.5038862, -12.2003813, -10.5030346, -0.9906936, 0.9953420
4: -5.6800690, -4.6543694, -5.6813374, -4.6508598, -0.7983177, 0.7961638
5: -2.4825191, -1.3682163, -2.4843731, -1.3566430, -0.7138705, 0.7042453
6: 2.6300783, 3.5720191, 2.6274977, 3.5815935, -0.7834466, 0.7756550
7: -9.8659668, -8.5775337, -9.8772621, -8.5767174, -0.8627820, 0.8736978
8: -1.3571384, 0.3021569, -1.3596635, 0.3047533, -1.0301645, 1.0324874
9: -8.2100067, -7.3067718, -8.2101517, -7.3060036, -0.6568842, 0.6560706

Time for backsubstitution: 20.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 430

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2800700, upper bound: 0.2800696
time: 3.46 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2800700, upper bound: 0.2800681
time: 4.83 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -4.4821129, -3.4050286, -4.4819708, -3.4171567, -0.6430843, 0.6506436
1: -2.4484644, -1.2162095, -2.4448371, -1.2164094, -0.7605793, 0.7576809
2: -3.8718791, -2.7399030, -3.8712258, -2.7582927, -0.8720870, 0.8867257
3: -12.2068901, -10.4748745, -12.2057428, -10.5023785, -1.0044107, 1.0261426
4: -5.7034121, -4.6459498, -5.6823087, -4.6463857, -0.8700294, 0.8640258
5: -2.5399332, -1.3448799, -2.4859028, -1.3458261, -0.7429976, 0.7257071
6: 2.5826893, 3.5917764, 2.6252389, 3.5906596, -0.8059969, 0.7873921
7: -9.8894730, -8.5253963, -9.8878231, -8.5758915, -0.8845136, 0.8999314
8: -1.3676722, 0.3077548, -1.3620286, 0.3071899, -1.0842345, 1.0781050
9: -8.2120647, -7.3031225, -8.2113304, -7.3053508, -0.6995313, 0.6899536

Time for backsubstitution: 21.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 430

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 430

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2800700, upper bound: 0.2872990
time: 3.57 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2800700, upper bound: 0.2876151
time: 3.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.75 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.75
Output dim: 6, lower bound: -0.2800700, upper bound: 0.2800696
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.75
Output dim: 6, lower bound: -0.2800700, upper bound: 0.2800681
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.75
Output dim: 6, lower bound: -0.2800700, upper bound: 0.2872990
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.75
Output dim: 6, lower bound: -0.2800700, upper bound: 0.2876151

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4.4763131, -3.4187107, -4.4763131, -3.4187107, -0.6262934, 0.6262934
1: -2.4403651, -1.2178694, -2.4403651, -1.2178694, -0.7407734, 0.7407733
2: -3.8639922, -2.7615929, -3.8639922, -2.7615929, -0.8618259, 0.8618257
3: -12.1947575, -10.5038862, -12.1947575, -10.5038862, -0.9892020, 0.9892020
4: -5.6800690, -4.6543694, -5.6800690, -4.6543694, -0.7947912, 0.7947912
5: -2.4825191, -1.3682163, -2.4825191, -1.3682163, -0.7021401, 0.7021401
6: 2.6300783, 3.5720191, 2.6300783, 3.5720191, -0.7738664, 0.7738664
7: -9.8659668, -8.5775337, -9.8659668, -8.5775337, -0.8624976, 0.8624976
8: -1.3571384, 0.3021569, -1.3571384, 0.3021569, -1.0276327, 1.0276325
9: -8.2100067, -7.3067718, -8.2100067, -7.3067718, -0.6542296, 0.6542295

Time for backsubstitution: 21.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 773
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1837
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 1745
type: A, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 2810

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2722922, upper bound: 0.2775057
time: 3.65 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775029, upper bound: 0.2775055
time: 3.54 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.4763131, -3.4187107, -4.4817119, -3.4050446, -0.6362967, 0.6319025
1: -2.4403651, -1.2178694, -2.4483716, -1.2164898, -0.7425821, 0.7484820
2: -3.8639922, -2.7615929, -3.8716097, -2.7403965, -0.8776002, 0.8692923
3: -12.1947575, -10.5038862, -12.2067585, -10.4752293, -1.0112722, 1.0018458
4: -5.6800690, -4.6543694, -5.7029448, -4.6472664, -0.8019617, 0.8136096
5: -2.4825191, -1.3682163, -2.5390143, -1.3448873, -0.7205102, 0.7146986
6: 2.6300783, 3.5720191, 2.5829020, 3.5915785, -0.7934318, 0.7903395
7: -9.8659668, -8.5775337, -9.8893509, -8.5261831, -0.8683004, 0.8772395
8: -1.3571384, 0.3021569, -1.3670413, 0.3077388, -1.0330138, 1.0386961
9: -8.2100067, -7.3067718, -8.2108936, -7.3033404, -0.6576204, 0.6548942

Time for backsubstitution: 21.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 773
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1837
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 1745
type: A, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.40 seconds

### Candidate
type: A, layer: 3, pos: 2810

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2722922, upper bound: 0.2775062
time: 3.81 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775029, upper bound: 0.2775063
time: 3.84 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -4.4819059, -3.4050446, -4.4763131, -3.4187107, -0.6396747, 0.6362972
1: -2.4483724, -1.2163609, -2.4403651, -1.2178694, -0.7484829, 0.7526927
2: -3.8718536, -2.7403960, -3.8639922, -2.7615929, -0.8703938, 0.8776181
3: -12.2068796, -10.4752235, -12.1947575, -10.5038862, -1.0042696, 1.0112753
4: -5.7029476, -4.6459694, -5.6800690, -4.6543694, -0.8136110, 0.8420336
5: -2.5390897, -1.3448875, -2.4825191, -1.3682163, -0.7195721, 0.7205218
6: 2.5829015, 3.5917015, 2.6300783, 3.5720191, -0.7903776, 0.7906997
7: -9.8893538, -8.5260143, -9.8659668, -8.5775337, -0.8772397, 0.8776357
8: -1.3671050, 0.3077409, -1.3571384, 0.3021569, -1.0764651, 1.0330150
9: -8.2120361, -7.3033404, -8.2100067, -7.3067718, -0.6824195, 0.6576375

Time for backsubstitution: 21.45 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 773
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1837
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 1745
type: A, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.48 seconds

### Candidate
type: A, layer: 3, pos: 2810

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2722922, upper bound: 0.2847251
time: 3.57 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775029, upper bound: 0.2847251
time: 3.75 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -4.4821587, -3.4050150, -4.4821587, -3.4050150, -0.6444249, 0.6444249
1: -2.4485419, -1.2161297, -2.4485419, -1.2161297, -0.7612450, 0.7612449
2: -3.8718958, -2.7395420, -3.8718958, -2.7395420, -0.8811352, 0.8811355
3: -12.2068977, -10.4746428, -12.2068977, -10.4746428, -1.0054193, 1.0054193
4: -5.7037096, -4.6459417, -5.7037096, -4.6459417, -0.8674450, 0.8674450
5: -2.5405402, -1.3448751, -2.5405402, -1.3448751, -0.7437136, 0.7358339
6: 2.5825589, 3.5918446, 2.5825589, 3.5918446, -0.8002436, 0.8002436
7: -9.8895855, -8.5248966, -9.8895855, -8.5248966, -0.8912404, 0.8941152
8: -1.3681924, 0.3077681, -1.3681924, 0.3077681, -1.0851841, 1.0851839
9: -8.2120819, -7.3029590, -8.2120819, -7.3029590, -0.7005031, 0.7033517

Time for backsubstitution: 21.25 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 773
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1837
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 1745
type: A, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 2810

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2722922, upper bound: 0.2850105
time: 3.68 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775029, upper bound: 0.2850120
time: 3.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.99 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 6, lower bound: -0.2722922, upper bound: 0.2775057
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 6, lower bound: -0.2775029, upper bound: 0.2775055
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 6, lower bound: -0.2722922, upper bound: 0.2775062
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 6, lower bound: -0.2775029, upper bound: 0.2775063
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 6, lower bound: -0.2722922, upper bound: 0.2847251
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 6, lower bound: -0.2775029, upper bound: 0.2847251
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 6, lower bound: -0.2722922, upper bound: 0.2850105
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.99
Output dim: 6, lower bound: -0.2775029, upper bound: 0.2850120

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -4.4760594, -3.4235706, -4.4761744, -3.4213986, -0.6202579, 0.6170518
1: -2.4355426, -1.2178701, -2.4376569, -1.2178704, -0.7340806, 0.7343739
2: -3.8488879, -2.7630563, -3.8555377, -2.7624540, -0.8325579, 0.8423145
3: -12.1815214, -10.5058784, -12.1874428, -10.5051088, -0.9777832, 0.9803753
4: -5.6726866, -4.6544447, -5.6758108, -4.6544137, -0.7714894, 0.7812300
5: -2.4816794, -1.3699217, -2.4820483, -1.3691585, -0.6899691, 0.6893561
6: 2.6583705, 3.5718725, 2.6475103, 3.5719342, -0.7475669, 0.7572668
7: -9.8657866, -8.5793180, -9.8658581, -8.5785189, -0.8552787, 0.8551943
8: -1.3554013, 0.2870252, -1.3560843, 0.2936630, -1.0157170, 1.0084987
9: -8.2099228, -7.3139319, -8.2099533, -7.3107281, -0.6509838, 0.6484611

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1837
type: B, layer: 3, pos: 2588
type: B, layer: 3, pos: 1389
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 1745
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 578

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 1249

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2672743, upper bound: 0.2747460
time: 3.80 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2675894, upper bound: 0.2747459
time: 3.83 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.4794407, -3.4192529, -4.4759431, -3.4194927, -0.6293275, 0.6178758
1: -2.4294529, -1.2213348, -2.4335370, -1.2178704, -0.7262690, 0.7198180
2: -3.8602819, -2.7490425, -3.8618855, -2.7622621, -0.8179493, 0.8684802
3: -12.1891241, -10.4931202, -12.1907673, -10.5047970, -0.9818068, 0.9775870
4: -5.6638212, -4.6585722, -5.6685781, -4.6544294, -0.7691791, 0.8154247
5: -2.4821122, -1.3722084, -2.4820392, -1.3704979, -0.6939967, 0.6897616
6: 2.6392365, 3.6088099, 2.6384892, 3.5719638, -0.7535143, 0.7842574
7: -9.8670902, -8.5793877, -9.8659134, -8.5786400, -0.8550754, 0.8608162
8: -1.3587065, 0.2789605, -1.3562143, 0.2866135, -1.0380015, 1.0035658
9: -8.2171974, -7.3114614, -8.2099838, -7.3102832, -0.6501698, 0.6492432

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1837
type: B, layer: 3, pos: 2588
type: B, layer: 3, pos: 1389
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 1745
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 578

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 1249

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2664218, upper bound: 0.2747446
time: 3.94 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2747446, upper bound: 0.2747438
time: 3.83 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.4760594, -3.4235706, -4.4815731, -3.4076967, -0.6300135, 0.6226614
1: -2.4355426, -1.2178701, -2.4456644, -1.2164897, -0.7358896, 0.7420681
2: -3.8488879, -2.7630563, -3.8631501, -2.7412932, -0.8463380, 0.8497798
3: -12.1815214, -10.5058784, -12.1994572, -10.4764500, -0.9979100, 0.9930131
4: -5.6726866, -4.6544447, -5.6986737, -4.6473126, -0.7786572, 0.8000319
5: -2.4816794, -1.3699217, -2.5385232, -1.3458300, -0.7080551, 0.7016655
6: 2.6583705, 3.5718725, 2.6002614, 3.5914922, -0.7660906, 0.7730165
7: -9.8657866, -8.5793180, -9.8892241, -8.5271721, -0.8607659, 0.8692458
8: -1.3554013, 0.2870252, -1.3659909, 0.2992182, -1.0211000, 1.0195708
9: -8.2099228, -7.3139319, -8.2108393, -7.3072901, -0.6543756, 0.6491239

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1837
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 2588
type: B, layer: 3, pos: 1389
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 1745
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 578

Time for candidate selection: 0.48 seconds

### Candidate
type: B, layer: 3, pos: 1249

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2672730, upper bound: 0.2747466
time: 4.09 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2748138, upper bound: 0.2747437
time: 3.80 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.4794407, -3.4192529, -4.4813442, -3.4058356, -0.6382818, 0.6234860
1: -2.4294529, -1.2213348, -2.4416051, -1.2164901, -0.7280781, 0.7275997
2: -3.8602819, -2.7490425, -3.8695045, -2.7411094, -0.8340435, 0.8759506
3: -12.1891241, -10.4931202, -12.2028465, -10.4761372, -1.0024705, 0.9902265
4: -5.6638212, -4.6585722, -5.6913824, -4.6473284, -0.7763669, 0.8343320
5: -2.4821122, -1.3722084, -2.5385118, -1.3471694, -0.7128775, 0.7020969
6: 2.6392365, 3.6088099, 2.5914917, 3.5915248, -0.7730668, 0.8043005
7: -9.8670902, -8.5793877, -9.8892794, -8.5272913, -0.8616660, 0.8744922
8: -1.3587065, 0.2789605, -1.3661208, 0.2921984, -1.0433629, 1.0146265
9: -8.2171974, -7.3114614, -8.2108727, -7.3068452, -0.6535678, 0.6499076

Time for backsubstitution: 22.34 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1837
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 2588
type: B, layer: 3, pos: 1389
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 1745
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 578

Time for candidate selection: 0.55 seconds

### Candidate
type: B, layer: 3, pos: 1249

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2736882, upper bound: 0.2747447
time: 4.33 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2819922, upper bound: 0.2747458
time: 4.09 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.4816532, -3.4098334, -4.4761744, -3.4213986, -0.6335671, 0.6271309
1: -2.4435568, -1.2163619, -2.4376569, -1.2178704, -0.7417691, 0.7463682
2: -3.8567455, -2.7419195, -3.8555377, -2.7624540, -0.8413041, 0.8569704
3: -12.1936703, -10.4772148, -12.1874428, -10.5051088, -0.9925857, 1.0014596
4: -5.6955452, -4.6460476, -5.6758108, -4.6544137, -0.7902806, 0.8288331
5: -2.5382149, -1.3465924, -2.4820483, -1.3691585, -0.7070967, 0.7074630
6: 2.6111660, 3.5915539, 2.6475103, 3.5719342, -0.7627702, 0.7738171
7: -9.8891439, -8.5278053, -9.8658581, -8.5785189, -0.8697276, 0.8697457
8: -1.3653715, 0.2925882, -1.3560843, 0.2936630, -1.0645881, 1.0138960
9: -8.2119484, -7.3104868, -8.2099533, -7.3107281, -0.6793013, 0.6518729

Time for backsubstitution: 22.64 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1837
type: B, layer: 3, pos: 2588
type: B, layer: 3, pos: 1389
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 1745
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 578

Time for candidate selection: 0.51 seconds

### Candidate
type: B, layer: 3, pos: 1249

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2672724, upper bound: 0.2819926
time: 3.83 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2675894, upper bound: 0.2819936
time: 3.86 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.4850364, -3.4056401, -4.4759431, -3.4194927, -0.6425924, 0.6282566
1: -2.4376385, -1.2198263, -2.4335370, -1.2178704, -0.7341890, 0.7316126
2: -3.8681226, -2.7277551, -3.8618855, -2.7622621, -0.8264871, 0.8861420
3: -12.2013512, -10.4644318, -12.1907673, -10.5047970, -0.9967113, 1.0020463
4: -5.6866088, -4.6501760, -5.6685781, -4.6544294, -0.7879193, 0.8620598
5: -2.5385885, -1.3488789, -2.4820392, -1.3704979, -0.7119331, 0.7079092
6: 2.5923219, 3.6287284, 2.6384892, 3.5719638, -0.7698095, 0.7992654
7: -9.8904943, -8.5278759, -9.8659134, -8.5786400, -0.8707142, 0.8746154
8: -1.3686960, 0.2846043, -1.3562143, 0.2866135, -1.0843725, 1.0090609
9: -8.2192459, -7.3080463, -8.2099838, -7.3102832, -0.6777391, 0.6526514

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1837
type: B, layer: 3, pos: 2588
type: B, layer: 3, pos: 1389
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 1745
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 578

Time for candidate selection: 0.55 seconds

### Candidate
type: B, layer: 3, pos: 1249

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2664218, upper bound: 0.2819922
time: 4.09 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2747446, upper bound: 0.2819915
time: 3.61 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.4819045, -3.4098053, -4.4820194, -3.4076667, -0.6383550, 0.6352353
1: -2.4437263, -1.2161298, -2.4458342, -1.2161293, -0.7546678, 0.7549057
2: -3.8567867, -2.7410655, -3.8634357, -2.7404366, -0.8520515, 0.8619831
3: -12.1936874, -10.4766350, -12.1995974, -10.4758644, -0.9937086, 0.9964516
4: -5.6963038, -4.6460199, -5.6994371, -4.6459880, -0.8448098, 0.8542378
5: -2.5396657, -1.3465800, -2.5400500, -1.3458166, -0.7312168, 0.7228584
6: 2.6108208, 3.5916958, 2.5999153, 3.5917575, -0.7749004, 0.7834954
7: -9.8893757, -8.5266876, -9.8894577, -8.5258846, -0.8839111, 0.8869805
8: -1.3664598, 0.2926173, -1.3671415, 0.2992465, -1.0733082, 1.0660901
9: -8.2119942, -7.3101048, -8.2120285, -7.3069077, -0.6975822, 0.6981721

Time for backsubstitution: 22.46 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1837
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 2588
type: B, layer: 3, pos: 1389
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 1745
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 578

Time for candidate selection: 0.58 seconds

### Candidate
type: B, layer: 3, pos: 1249

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2679735, upper bound: 0.2822794
time: 3.96 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2683012, upper bound: 0.2822802
time: 4.06 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.4852877, -3.4056113, -4.4817896, -3.4058056, -0.6473429, 0.6361766
1: -2.4378080, -1.2195947, -2.4417746, -1.2161293, -0.7468181, 0.7402394
2: -3.8681669, -2.7268984, -3.8697712, -2.7402537, -0.8372662, 0.8892319
3: -12.2013674, -10.4638500, -12.2029877, -10.4755545, -0.9977946, 0.9942739
4: -5.6873465, -4.6501484, -5.6921482, -4.6460042, -0.8429503, 0.8874662
5: -2.5400398, -1.3488665, -2.5400379, -1.3471568, -0.7360729, 0.7237792
6: 2.5919826, 3.6288705, 2.5911512, 3.5917897, -0.7794807, 0.8086147
7: -9.8907261, -8.5267563, -9.8895130, -8.5260057, -0.8849223, 0.8921227
8: -1.3697760, 0.2846313, -1.3672681, 0.2922246, -1.0930533, 1.0612361
9: -8.2192917, -7.3076420, -8.2120590, -7.3064404, -0.6948557, 0.6986967

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1837
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 2588
type: B, layer: 3, pos: 1389
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 1745
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 578

Time for candidate selection: 0.49 seconds

### Candidate
type: B, layer: 3, pos: 1249

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2672103, upper bound: 0.2822798
time: 3.92 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2754637, upper bound: 0.2822820
time: 4.09 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.69 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2672743, upper bound: 0.2747460
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2675894, upper bound: 0.2747459
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2664218, upper bound: 0.2747446
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2747446, upper bound: 0.2747438
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2672730, upper bound: 0.2747466
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2748138, upper bound: 0.2747437
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2736882, upper bound: 0.2747447
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2819922, upper bound: 0.2747458
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2672724, upper bound: 0.2819926
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2675894, upper bound: 0.2819936
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2664218, upper bound: 0.2819922
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2747446, upper bound: 0.2819915
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2679735, upper bound: 0.2822794
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2683012, upper bound: 0.2822802
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2672103, upper bound: 0.2822798
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 6, lower bound: -0.2754637, upper bound: 0.2822820

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.4784369, -3.4192991, -4.4802265, -3.4051275, -0.6386728, 0.5811987
1: -2.4249430, -1.2213348, -2.4412754, -1.2164898, -0.7235212, 0.6832883
2: -3.8602395, -2.7550564, -3.8715367, -2.7506952, -0.8105707, 0.8715456
3: -12.1762133, -10.4932661, -12.1821690, -10.4754639, -0.9962935, 0.9634690
4: -5.6637163, -4.6681023, -5.7028084, -4.6672797, -0.7266119, 0.8321879
5: -2.4820445, -1.3757205, -2.5388985, -1.3509362, -0.6919340, 0.7015328
6: 2.6393113, 3.6048961, 2.5830832, 3.5799379, -0.7530956, 0.8099821
7: -9.8603010, -8.5794029, -9.8769798, -8.5262127, -0.8595743, 0.8706293
8: -1.3551314, 0.2789242, -1.3570864, 0.3076835, -1.0508251, 1.0006311
9: -8.2130604, -7.3121901, -8.2016754, -7.3047366, -0.6517212, 0.6300404

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1837
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 1745
type: A, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.47 seconds

### Candidate
type: A, layer: 3, pos: 2810

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2746441, upper bound: 0.2747445
time: 5.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2819922, upper bound: 0.2747458
time: 4.22 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -4.4559488, -3.4099376, -4.4326658, -3.4477336, -0.5962725, 0.5919094
1: -2.4131455, -1.2163618, -2.3885243, -1.2402377, -0.6968949, 0.7019573
2: -3.8567185, -2.7625349, -3.8562560, -2.7956066, -0.8096659, 0.8484403
3: -12.1825962, -10.4772816, -12.1751337, -10.4784813, -0.9587705, 0.9716198
4: -5.6955042, -4.6484213, -5.7063799, -4.6585989, -0.7616973, 0.8082860
5: -2.5381329, -1.3672407, -2.4698002, -1.4028687, -0.6863008, 0.6841687
6: 2.6115429, 3.5802178, 2.6329253, 3.5529823, -0.7468965, 0.7735376
7: -9.8813763, -8.5278711, -9.8557377, -8.5697765, -0.8671145, 0.8619828
8: -1.3591561, 0.2925735, -1.3462718, 0.3019693, -1.0644271, 0.9967000
9: -8.2009964, -7.3115597, -8.1908693, -7.3071165, -0.6627822, 0.6323380

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1837
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 1745
type: A, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.52 seconds

### Candidate
type: A, layer: 3, pos: 2810

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2672724, upper bound: 0.2819926
time: 3.98 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2657131, upper bound: 0.2819937
time: 3.94 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -4.4806499, -3.4098763, -4.4748259, -3.4187865, -0.6367271, 0.5846351
1: -2.4392610, -1.2163618, -2.4333458, -1.2178694, -0.7388999, 0.6906017
2: -3.8567076, -2.7473741, -3.8639207, -2.7718933, -0.8064508, 0.8664564
3: -12.1807480, -10.4773264, -12.1701679, -10.5041132, -0.9889245, 0.9576893
4: -5.6954794, -4.6565652, -5.6799369, -4.6743822, -0.7397404, 0.8381901
5: -2.5381510, -1.3500392, -2.4824028, -1.3742743, -0.6875603, 0.7059350
6: 2.6112666, 3.5850255, 2.6302414, 3.5603659, -0.7561581, 0.7841673
7: -9.8824492, -8.5278234, -9.8536110, -8.5775604, -0.8668594, 0.8624325
8: -1.3594627, 0.2925625, -1.3472016, 0.3021026, -1.0695992, 0.9987352
9: -8.2065458, -7.3112783, -8.2007713, -7.3081837, -0.6727798, 0.6341205

Time for backsubstitution: 22.30 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1837
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 1745
type: A, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.55 seconds

### Candidate
type: A, layer: 3, pos: 2810

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2675894, upper bound: 0.2819936
time: 3.86 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2674192, upper bound: 0.2819916
time: 4.05 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.4589176, -3.4057393, -4.4326658, -3.4477336, -0.5994403, 0.5948640
1: -2.4079432, -1.2198265, -2.3885243, -1.2402377, -0.6888270, 0.6952707
2: -3.8681021, -2.7476194, -3.8562560, -2.7956066, -0.8030074, 0.8645264
3: -12.1902647, -10.4644766, -12.1751337, -10.4784813, -0.9618762, 0.9719658
4: -5.6865807, -4.6528072, -5.7063799, -4.6585989, -0.7599916, 0.8304656
5: -2.5385094, -1.3681812, -2.4698002, -1.4028687, -0.6880040, 0.6816509
6: 2.5926294, 3.6118872, 2.6329253, 3.5529823, -0.7539303, 0.7820082
7: -9.8815794, -8.5279408, -9.8557377, -8.5697765, -0.8682680, 0.8662610
8: -1.3569951, 0.2845962, -1.3462718, 0.3019693, -1.0819590, 0.9917061
9: -8.2063341, -7.3090143, -8.1908693, -7.3071165, -0.6609950, 0.6330974

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1837
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 1745
type: A, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.43 seconds

### Candidate
type: A, layer: 3, pos: 2810

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2657131, upper bound: 0.2819937
time: 3.99 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2664218, upper bound: 0.2819922
time: 4.10 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.47 + 551.29 = 606.76 seconds
