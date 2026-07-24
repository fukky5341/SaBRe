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
execution time: IAR + RelationalAnalysis = 22.28 + 33.89 = 56.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.2876219, upper bound: 0.2876191

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 430
type: B, layer: 1, pos: 430

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 430

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2872976, upper bound: 0.2800684
time: 3.44 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2876152, upper bound: 0.2876110
time: 3.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.25 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.25
Output dim: 6, lower bound: -0.2872976, upper bound: 0.2800684
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.25
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

Time for backsubstitution: 20.95 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 2810
type: B, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 773
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1745
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 2634
type: A, layer: 3, pos: 1780
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 213
type: A, layer: 3, pos: 213
type: B, layer: 3, pos: 1727
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930

Time for candidate selection: 0.45 seconds

### Candidate
type: B, layer: 3, pos: 1459

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2804735, upper bound: 0.2741462
time: 3.54 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2823097, upper bound: 0.2750837
time: 3.75 seconds

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

Time for backsubstitution: 21.42 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 2810
type: B, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 773
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1745
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2634
type: A, layer: 3, pos: 1780
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 1727
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 1727
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 213
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.49 seconds

### Candidate
type: A, layer: 3, pos: 1459

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2816061, upper bound: 0.2807429
time: 3.71 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2826230, upper bound: 0.2826203
time: 3.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.22 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.22
Output dim: 6, lower bound: -0.2804735, upper bound: 0.2741462
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.22
Output dim: 6, lower bound: -0.2823097, upper bound: 0.2750837
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 29.22
Output dim: 6, lower bound: -0.2816061, upper bound: 0.2807429
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 29.22
Output dim: 6, lower bound: -0.2826230, upper bound: 0.2826203

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4.4727578, -3.4191482, -4.4720201, -3.4001856, -0.5965264, 0.5815737
1: -2.4386559, -1.2353368, -2.4831531, -1.2504295, -0.6391590, 0.6802227
2: -3.8301330, -2.7625766, -3.8080330, -2.7509725, -0.7546871, 0.7048589
3: -12.1715736, -10.5038977, -12.1540127, -10.4931507, -1.0021715, 0.9636838
4: -5.6800599, -4.6772132, -5.6925259, -4.6954932, -0.7553480, 0.7905908
5: -2.4822748, -1.3839719, -2.5067937, -1.3855910, -0.6394000, 0.6659647
6: 2.6432323, 3.5718365, 2.6537600, 3.6054716, -0.7132080, 0.6665547
7: -9.8655148, -8.6005669, -9.8965263, -8.6226721, -0.7820344, 0.8312395
8: -1.3516617, 0.3019524, -1.3497703, 0.3205993, -1.0174768, 0.9939950
9: -8.2099733, -7.3165455, -8.2198019, -7.3234291, -0.6276522, 0.6484917

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2810
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 773
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 213
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 2235
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 1780
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 2816
type: A, layer: 3, pos: 2634
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 2810

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2779098, upper bound: 0.2663941
time: 3.69 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2779098, upper bound: 0.2715800
time: 3.61 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.4745941, -3.4188762, -4.4735970, -3.4184265, -0.6259122, 0.6209599
1: -2.4401467, -1.2211790, -2.4419384, -1.2266753, -0.7073526, 0.7397863
2: -3.8618712, -2.7618146, -3.8608825, -2.7605062, -0.8624141, 0.8132229
3: -12.1927309, -10.5038919, -12.1938639, -10.5030508, -0.9874012, 0.9803782
4: -5.6800652, -4.6563292, -5.6813231, -4.6571555, -0.7727225, 0.7933617
5: -2.4823451, -1.3702250, -2.4838202, -1.3630946, -0.6866517, 0.7022762
6: 2.6329911, 3.5719807, 2.6368229, 3.5814734, -0.7721245, 0.7859783
7: -9.8657799, -8.5803671, -9.8766575, -8.5858212, -0.8327191, 0.8711066
8: -1.3560505, 0.3019989, -1.3562200, 0.3042383, -1.0287976, 1.0370440
9: -8.2100048, -7.3083029, -8.2101469, -7.3104472, -0.6641674, 0.6530309

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 773
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 1389
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 2235
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 578

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 1249

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2786080, upper bound: 0.2690996
time: 3.94 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2786976, upper bound: 0.2714735
time: 3.68 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -4.4750137, -3.3876843, -4.4784145, -3.4183035, -0.6002674, 0.6199269
1: -2.4887650, -1.2494311, -2.4423852, -1.2338762, -0.6988587, 0.6636575
2: -3.8123293, -2.7313454, -3.8373661, -2.7601147, -0.7103503, 0.7762085
3: -12.1603565, -10.4650002, -12.1822395, -10.5023880, -0.9703391, 1.0380237
4: -5.7146001, -4.6907210, -5.6823006, -4.6694512, -0.8641579, 0.8195837
5: -2.5621305, -1.3737712, -2.4849498, -1.3615646, -0.7033892, 0.6514323
6: 2.6099887, 3.6157246, 2.6402607, 3.5904770, -0.6974130, 0.7158325
7: -9.9084415, -8.5714111, -9.8868036, -8.5989227, -0.8423641, 0.8172545
8: -1.3573065, 0.3232138, -1.3565226, 0.3062727, -1.0499287, 1.0637791
9: -8.2217426, -7.3207479, -8.2112961, -7.3154888, -0.6928368, 0.6598532

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2810
type: A, layer: 3, pos: 2810
type: B, layer: 3, pos: 773
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 213
type: A, layer: 3, pos: 213
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1263
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 1780
type: A, layer: 3, pos: 1780
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 2634
type: A, layer: 3, pos: 2634
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 1727
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 2810

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2790499, upper bound: 0.2729604
time: 4.77 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2790499, upper bound: 0.2781857
time: 3.93 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -4.4765911, -3.4055338, -4.4802523, -3.4173179, -0.6348054, 0.6493975
1: -2.4477143, -1.2256768, -2.4446144, -1.2197176, -0.7573824, 0.7226369
2: -3.8650723, -2.7405877, -3.8691051, -2.7585106, -0.8200314, 0.8851516
3: -12.2003613, -10.4748869, -12.2037144, -10.5023804, -0.9894795, 1.0228236
4: -5.7033973, -4.6522512, -5.6823058, -4.6483431, -0.8672285, 0.8383596
5: -2.5394039, -1.3513360, -2.4857335, -1.3478320, -0.7411606, 0.6984818
6: 2.5920291, 3.5916567, 2.6281483, 3.5906210, -0.8173604, 0.7760816
7: -9.8888531, -8.5345078, -9.8876362, -8.5787230, -0.8818638, 0.8698053
8: -1.3641591, 0.3072071, -1.3609710, 0.3070273, -1.0887015, 1.0767395
9: -8.2120590, -7.3075390, -8.2113285, -7.3068829, -0.6960926, 0.6972141

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 2810
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1263
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 213
type: A, layer: 3, pos: 578
type: A, layer: 3, pos: 1727
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 213
type: B, layer: 3, pos: 578

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 1249

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2766382, upper bound: 0.2789247
time: 3.60 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2790042, upper bound: 0.2790037
time: 3.96 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.87 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 29.87
Output dim: 6, lower bound: -0.2779098, upper bound: 0.2663941
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 29.87
Output dim: 6, lower bound: -0.2779098, upper bound: 0.2715800
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.87
Output dim: 6, lower bound: -0.2786080, upper bound: 0.2690996
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.87
Output dim: 6, lower bound: -0.2786976, upper bound: 0.2714735
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 29.87
Output dim: 6, lower bound: -0.2790499, upper bound: 0.2729604
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 29.87
Output dim: 6, lower bound: -0.2790499, upper bound: 0.2781857
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.87
Output dim: 6, lower bound: -0.2766382, upper bound: 0.2789247
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.87
Output dim: 6, lower bound: -0.2790042, upper bound: 0.2790037

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -4.4761744, -3.4216135, -4.4717660, -3.4050770, -0.6003871, 0.5787966
1: -2.4374626, -1.2178704, -2.4781125, -1.2504290, -0.6384615, 0.7108420
2: -3.8555377, -2.7626748, -3.7929163, -2.7525449, -0.7832246, 0.6862547
3: -12.1873474, -10.5051088, -12.1407738, -10.4952850, -1.0119078, 0.9503622
4: -5.6758108, -4.6544852, -5.6851325, -4.6955662, -0.7419555, 0.7897120
5: -2.4818418, -1.3691585, -2.5058851, -1.3872960, -0.6348873, 0.6841048
6: 2.6480751, 3.5719342, 2.6823554, 3.6053267, -0.7029645, 0.6252066
7: -9.8657017, -8.5785189, -9.8963509, -8.6244593, -0.7797322, 0.8543832
8: -1.3560843, 0.2934692, -1.3480368, 0.3054709, -0.9995503, 0.9775023
9: -8.2099533, -7.3108411, -8.2197170, -7.3305936, -0.6216812, 0.6519718

Time for backsubstitution: 21.03 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 773
type: A, layer: 3, pos: 773
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 2810
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 1837
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 213
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 2235
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 1780
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 2816
type: A, layer: 3, pos: 2634
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 1727
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 578

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 773

## Relational analysis of NS_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2759177, upper bound: 0.2546851
time: 4.07 seconds

## Relational analysis of NS_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2763632, upper bound: 0.2652557
time: 4.14 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -4.4759431, -3.4196920, -4.4751482, -3.4013963, -0.6014061, 0.5836784
1: -2.4333258, -1.2178704, -2.4712179, -1.2538952, -0.6253097, 0.7086326
2: -3.8618855, -2.7624807, -3.8042996, -2.7395444, -0.8105855, 0.6866318
3: -12.1906776, -10.5047970, -12.1480885, -10.4825096, -1.0105665, 0.9567785
4: -5.6685781, -4.6545005, -5.6762252, -4.6999807, -0.7758384, 0.7899814
5: -2.4818339, -1.3704979, -2.5056705, -1.3895829, -0.6347747, 0.6837575
6: 2.6389842, 3.5719638, 2.6640780, 3.6424253, -0.7581918, 0.6311684
7: -9.8657570, -8.5786400, -9.8971252, -8.6245270, -0.7810869, 0.8519666
8: -1.3562143, 0.2864103, -1.3508465, 0.2967036, -0.9955711, 1.0116591
9: -8.2099838, -7.3103933, -8.2269917, -7.3286343, -0.6246161, 0.6543866

Time for backsubstitution: 20.42 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 773
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 2810
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 1837
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 213
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 2235
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 1780
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 2634
type: A, layer: 3, pos: 2634
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 1727
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 578

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 773

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2674359, upper bound: 0.2700051
time: 3.58 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2763631, upper bound: 0.2704414
time: 3.90 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -4.4326658, -3.4477336, -4.4478979, -3.4185185, -0.5901592, 0.5800285
1: -2.3885243, -1.2402377, -2.4115062, -1.2266752, -0.6603570, 0.6961575
2: -3.8562560, -2.7956066, -3.8608599, -2.7811272, -0.8402967, 0.7792563
3: -12.1751337, -10.4784813, -12.1827888, -10.5030947, -0.9553587, 0.9457531
4: -5.7063799, -4.6585989, -5.6812983, -4.6595306, -0.7393401, 0.7668982
5: -2.4698002, -1.4028687, -2.4837408, -1.3837459, -0.6585258, 0.6778288
6: 2.6329253, 3.5529823, 2.6371455, 3.5701537, -0.7589655, 0.7714386
7: -9.8557377, -8.5697765, -9.8688622, -8.5858831, -0.8215694, 0.8711524
8: -1.3462718, 0.3019693, -1.3500826, 0.3042290, -1.0113945, 1.0271242
9: -8.1908693, -7.3071165, -8.1991444, -7.3114762, -0.6446841, 0.6355155

Time for backsubstitution: 21.09 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 773
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 1389
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 2235
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 578

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1459

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2766632, upper bound: 0.2672445
time: 3.52 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2786080, upper bound: 0.2690996
time: 3.84 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.4748259, -3.4187865, -4.4726000, -3.4184699, -0.5813339, 0.6195586
1: -2.4333458, -1.2178694, -2.4375722, -1.2266752, -0.6488094, 0.7374270
2: -3.8639207, -2.7718933, -3.8608394, -2.7659578, -0.8569982, 0.7776036
3: -12.1701679, -10.5041132, -12.1809406, -10.5031767, -0.9414222, 0.9728541
4: -5.6799369, -4.6743822, -5.6812487, -4.6676712, -0.7691624, 0.7419541
5: -2.4824028, -1.3742743, -2.4837561, -1.3665972, -0.6800580, 0.6752625
6: 2.6302414, 3.5603659, 2.6369109, 3.5749235, -0.7701480, 0.7689574
7: -9.8536110, -8.5775604, -9.8699675, -8.5858374, -0.8222470, 0.8667812
8: -1.3472016, 0.3021026, -1.3502293, 0.3042054, -1.0132866, 1.0322287
9: -8.2007713, -7.3081837, -8.2047367, -7.3112078, -0.6441588, 0.6454823

Time for backsubstitution: 21.03 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 773
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 2235
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1727
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 213
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 578

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 1459

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2768500, upper bound: 0.2697264
time: 4.00 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2786976, upper bound: 0.2714736
time: 3.71 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -4.4748731, -3.3903959, -4.4817157, -3.4229248, -0.5955791, 0.6260073
1: -2.4859190, -1.2494308, -2.4391174, -1.2164088, -0.7311749, 0.6614444
2: -3.8038623, -2.7322738, -3.8561223, -2.7607589, -0.6985354, 0.7978055
3: -12.1530361, -10.4662914, -12.1920881, -10.5043621, -0.9581707, 1.0440226
4: -5.7103291, -4.6907635, -5.6749096, -4.6467829, -0.8736544, 0.7968955
5: -2.5616181, -1.3747134, -2.4841380, -1.3475308, -0.7220776, 0.6470237
6: 2.6275468, 3.6156392, 2.6561074, 3.5905132, -0.6735613, 0.6872332
7: -9.9083290, -8.5723991, -9.8869238, -8.5776777, -0.8654506, 0.8150623
8: -1.3562562, 0.3147140, -1.3602948, 0.2911463, -1.0260475, 1.0538778
9: -8.2216873, -7.3247108, -8.2112465, -7.3129973, -0.6948729, 0.6563432

Time for backsubstitution: 21.20 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 2810
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 1837
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 213
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1263
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 2235
type: A, layer: 3, pos: 1780
type: B, layer: 3, pos: 1780
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 2816
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 2634
type: A, layer: 3, pos: 2634
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1727
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1727
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 578

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 773

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2674595, upper bound: 0.2709213
time: 3.78 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2779097, upper bound: 0.2714171
time: 3.85 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -4.4746442, -3.3884697, -4.4850988, -3.4192462, -0.5964944, 0.6302221
1: -2.4818244, -1.2494311, -2.4322369, -1.2198738, -0.7178106, 0.6591661
2: -3.8101842, -2.7320633, -3.8674936, -2.7477446, -0.7268565, 0.7996619
3: -12.1564722, -10.4659519, -12.1993790, -10.4916067, -0.9571140, 1.0502779
4: -5.7030396, -4.6907797, -5.6659937, -4.6511755, -0.9065461, 0.7973111
5: -2.5616140, -1.3760533, -2.4839253, -1.3498168, -0.7220483, 0.6464696
6: 2.6181631, 3.6156695, 2.6381142, 3.6274500, -0.7335274, 0.6932049
7: -9.9083900, -8.5725193, -9.8876934, -8.5777483, -0.8667181, 0.8128548
8: -1.3563848, 0.3076539, -1.3632345, 0.2823780, -1.0220644, 1.0851393
9: -8.2217178, -7.3243790, -8.2185163, -7.3109069, -0.6981244, 0.6579669

Time for backsubstitution: 21.02 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 2810
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 1837
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 213
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 2902
type: B, layer: 3, pos: 1263
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 2235
type: A, layer: 3, pos: 1780
type: B, layer: 3, pos: 1780
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 2816
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 2634
type: A, layer: 3, pos: 2634
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1727
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 1727
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 578

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 773

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2674595, upper bound: 0.2762042
time: 3.51 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2779097, upper bound: 0.2766419
time: 4.17 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -4.4508905, -3.4056330, -4.4383221, -3.4462399, -0.5938122, 0.6139902
1: -2.4173036, -1.2256769, -2.3931274, -1.2387779, -0.7138035, 0.6756657
2: -3.8650489, -2.7612357, -3.8634894, -2.7920282, -0.7861135, 0.8619796
3: -12.1892872, -10.4749336, -12.1861191, -10.4769630, -0.9548423, 0.9891179
4: -5.7033715, -4.6546240, -5.7086163, -4.6506143, -0.8407667, 0.8049021
5: -2.5393229, -1.3719826, -2.4731181, -1.3804307, -0.7169336, 0.6702818
6: 2.5923700, 3.5803328, 2.6283541, 3.5716209, -0.8025038, 0.7630234
7: -9.8810768, -8.5345688, -9.8776245, -8.5681553, -0.8819718, 0.8588190
8: -1.3579571, 0.3071980, -1.3512015, 0.3069739, -1.0787301, 1.0594361
9: -8.2011099, -7.3085632, -8.1921921, -7.3056779, -0.6790981, 0.6777389

Time for backsubstitution: 20.52 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 2810
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1935
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1263
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2816
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1727
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 578

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 1459

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2746947, upper bound: 0.2768987
time: 3.85 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2766382, upper bound: 0.2789247
time: 3.59 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -4.4755950, -3.4055796, -4.4804821, -3.4172378, -0.6334286, 0.6049135
1: -2.4434483, -1.2256769, -2.4374599, -1.2164090, -0.7550268, 0.6641492
2: -3.8650279, -2.7460403, -3.8711534, -2.7685852, -0.7844484, 0.8799911
3: -12.1874371, -10.4750204, -12.1811562, -10.5026016, -0.9819534, 0.9742618
4: -5.7033195, -4.6627693, -5.6821795, -4.6663909, -0.8157146, 0.8347895
5: -2.5393391, -1.3548105, -2.4857883, -1.3519263, -0.7144532, 0.6919255
6: 2.5921221, 3.5851178, 2.6254187, 3.5790060, -0.8012464, 0.7740750
7: -9.8821640, -8.5345221, -9.8754311, -8.5759220, -0.8775926, 0.8593109
8: -1.3582532, 0.3071747, -1.3519366, 0.3071363, -1.0838885, 1.0612826
9: -8.2066488, -7.3082962, -8.2020903, -7.3067431, -0.6890094, 0.6772120

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 2810
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1249
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 423
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 1242
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1263
type: A, layer: 3, pos: 1263
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2816
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2816
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 1789
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 213
type: A, layer: 3, pos: 1727
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 578
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 1727
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 1977
type: A, layer: 3, pos: 1411
type: A, layer: 3, pos: 213
type: B, layer: 3, pos: 578

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 1459

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2771584, upper bound: 0.2770933
time: 3.74 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2790042, upper bound: 0.2790034
time: 4.13 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.18 seconds
NS_A1_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2759177, upper bound: 0.2546851
NS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2763632, upper bound: 0.2652557
NS_A1_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2674359, upper bound: 0.2700051
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2763631, upper bound: 0.2704414
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2766632, upper bound: 0.2672445
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2786080, upper bound: 0.2690996
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2768500, upper bound: 0.2697264
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2786976, upper bound: 0.2714736
NS_A2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2674595, upper bound: 0.2709213
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2779097, upper bound: 0.2714171
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2674595, upper bound: 0.2762042
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2779097, upper bound: 0.2766419
NS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2746947, upper bound: 0.2768987
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2766382, upper bound: 0.2789247
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2771584, upper bound: 0.2770933
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.18
Output dim: 6, lower bound: -0.2790042, upper bound: 0.2790034

## BFS NS instance: NS_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -4.4739318, -3.4190550, -4.4676518, -3.4053323, -0.5941944, 0.5425025
1: -2.4390457, -1.2200300, -2.4755552, -1.2537526, -0.6297196, 0.6984663
2: -3.8593259, -2.7629697, -3.7854507, -2.7548707, -0.7813900, 0.6863351
3: -12.1945782, -10.5163603, -12.1406097, -10.5189219, -0.9618850, 0.9466295
4: -5.6785631, -4.6545486, -5.6823616, -4.6957626, -0.7426307, 0.7778835
5: -2.4819093, -1.3761942, -2.5050819, -1.4005527, -0.6013447, 0.6779878
6: 2.6329923, 3.5717793, 2.6868205, 3.6049352, -0.7151062, 0.5807313
7: -9.8653479, -8.5783815, -9.8955650, -8.6260395, -0.7614305, 0.8439920
8: -1.3516130, 0.3018427, -1.3413451, 0.3052793, -0.9925551, 0.8418950
9: -8.2071524, -7.3071756, -8.2133169, -7.3311477, -0.6181254, 0.6467648

Time for backsubstitution: 21.06 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2810
type: B, layer: 3, pos: 2810
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 1824
type: A, layer: 3, pos: 773
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 1389
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 1837
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 213
type: B, layer: 3, pos: 213
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1978
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1977
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 2235
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 1780
type: B, layer: 3, pos: 1780
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 2634
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 2634
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 1727
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 1727
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2810

## Relational analysis of NS_A1_B1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2705032, upper bound: 0.2652559
time: 3.81 seconds

## Relational analysis of NS_A1_B1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2705032, upper bound: 0.2652589
time: 4.17 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -4.4721985, -3.4191670, -4.4727702, -3.4015422, -0.5554736, 0.5799298
1: -2.4381044, -1.2219017, -2.4698458, -1.2556753, -0.6245456, 0.6986177
2: -3.8553066, -2.7639232, -3.8002841, -2.7407660, -0.8035519, 0.6808232
3: -12.1944990, -10.5269375, -12.1479797, -10.4959221, -1.0086575, 0.9033167
4: -5.6773157, -4.6546388, -5.6746202, -4.7000923, -0.7669601, 0.7740946
5: -2.4815722, -1.3828158, -2.5052297, -1.3966870, -0.6309347, 0.6354253
6: 2.6350422, 3.5716138, 2.6664104, 3.6422420, -0.7127116, 0.6234862
7: -9.8650265, -8.5791168, -9.8966618, -8.6253757, -0.7697895, 0.8356094
8: -1.3487387, 0.3017659, -1.3470733, 0.2965882, -0.8385661, 1.0104589
9: -8.2047129, -7.3074331, -8.2235203, -7.3289223, -0.6166337, 0.6545234

Time for backsubstitution: 20.98 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2810
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 773
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: B, layer: 3, pos: 213
type: A, layer: 3, pos: 213
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 2467
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1745
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 2235
type: B, layer: 3, pos: 1780
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2634
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 2634
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 710
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 578
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 2930
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 578

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 2810

## Relational analysis of NS_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2763631, upper bound: 0.2645840
time: 3.84 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2763631, upper bound: 0.2704414
time: 3.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.4291115, -3.4481802, -4.4720201, -3.4001856, -0.5552759, 0.5639501
1: -2.3868494, -1.2577052, -2.4831531, -1.2504295, -0.5903153, 0.6473064
2: -3.8224380, -2.7965884, -3.8080330, -2.7509725, -0.7421920, 0.6679841
3: -12.1519518, -10.4784927, -12.1540127, -10.4931507, -0.9699912, 0.9509661
4: -5.7063684, -4.6814342, -5.6925259, -4.6954932, -0.7387037, 0.7620797
5: -2.4695899, -1.4186006, -2.5067937, -1.3855910, -0.6236553, 0.6326675
6: 2.6461575, 3.5528095, 2.6537600, 3.6054716, -0.7097430, 0.6426356
7: -9.8552818, -8.5928183, -9.8965263, -8.6226721, -0.7747588, 0.8397956
8: -1.3408222, 0.3017759, -1.3497703, 0.3205993, -0.9990375, 0.9940171
9: -8.1908379, -7.3169403, -8.2198019, -7.3234291, -0.6109352, 0.6429292

Time for backsubstitution: 21.06 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2810
type: A, layer: 3, pos: 2810
type: B, layer: 3, pos: 773
type: A, layer: 3, pos: 773
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 1824
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 1935
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 213
type: B, layer: 3, pos: 213
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 2467
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 2902
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 2235
type: A, layer: 3, pos: 2235
type: B, layer: 3, pos: 1780
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 2816
type: A, layer: 3, pos: 2634
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2634
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 710
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 578

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 2810

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2751801, upper bound: 0.2612850
time: 4.02 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2751801, upper bound: 0.2604890
time: 3.61 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.4309483, -3.4478929, -4.4735970, -3.4184265, -0.5894101, 0.6014091
1: -2.3882992, -1.2435464, -2.4419384, -1.2266753, -0.6602249, 0.7230719
2: -3.8541355, -2.7958663, -3.8608825, -2.7605062, -0.8589408, 0.7791986
3: -12.1731071, -10.4784842, -12.1938639, -10.5030508, -0.9532030, 0.9693332
4: -5.7063742, -4.6605577, -5.6813231, -4.6571555, -0.7565191, 0.7648594
5: -2.4696484, -1.4048769, -2.4838202, -1.3630946, -0.6767688, 0.6763401
6: 2.6358244, 3.5529470, 2.6368229, 3.5814734, -0.7646112, 0.7708135
7: -9.8555431, -8.5726118, -9.8766575, -8.5858212, -0.8215191, 0.8774164
8: -1.3451877, 0.3018184, -1.3562200, 0.3042383, -1.0106449, 1.0372438
9: -8.1908674, -7.3085642, -8.2101469, -7.3104472, -0.6473117, 0.6450858

Time for backsubstitution: 21.45 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1249
type: B, layer: 3, pos: 1249
type: A, layer: 3, pos: 1459
type: B, layer: 3, pos: 2810
type: B, layer: 3, pos: 773
type: A, layer: 3, pos: 2810
type: A, layer: 3, pos: 773
type: B, layer: 3, pos: 1824
type: A, layer: 3, pos: 1824
type: B, layer: 3, pos: 2342
type: A, layer: 3, pos: 2342
type: B, layer: 3, pos: 1389
type: A, layer: 3, pos: 1389
type: B, layer: 3, pos: 2588
type: A, layer: 3, pos: 2588
type: B, layer: 3, pos: 1837
type: A, layer: 3, pos: 1837
type: B, layer: 3, pos: 2467
type: A, layer: 3, pos: 2467
type: B, layer: 3, pos: 423
type: A, layer: 3, pos: 423
type: A, layer: 3, pos: 1242
type: B, layer: 3, pos: 1242
type: B, layer: 3, pos: 1978
type: A, layer: 3, pos: 1978
type: B, layer: 3, pos: 1935
type: A, layer: 3, pos: 1745
type: B, layer: 3, pos: 2902
type: A, layer: 3, pos: 2902
type: B, layer: 3, pos: 1745
type: A, layer: 3, pos: 1263
type: B, layer: 3, pos: 1263
type: B, layer: 3, pos: 2634
type: B, layer: 3, pos: 1780
type: B, layer: 3, pos: 2235
type: A, layer: 3, pos: 2235
type: A, layer: 3, pos: 2634
type: A, layer: 3, pos: 1780
type: A, layer: 3, pos: 2816
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: A, layer: 3, pos: 1977
type: B, layer: 3, pos: 1727
type: B, layer: 3, pos: 2816
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 1935
type: B, layer: 3, pos: 578
type: A, layer: 3, pos: 213
type: A, layer: 3, pos: 710
type: B, layer: 3, pos: 710
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 1977
type: A, layer: 3, pos: 1411
type: B, layer: 3, pos: 2930
type: B, layer: 3, pos: 213
type: B, layer: 3, pos: 1411
type: A, layer: 3, pos: 2930
type: A, layer: 3, pos: 1727
type: A, layer: 3, pos: 578

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1249

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2786080, upper bound: 0.2690996
time: 4.17 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2786080, upper bound: 0.2690965
time: 4.05 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.4712706, -3.4192276, -4.4720201, -3.4001856, -0.5483990, 0.5813920
1: -2.4315414, -1.2353370, -2.4831531, -1.2504295, -0.5652139, 0.6802227
2: -3.8300638, -2.7728770, -3.8080330, -2.7509725, -0.7542696, 0.6594532
3: -12.1469851, -10.5041275, -12.1540127, -10.4931507, -0.9545858, 0.9613860
4: -5.6799283, -4.6972189, -5.6925259, -4.6954932, -0.7536194, 0.7371304
5: -2.4821610, -1.3900537, -2.5067937, -1.3855910, -0.6386786, 0.6242297
6: 2.6433990, 3.5601909, 2.6537600, 3.6054716, -0.7131038, 0.6420376
7: -9.8531742, -8.6005945, -9.8965263, -8.6226721, -0.7752113, 0.8312056
8: -1.3416488, 0.3018990, -1.3497703, 0.3205993, -1.0001335, 0.9934688
9: -8.2007351, -7.3180065, -8.2198019, -7.3234291, -0.6118913, 0.6439646

Time for backsubstitution: 22.14 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.17 + 555.42 = 611.59 seconds
