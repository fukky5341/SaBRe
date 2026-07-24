## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.0506091392
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9354134, -5.3531485, -8.9354134, -5.3531485, -3.5822649, 3.5822649)
1: (-7.3978786, -4.1556597, -7.3978786, -4.1556597, -3.1666536, 3.1666532)
2: (-7.4789820, -4.5742297, -7.4789820, -4.5742297, -2.9047523, 2.9047523)
3: (-11.2633400, -7.7441711, -11.2633400, -7.7441711, -3.4115591, 3.4115601)
4: (6.5621042, 8.8026104, 6.5621042, 8.8026104, -2.1841531, 2.1841531)
5: (-8.9045181, -5.9158378, -8.9045181, -5.9158378, -2.9886804, 2.9886804)
6: (-12.0150757, -8.2602482, -12.0150757, -8.2602482, -3.7548275, 3.7548275)
7: (-3.2182775, -0.5745678, -3.2182775, -0.5745678, -2.6437097, 2.6437097)
8: (-6.9675961, -3.5078919, -6.9675961, -3.5078919, -3.2341232, 3.2341237)
9: (-5.5373082, -3.0319777, -5.5373082, -3.0319777, -2.4804778, 2.4804778)

## BASE Result
execution time: IAR + LP analysis = 13.31 + 34.04 = 47.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.7321513, upper bound: 1.7321501


# Binary Search by BASE starts (time budget: 3552.66 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.8360369205474854
rel_dist={4: [-1.3410396013151953, 1.3410415304274936]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.6619789600372314
rel_dist={4: [-1.051660938425652, 1.051661396029865]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.545940637588501
rel_dist={4: [-0.792442823129953, 0.7924427706559456]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.603959560394287
rel_dist={4: [-0.9387819866503717, 0.9387850582514945]}

## Binary Search Result
Binary search time: 196.12 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3356.54 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184892, upper bound: 1.4077454
time: 6.28 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184892, upper bound: 1.4184880
time: 5.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.67 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 11.67
Output dim: 4, lower bound: -1.4184892, upper bound: 1.4077454
IS_B2, status: Status.UNKNOWN, split count: 1, time: 11.67
Output dim: 4, lower bound: -1.4184892, upper bound: 1.4184880

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -8.9354134, -5.3531485, -8.9197674, -5.3591337, -3.3405361, 3.3299439
1: -7.3978786, -4.1556597, -7.3834295, -4.1582737, -2.7439027, 2.7315743
2: -7.4789820, -4.5742297, -7.4752836, -4.5928955, -2.5727406, 2.5836208
3: -11.2633400, -7.7441711, -11.2590923, -7.7627215, -2.9700165, 2.9827383
4: 6.5621042, 8.8026104, 6.5971775, 8.8024073, -1.8905063, 1.8558021
5: -8.9045181, -5.9158378, -8.9024849, -5.9274836, -2.5986295, 2.6014934
6: -12.0150757, -8.2602482, -11.9991140, -8.2676954, -3.5510759, 3.5400662
7: -3.2182775, -0.5745678, -3.1996956, -0.5760213, -2.6422563, 2.6251278
8: -6.9675961, -3.5078919, -6.9664278, -3.5248680, -2.7491474, 2.7652020
9: -5.5373082, -3.0319777, -5.5144129, -3.0330338, -2.2139018, 2.1919844

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4077428
time: 5.42 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4077412
time: 4.51 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -8.9353971, -5.3531528, -8.9553642, -5.3070812, -3.4124756, 3.3636155
1: -7.3978720, -4.1556597, -7.4131517, -4.0979872, -2.7963676, 2.7588449
2: -7.4789791, -4.5742402, -7.5399389, -4.5638552, -2.6088834, 2.6313014
3: -11.2633362, -7.7441897, -11.3187714, -7.7308869, -3.0030141, 3.0318432
4: 6.5621233, 8.8026104, 6.5160456, 8.8161116, -1.9048624, 1.9368992
5: -8.9045162, -5.9158492, -8.9200726, -5.8995981, -2.6395264, 2.6155272
6: -12.0150566, -8.2602539, -12.0180368, -8.1864262, -3.6226573, 3.5629940
7: -3.2182701, -0.5745701, -3.2622232, -0.5131162, -2.7051539, 2.6876531
8: -6.9675951, -3.5079165, -7.0121460, -3.4867215, -2.7942519, 2.8103395
9: -5.5372796, -3.0319781, -5.5743561, -3.0260987, -2.2204757, 2.2545891

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4184890
time: 5.17 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4184892
time: 5.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.66 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 24.66
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4077428
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 24.66
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4077412
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 24.66
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4184890
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 24.66
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4184892

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -8.9197674, -5.3591337, -8.9197674, -5.3591337, -3.3190107, 3.3190103
1: -7.3834295, -4.1582737, -7.3834295, -4.1582737, -2.7294049, 2.7294056
2: -7.4752836, -4.5928955, -7.4752836, -4.5928955, -2.5651479, 2.5651476
3: -11.2590923, -7.7627215, -11.2590923, -7.7627215, -2.9635572, 2.9635570
4: 6.5971775, 8.8024073, 6.5971775, 8.8024073, -1.8522520, 1.8522518
5: -8.9024849, -5.9274836, -8.9024849, -5.9274836, -2.5891528, 2.5891523
6: -11.9991140, -8.2676954, -11.9991140, -8.2676954, -3.5332289, 3.5332289
7: -3.1996956, -0.5760213, -3.1996956, -0.5760213, -2.6236744, 2.6236744
8: -6.9664278, -3.5248680, -6.9664278, -3.5248680, -2.7458000, 2.7457998
9: -5.5144129, -3.0330338, -5.5144129, -3.0330338, -2.1883738, 2.1883738

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077290, upper bound: 1.4063845
time: 5.83 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077288, upper bound: 1.4077347
time: 5.32 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -8.9553642, -5.3070812, -8.9197674, -5.3591337, -3.3526945, 3.3704667
1: -7.4131517, -4.0979872, -7.3834295, -4.1582737, -2.7563615, 2.7818680
2: -7.5399389, -4.5638552, -7.4752836, -4.5928955, -2.6128230, 2.5924342
3: -11.3187714, -7.7308869, -11.2590923, -7.7627215, -3.0126047, 2.9935024
4: 6.5160456, 8.8161116, 6.5971775, 8.8024073, -1.9304633, 1.8666394
5: -8.9200726, -5.8995981, -8.9024849, -5.9274836, -2.6031961, 2.6175699
6: -12.0180368, -8.1864262, -11.9991140, -8.2676954, -3.5549583, 3.6048446
7: -3.2622232, -0.5131162, -3.1996956, -0.5760213, -2.6862020, 2.6865792
8: -7.0121460, -3.4867215, -6.9664278, -3.5248680, -2.7909565, 2.7833567
9: -5.5743561, -3.0260987, -5.5144129, -3.0330338, -2.2481244, 2.1949694

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077290, upper bound: 1.4063829
time: 4.61 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077288, upper bound: 1.4077331
time: 4.70 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -8.9197674, -5.3591337, -8.9553642, -5.3070812, -3.3704672, 3.3526945
1: -7.3834295, -4.1582737, -7.4131517, -4.0979872, -2.7818680, 2.7563610
2: -7.4752836, -4.5928955, -7.5399389, -4.5638552, -2.5924344, 2.6128230
3: -11.2590923, -7.7627215, -11.3187714, -7.7308869, -2.9935026, 3.0126050
4: 6.5971775, 8.8024073, 6.5160456, 8.8161116, -1.8666396, 1.9304631
5: -8.9024849, -5.9274836, -8.9200726, -5.8995981, -2.6175699, 2.6031961
6: -11.9991140, -8.2676954, -12.0180368, -8.1864262, -3.6048446, 3.5549583
7: -3.1996956, -0.5760213, -3.2622232, -0.5131162, -2.6865792, 2.6862020
8: -6.9664278, -3.5248680, -7.0121460, -3.4867215, -2.7833567, 2.7909565
9: -5.5144129, -3.0330338, -5.5743561, -3.0260987, -2.1949692, 2.2481241

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063783, upper bound: 1.4184765
time: 5.30 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077286, upper bound: 1.4184746
time: 7.29 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -8.9553642, -5.3070812, -8.9553642, -5.3070812, -3.4254913, 3.4254913
1: -7.4131517, -4.0979872, -7.4131517, -4.0979872, -2.8087630, 2.8087630
2: -7.5399389, -4.5638552, -7.5399389, -4.5638552, -2.6229138, 2.6229138
3: -11.3187714, -7.7308869, -11.3187714, -7.7308869, -3.0426373, 3.0426373
4: 6.5160456, 8.8161116, 6.5160456, 8.8161116, -1.9378514, 1.9378512
5: -8.9200726, -5.8995981, -8.9200726, -5.8995981, -2.6448889, 2.6448891
6: -12.0180368, -8.1864262, -12.0180368, -8.1864262, -3.6266446, 3.6266446
7: -3.2622232, -0.5131162, -3.2622232, -0.5131162, -2.7491069, 2.7491069
8: -7.0121460, -3.4867215, -7.0121460, -3.4867215, -2.7961974, 2.7961974
9: -5.5743561, -3.0260987, -5.5743561, -3.0260987, -2.2557113, 2.2557111

Time for backsubstitution: 14.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077290, upper bound: 1.4105793
time: 4.52 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077288, upper bound: 1.4120988
time: 4.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.59 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 23.59
Output dim: 4, lower bound: -1.4077290, upper bound: 1.4063845
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 23.59
Output dim: 4, lower bound: -1.4077288, upper bound: 1.4077347
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 23.59
Output dim: 4, lower bound: -1.4077290, upper bound: 1.4063829
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 23.59
Output dim: 4, lower bound: -1.4077288, upper bound: 1.4077331
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 23.59
Output dim: 4, lower bound: -1.4063783, upper bound: 1.4184765
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 23.59
Output dim: 4, lower bound: -1.4077286, upper bound: 1.4184746
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 23.59
Output dim: 4, lower bound: -1.4077290, upper bound: 1.4105793
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 23.59
Output dim: 4, lower bound: -1.4077288, upper bound: 1.4120988

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9197674, -5.3591337, -3.3156261, 3.3156724
1: -7.3795280, -4.1591473, -7.3834295, -4.1582737, -2.7249875, 2.7271726
2: -7.4732456, -4.5950751, -7.4752836, -4.5928955, -2.5633616, 2.5625660
3: -11.2570133, -7.7643375, -11.2590923, -7.7627215, -2.9609108, 2.9616303
4: 6.5990024, 8.8007011, 6.5971775, 8.8024073, -1.8504705, 1.8504057
5: -8.8986826, -5.9286919, -8.9024849, -5.9274836, -2.5849872, 2.5876861
6: -11.9953022, -8.2697430, -11.9991140, -8.2676954, -3.5293970, 3.5316119
7: -3.1971812, -0.5803651, -3.1996956, -0.5760213, -2.6211600, 2.6193304
8: -6.9589167, -3.5266910, -6.9664278, -3.5248680, -2.7382097, 2.7435162
9: -5.5121059, -3.0363898, -5.5144129, -3.0330338, -2.1862228, 2.1848228

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4063824
time: 6.49 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4063828
time: 6.02 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.9362144, -5.3529425, -8.9197655, -5.3591366, -3.3357296, 3.3266954
1: -7.4028411, -4.1309299, -7.3834219, -4.1582775, -2.7589455, 2.7531879
2: -7.4982338, -4.5862241, -7.4752812, -4.5928984, -2.5909524, 2.5725298
3: -11.2656307, -7.7241049, -11.2590885, -7.7627258, -2.9731631, 3.0024526
4: 6.5204015, 8.8054256, 6.5971794, 8.8024044, -1.9247701, 1.8582754
5: -8.9051714, -5.8647685, -8.9024773, -5.9274855, -2.5945940, 2.6462259
6: -12.0111465, -8.2177105, -11.9991055, -8.2676983, -3.5495758, 3.5826774
7: -3.2468393, -0.5716875, -3.1996911, -0.5760288, -2.6708105, 2.6280036
8: -6.9749389, -3.4555233, -6.9664173, -3.5248699, -2.7665954, 2.8082247
9: -5.5897975, -3.0277028, -5.5144110, -3.0330400, -2.2539482, 2.1976929

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4077350
time: 5.16 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4077329
time: 6.73 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.9533844, -5.3091950, -8.9197674, -5.3591337, -3.3493729, 3.3671350
1: -7.4092550, -4.0988493, -7.3834295, -4.1582737, -2.7519612, 2.7796276
2: -7.5379148, -4.5660362, -7.4752836, -4.5928955, -2.6109424, 2.5898662
3: -11.3167067, -7.7324371, -11.2590923, -7.7627215, -3.0099430, 2.9916179
4: 6.5177813, 8.8144073, 6.5971775, 8.8024073, -1.9287515, 1.8647928
5: -8.9162674, -5.9007535, -8.9024849, -5.9274836, -2.5990329, 2.6161652
6: -12.0142260, -8.1884604, -11.9991140, -8.2676954, -3.5511312, 3.6032877
7: -3.2597790, -0.5174644, -3.1996956, -0.5760213, -2.6837578, 2.6822312
8: -7.0046282, -3.4885011, -6.9664278, -3.5248680, -2.7833652, 2.7810929
9: -5.5721159, -3.0294533, -5.5144129, -3.0330338, -2.2458951, 2.1914203

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4063827
time: 6.41 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4063827
time: 6.60 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -8.9719667, -5.3008866, -8.9197655, -5.3591366, -3.3696346, 3.3781769
1: -7.4332204, -4.0706139, -7.3834219, -4.1582775, -2.7862158, 2.7880986
2: -7.5612316, -4.5572190, -7.4752812, -4.5928984, -2.6334553, 2.5997767
3: -11.3253937, -7.6917968, -11.2590885, -7.7627258, -3.0210028, 3.0326190
4: 6.4383445, 8.8191280, 6.5971794, 8.8024044, -2.0063202, 1.8726630
5: -8.9227638, -5.8366723, -8.9024773, -5.9274855, -2.6086597, 2.6752050
6: -12.0300827, -8.1381168, -11.9991055, -8.2676983, -3.5713263, 3.6379609
7: -3.3094108, -0.5087389, -3.1996911, -0.5760288, -2.7333820, 2.6909523
8: -7.0206347, -3.4173870, -6.9664173, -3.5248699, -2.8108382, 2.8468390
9: -5.6503248, -3.0207715, -5.5144110, -3.0330400, -2.3141413, 2.2042856

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4077349
time: 4.55 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4077350
time: 5.07 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9197674, -5.3591337, -8.9533844, -5.3091950, -3.3671355, 3.3493721
1: -7.3834295, -4.1582737, -7.4092550, -4.0988493, -2.7796276, 2.7519610
2: -7.4752836, -4.5928955, -7.5379148, -4.5660362, -2.5898662, 2.6109421
3: -11.2590923, -7.7627215, -11.3167067, -7.7324371, -2.9916177, 3.0099428
4: 6.5971775, 8.8024073, 6.5177813, 8.8144073, -1.8647931, 1.9287512
5: -8.9024849, -5.9274836, -8.9162674, -5.9007535, -2.6161647, 2.5990329
6: -11.9991140, -8.2676954, -12.0142260, -8.1884604, -3.6032877, 3.5511312
7: -3.1996956, -0.5760213, -3.2597790, -0.5174644, -2.6822312, 2.6837578
8: -6.9664278, -3.5248680, -7.0046282, -3.4885011, -2.7810926, 2.7833648
9: -5.5144129, -3.0330338, -5.5721159, -3.0294533, -2.1914206, 2.2458954

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4171020
time: 5.23 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4184749
time: 6.28 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9197655, -5.3591366, -8.9719667, -5.3008866, -3.3781767, 3.3696351
1: -7.3834219, -4.1582775, -7.4332204, -4.0706139, -2.7880986, 2.7862153
2: -7.4752812, -4.5928984, -7.5612316, -4.5572190, -2.5997767, 2.6334553
3: -11.2590885, -7.7627258, -11.3253937, -7.6917968, -3.0326190, 3.0210030
4: 6.5971794, 8.8024044, 6.4383445, 8.8191280, -1.8726635, 2.0063200
5: -8.9024773, -5.9274855, -8.9227638, -5.8366723, -2.6752050, 2.6086600
6: -11.9991055, -8.2676983, -12.0300827, -8.1381168, -3.6379614, 3.5713267
7: -3.1996911, -0.5760288, -3.3094108, -0.5087389, -2.6909523, 2.7333820
8: -6.9664173, -3.5248699, -7.0206347, -3.4173870, -2.8468390, 2.8108382
9: -5.5144110, -3.0330400, -5.6503248, -3.0207715, -2.2042851, 2.3141415

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077331, upper bound: 1.4170999
time: 7.39 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077331, upper bound: 1.4184743
time: 11.97 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -8.9533844, -5.3091950, -8.9553642, -5.3070812, -3.4221134, 3.4221582
1: -7.4092550, -4.0988493, -7.4131517, -4.0979872, -2.8043513, 2.8065224
2: -7.5379148, -4.5660362, -7.5399389, -4.5638552, -2.6211901, 2.6203461
3: -11.3167067, -7.7324371, -11.3187714, -7.7308869, -3.0399747, 3.0407112
4: 6.5177813, 8.8144073, 6.5160456, 8.8161116, -1.9361391, 1.9360046
5: -8.9162674, -5.9007535, -8.9200726, -5.8995981, -2.6407270, 2.6434195
6: -12.0142260, -8.1884604, -12.0180368, -8.1864262, -3.6227612, 3.6250873
7: -3.2597790, -0.5174644, -3.2622232, -0.5131162, -2.7466626, 2.7447588
8: -7.0046282, -3.4885011, -7.0121460, -3.4867215, -2.7886071, 2.7939341
9: -5.5721159, -3.0294533, -5.5743561, -3.0260987, -2.2534826, 2.2521605

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4105798
time: 7.71 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4105801
time: 7.15 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -8.9719667, -5.3008866, -8.9553623, -5.3070850, -3.4423184, 3.4331837
1: -7.4332204, -4.0706139, -7.4131451, -4.0979881, -2.8380196, 2.8149939
2: -7.5612316, -4.5572190, -7.5399365, -4.5638580, -2.6492758, 2.6302564
3: -11.3253937, -7.6917968, -11.3187675, -7.7308884, -3.0520296, 3.0703623
4: 6.4383445, 8.8191280, 6.5160475, 8.8161087, -2.0107813, 1.9438775
5: -8.9227638, -5.8366723, -8.9200649, -5.8995991, -2.6503258, 2.6907620
6: -12.0300827, -8.1381168, -12.0180311, -8.1864281, -3.6430678, 3.6597600
7: -3.3094108, -0.5087389, -3.2622199, -0.5131224, -2.7962885, 2.7534809
8: -7.0206347, -3.4173870, -7.0121317, -3.4867249, -2.8170185, 2.8522942
9: -5.6503248, -3.0207715, -5.5743518, -3.0261054, -2.3189836, 2.2650387

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4121007
time: 4.79 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4121008
time: 5.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.87 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4063824
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4063828
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4077350
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4077329
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4063827
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4063827
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4077349
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4077350
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4171020
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4184749
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4077331, upper bound: 1.4170999
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4077331, upper bound: 1.4184743
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4105798
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4105801
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4121007
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.87
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4121008

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9177856, -5.3612566, -3.3122883, 3.3122878
1: -7.3795280, -4.1591473, -7.3795280, -4.1591473, -2.7227550, 2.7227552
2: -7.4732456, -4.5950751, -7.4732456, -4.5950751, -2.5607800, 2.5607800
3: -11.2570133, -7.7643375, -11.2570133, -7.7643375, -2.9589849, 2.9589841
4: 6.5990024, 8.8007011, 6.5990024, 8.8007011, -1.8486245, 1.8486245
5: -8.8986826, -5.9286919, -8.8986826, -5.9286919, -2.5835214, 2.5835209
6: -11.9953022, -8.2697430, -11.9953022, -8.2697430, -3.5277796, 3.5277801
7: -3.1971812, -0.5803651, -3.1971812, -0.5803651, -2.6168160, 2.6168160
8: -6.9589167, -3.5266910, -6.9589167, -3.5266910, -2.7359266, 2.7359264
9: -5.5121059, -3.0363898, -5.5121059, -3.0363898, -2.1826720, 2.1826723

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3996528, upper bound: 1.4062786
time: 6.66 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062789, upper bound: 1.4062790
time: 4.01 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9361820, -5.3530612, -3.3226566, 3.3323393
1: -7.3795280, -4.1591473, -7.4027824, -4.1315842, -2.7482657, 2.7511160
2: -7.4732456, -4.5950751, -7.4976530, -4.5862246, -2.5707030, 2.5878129
3: -11.2570133, -7.7643375, -11.2656317, -7.7246079, -2.9992990, 2.9702418
4: 6.5990024, 8.8007011, 6.5214138, 8.8054247, -1.8541167, 1.9221119
5: -8.8986826, -5.9286919, -8.9051685, -5.8653121, -2.6416907, 2.5902672
6: -11.9953022, -8.2697430, -12.0111408, -8.2183380, -3.5782194, 3.5449009
7: -3.1971812, -0.5803651, -3.2467051, -0.5717285, -2.6254528, 2.6663399
8: -6.9589167, -3.5266910, -6.9748068, -3.4556038, -2.8005524, 2.7529857
9: -5.5121059, -3.0363898, -5.5889773, -3.0277052, -2.1921167, 2.2497573

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3996528, upper bound: 1.4062785
time: 5.44 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062789, upper bound: 1.4062784
time: 5.98 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9361820, -5.3530612, -8.9177856, -5.3612566, -3.3323393, 3.3226562
1: -7.4027824, -4.1315842, -7.3795280, -4.1591473, -2.7511163, 2.7482655
2: -7.4976530, -4.5862246, -7.4732456, -4.5950751, -2.5878129, 2.5707028
3: -11.2656317, -7.7246079, -11.2570133, -7.7643375, -2.9702420, 2.9992986
4: 6.5214138, 8.8054247, 6.5990024, 8.8007011, -1.9221120, 1.8541167
5: -8.9051685, -5.8653121, -8.8986826, -5.9286919, -2.5902667, 2.6416910
6: -12.0111408, -8.2183380, -11.9953022, -8.2697430, -3.5448999, 3.5782199
7: -3.2467051, -0.5717285, -3.1971812, -0.5803651, -2.6663399, 2.6254528
8: -6.9748068, -3.4556038, -6.9589167, -3.5266910, -2.7529860, 2.8005528
9: -5.5889773, -3.0277052, -5.5121059, -3.0363898, -2.2497573, 2.1921175

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4010365
time: 4.28 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4076078
time: 6.45 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9362364, -5.3528647, -8.9362364, -5.3528647, -3.3438206, 3.3438203
1: -7.4028764, -4.1304970, -7.4028764, -4.1304970, -2.7659073, 2.7659078
2: -7.4986153, -4.5862231, -7.4986153, -4.5862231, -2.5984902, 2.5984900
3: -11.2656326, -7.7237730, -11.2656326, -7.7237730, -3.0077243, 3.0077243
4: 6.5197315, 8.8054256, 6.5197315, 8.8054256, -1.9289927, 1.9289927
5: -8.9051723, -5.8644085, -8.9051723, -5.8644085, -2.6445298, 2.6445298
6: -12.0111532, -8.2172947, -12.0111532, -8.2172947, -3.5888720, 3.5888720
7: -3.2469287, -0.5716588, -3.2469287, -0.5716588, -2.6752698, 2.6752698
8: -6.9750233, -3.4554715, -6.9750233, -3.4554715, -2.7822704, 2.7822702
9: -5.5903368, -3.0276990, -5.5903368, -3.0276990, -2.2602723, 2.2602725

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4010347
time: 6.70 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4076077
time: 5.72 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9533844, -5.3091950, -8.9177856, -5.3612566, -3.3460340, 3.3637509
1: -7.4092550, -4.0988493, -7.3795280, -4.1591473, -2.7497277, 2.7751966
2: -7.5379148, -4.5660362, -7.4732456, -4.5950751, -2.6083441, 2.5880802
3: -11.3167067, -7.7324371, -11.2570133, -7.7643375, -3.0080261, 2.9889717
4: 6.5177813, 8.8144073, 6.5990024, 8.8007011, -1.9269044, 1.8630116
5: -8.9162674, -5.9007535, -8.8986826, -5.9286919, -2.5975671, 2.6120000
6: -12.0142260, -8.1884604, -11.9953022, -8.2697430, -3.5495138, 3.5994048
7: -3.2597790, -0.5174644, -3.1971812, -0.5803651, -2.6794138, 2.6797168
8: -7.0046282, -3.4885011, -6.9589167, -3.5266910, -2.7810802, 2.7735031
9: -5.5721159, -3.0294533, -5.5121059, -3.0363898, -2.2423444, 2.1892698

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4103544, upper bound: 1.4062786
time: 5.22 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4062783
time: 5.25 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9533844, -5.3091950, -8.9361820, -5.3530612, -3.3564024, 3.3838019
1: -7.4092550, -4.0988493, -7.4027824, -4.1315842, -2.7752385, 2.8036025
2: -7.5379148, -4.5660362, -7.4976530, -4.5862246, -2.6182599, 2.6151133
3: -11.3167067, -7.7324371, -11.2656317, -7.7246079, -3.0367699, 3.0002296
4: 6.5177813, 8.8144073, 6.5214138, 8.8054247, -1.9323974, 1.9252262
5: -8.9162674, -5.9007535, -8.9051685, -5.8653121, -2.6502514, 2.6187468
6: -12.0142260, -8.1884604, -12.0111408, -8.2183380, -3.6000185, 3.6166334
7: -3.2597790, -0.5174644, -3.2467051, -0.5717285, -2.6880505, 2.7136269
8: -7.0046282, -3.4885011, -6.9748068, -3.4556038, -2.8047290, 2.7905624
9: -5.5721159, -3.0294533, -5.5889773, -3.0277052, -2.2517896, 2.2520049

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4103544, upper bound: 1.4062804
time: 4.00 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4062785
time: 5.92 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9719296, -5.3010058, -8.9177856, -5.3612566, -3.3662434, 3.3741379
1: -7.4331570, -4.0712700, -7.3795280, -4.1591473, -2.7786736, 2.7834020
2: -7.5606685, -4.5572195, -7.4732456, -4.5950751, -2.6305003, 2.5979488
3: -11.3253937, -7.6923037, -11.2570133, -7.7643375, -3.0190887, 3.0294616
4: 6.4393682, 8.8191280, 6.5990024, 8.8007011, -2.0036356, 1.8684998
5: -8.9227610, -5.8372159, -8.8986826, -5.9286919, -2.6043334, 2.6706641
6: -12.0300741, -8.1387186, -11.9953022, -8.2697430, -3.5666513, 3.6335001
7: -3.3092690, -0.5087802, -3.1971812, -0.5803651, -2.7289038, 2.6884010
8: -7.0205030, -3.4174809, -6.9589167, -3.5266910, -2.7981205, 2.8391554
9: -5.6494908, -3.0207734, -5.5121059, -3.0363898, -2.3099394, 2.1987097

Time for backsubstitution: 14.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4010365
time: 3.80 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4076073
time: 6.87 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9719906, -5.3008099, -8.9362364, -5.3528647, -3.3777275, 3.3953025
1: -7.4332638, -4.0701799, -7.4028764, -4.1304970, -2.7931805, 2.8128924
2: -7.5616016, -4.5572181, -7.4986153, -4.5862231, -2.6410115, 2.6257367
3: -11.3253918, -7.6914606, -11.2656326, -7.7237730, -3.0484958, 3.0378928
4: 6.4376669, 8.8191299, 6.5197315, 8.8054256, -2.0083752, 1.9321352
5: -8.9227667, -5.8363128, -8.9051723, -5.8644085, -2.6576433, 2.6729794
6: -12.0300884, -8.1377153, -12.0111532, -8.2172947, -3.6106215, 3.6517031
7: -3.3095057, -0.5087115, -3.2469287, -0.5716588, -2.7378469, 2.7233593
8: -7.0207233, -3.4173274, -6.9750233, -3.4554715, -2.8221450, 2.8207881
9: -5.6508780, -3.0207691, -5.5903368, -3.0276990, -2.3195779, 2.2625377

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B1_A2_A2_B2_B1

### Relational analysis result of IS_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4103540, upper bound: 1.4076077
time: 6.14 seconds

## Relational analysis of IS_B1_A2_A2_B2_B2

### Relational analysis result of IS_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170009, upper bound: 1.4076098
time: 4.94 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9533844, -5.3091950, -3.3637519, 3.3460338
1: -7.3795280, -4.1591473, -7.4092550, -4.0988493, -2.7751968, 2.7497275
2: -7.4732456, -4.5950751, -7.5379148, -4.5660362, -2.5880799, 2.6083441
3: -11.2570133, -7.7643375, -11.3167067, -7.7324371, -2.9889722, 3.0080259
4: 6.5990024, 8.8007011, 6.5177813, 8.8144073, -1.8630116, 1.9269052
5: -8.8986826, -5.9286919, -8.9162674, -5.9007535, -2.6120000, 2.5975666
6: -11.9953022, -8.2697430, -12.0142260, -8.1884604, -3.5994053, 3.5495138
7: -3.1971812, -0.5803651, -3.2597790, -0.5174644, -2.6797168, 2.6794138
8: -6.9589167, -3.5266910, -7.0046282, -3.4885011, -2.7735033, 2.7810807
9: -5.5121059, -3.0363898, -5.5721159, -3.0294533, -2.1892695, 2.2423444

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4103561
time: 5.92 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4170015
time: 5.55 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9361820, -5.3530612, -8.9533844, -5.3091950, -3.3838019, 3.3564019
1: -7.4027824, -4.1315842, -7.4092550, -4.0988493, -2.8036025, 2.7752383
2: -7.4976530, -4.5862246, -7.5379148, -4.5660362, -2.6151133, 2.6182597
3: -11.2656317, -7.7246079, -11.3167067, -7.7324371, -3.0002294, 3.0367699
4: 6.5214138, 8.8054247, 6.5177813, 8.8144073, -1.9252262, 1.9323974
5: -8.9051685, -5.8653121, -8.9162674, -5.9007535, -2.6187463, 2.6502514
6: -12.0111408, -8.2183380, -12.0142260, -8.1884604, -3.6166334, 3.6000185
7: -3.2467051, -0.5717285, -3.2597790, -0.5174644, -2.7136269, 2.6880505
8: -6.9748068, -3.4556038, -7.0046282, -3.4885011, -2.7905626, 2.8047285
9: -5.5889773, -3.0277052, -5.5721159, -3.0294533, -2.2520053, 2.2517900

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4117554
time: 4.47 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4183549
time: 7.22 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9719296, -5.3010058, -3.3741384, 3.3662424
1: -7.3795280, -4.1591473, -7.4331570, -4.0712700, -2.7834020, 2.7786739
2: -7.4732456, -4.5950751, -7.5606685, -4.5572195, -2.5979490, 2.6305006
3: -11.2570133, -7.7643375, -11.3253937, -7.6923037, -3.0294619, 3.0190883
4: 6.5990024, 8.8007011, 6.4393682, 8.8191280, -1.8684998, 2.0036354
5: -8.8986826, -5.9286919, -8.9227610, -5.8372159, -2.6706643, 2.6043339
6: -11.9953022, -8.2697430, -12.0300741, -8.1387186, -3.6335001, 3.5666513
7: -3.1971812, -0.5803651, -3.3092690, -0.5087802, -2.6884010, 2.7289038
8: -6.9589167, -3.5266910, -7.0205030, -3.4174809, -2.8391552, 2.7981203
9: -5.5121059, -3.0363898, -5.6494908, -3.0207734, -2.1987100, 2.3099396

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3996524, upper bound: 1.4170009
time: 9.79 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062785, upper bound: 1.4170006
time: 5.96 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9362364, -5.3528647, -8.9719906, -5.3008099, -3.3953018, 3.3777277
1: -7.4028764, -4.1304970, -7.4332638, -4.0701799, -2.8128924, 2.7931814
2: -7.4986153, -4.5862231, -7.5616016, -4.5572181, -2.6257367, 2.6410115
3: -11.2656326, -7.7237730, -11.3253918, -7.6914606, -3.0378933, 3.0484958
4: 6.5197315, 8.8054256, 6.4376669, 8.8191299, -1.9321353, 2.0083752
5: -8.9051723, -5.8644085, -8.9227667, -5.8363128, -2.6729794, 2.6576436
6: -12.0111532, -8.2172947, -12.0300884, -8.1377153, -3.6517038, 3.6106224
7: -3.2469287, -0.5716588, -3.3095057, -0.5087115, -2.7233591, 2.7378469
8: -6.9750233, -3.4554715, -7.0207233, -3.4173274, -2.8207879, 2.8221450
9: -5.5903368, -3.0276990, -5.6508780, -3.0207691, -2.2625384, 2.3195775

Time for backsubstitution: 13.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4117551
time: 4.22 seconds

## Relational analysis of IS_B2_A1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4183547
time: 7.22 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9533844, -5.3091950, -8.9533844, -5.3091950, -3.4187794, 3.4187799
1: -7.4092550, -4.0988493, -7.4092550, -4.0988493, -2.8021104, 2.8021104
2: -7.5379148, -4.5660362, -7.5379148, -4.5660362, -2.6186218, 2.6186218
3: -11.3167067, -7.7324371, -11.3167067, -7.7324371, -3.0380492, 3.0380485
4: 6.5177813, 8.8144073, 6.5177813, 8.8144073, -1.9342928, 1.9342928
5: -8.9162674, -5.9007535, -8.9162674, -5.9007535, -2.6392565, 2.6392567
6: -12.0142260, -8.1884604, -12.0142260, -8.1884604, -3.6212025, 3.6212039
7: -3.2597790, -0.5174644, -3.2597790, -0.5174644, -2.7423146, 2.7423146
8: -7.0046282, -3.4885011, -7.0046282, -3.4885011, -2.7863436, 2.7863441
9: -5.5721159, -3.0294533, -5.5721159, -3.0294533, -2.2499316, 2.2499311

Time for backsubstitution: 13.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4037824
time: 5.27 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4105019
time: 9.40 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9533844, -5.3091950, -8.9719296, -5.3010058, -3.4291525, 3.4389243
1: -7.4092550, -4.0988493, -7.4331570, -4.0712700, -2.8103161, 2.8310997
2: -7.5379148, -4.5660362, -7.5606685, -4.5572195, -2.6284909, 2.6461427
3: -11.3167067, -7.7324371, -11.3253937, -7.6923037, -3.0672960, 3.0491111
4: 6.5177813, 8.8144073, 6.4393682, 8.8191280, -1.9397836, 2.0080962
5: -8.9162674, -5.9007535, -8.9227610, -5.8372159, -2.6862278, 2.6459980
6: -12.0142260, -8.1884604, -12.0300741, -8.1387186, -3.6553001, 3.6384411
7: -3.2597790, -0.5174644, -3.3092690, -0.5087802, -2.7509987, 2.7918046
8: -7.0046282, -3.4885011, -7.0205030, -3.4174809, -2.8446112, 2.8034284
9: -5.5721159, -3.0294533, -5.6494908, -3.0207734, -2.2593858, 2.3147812

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B2_A2_A1_B2_B1

### Relational analysis result of IS_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4103544, upper bound: 1.4105035
time: 4.38 seconds

## Relational analysis of IS_B2_A2_A1_B2_B2

### Relational analysis result of IS_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4105019
time: 5.89 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9719296, -5.3010058, -8.9533844, -5.3091950, -3.4389248, 3.4291523
1: -7.4331570, -4.0712700, -7.4092550, -4.0988493, -2.8310997, 2.8103161
2: -7.5606685, -4.5572195, -7.5379148, -4.5660362, -2.6461425, 2.6284907
3: -11.3253937, -7.6923037, -11.3167067, -7.7324371, -3.0491109, 3.0672958
4: 6.4393682, 8.8191280, 6.5177813, 8.8144073, -2.0080965, 1.9397836
5: -8.9227610, -5.8372159, -8.9162674, -5.9007535, -2.6459975, 2.6862276
6: -12.0300741, -8.1387186, -12.0142260, -8.1884604, -3.6384411, 3.6552992
7: -3.3092690, -0.5087802, -3.2597790, -0.5174644, -2.7918046, 2.7509987
8: -7.0205030, -3.4174809, -7.0046282, -3.4885011, -2.8034286, 2.8446112
9: -5.6494908, -3.0207734, -5.5721159, -3.0294533, -2.3147810, 2.2593858

Time for backsubstitution: 12.53 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.8940565586090088
rel_dist={4: [-1.418532051192658, 1.418531738496264]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597270, upper bound: 1.1476193
time: 4.55 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597270, upper bound: 1.1597265
time: 4.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.29 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 9.29
Output dim: 4, lower bound: -1.1597270, upper bound: 1.1476193
IS_B2, status: Status.UNKNOWN, split count: 1, time: 9.29
Output dim: 4, lower bound: -1.1597270, upper bound: 1.1597265

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -8.9354134, -5.3531485, -8.9197674, -5.3591337, -3.0747976, 3.0642061
1: -7.3978786, -4.1556597, -7.3834295, -4.1582737, -2.4915528, 2.4792249
2: -7.4789820, -4.5742297, -7.4752836, -4.5928955, -2.3597136, 2.3705935
3: -11.2633400, -7.7441711, -11.2590923, -7.7627215, -2.7165995, 2.7293210
4: 6.5621042, 8.8026104, 6.5971775, 8.8024073, -1.7164483, 1.6817441
5: -8.9045181, -5.9158378, -8.9024849, -5.9274836, -2.3516717, 2.3545356
6: -12.0150757, -8.2602482, -11.9991140, -8.2676954, -3.2565279, 3.2455177
7: -3.2182775, -0.5745678, -3.1996956, -0.5760213, -2.4589639, 2.4413095
8: -6.9675961, -3.5078919, -6.9664278, -3.5248680, -2.4698029, 2.4858575
9: -5.5373082, -3.0319777, -5.5144129, -3.0330338, -2.0561225, 2.0342047

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1476161, upper bound: 1.1476164
time: 7.35 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1476161, upper bound: 1.1476186
time: 6.23 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -8.9353848, -5.3531547, -8.9547949, -5.3097715, -3.1348925, 3.0973480
1: -7.3978682, -4.1556625, -7.4127445, -4.0997839, -2.5345149, 2.5058756
2: -7.4789782, -4.5742464, -7.5395980, -4.5640569, -2.3873639, 2.4142966
3: -11.2633362, -7.7442031, -11.3175611, -7.7311797, -2.7464662, 2.7724354
4: 6.5621371, 8.8026114, 6.5168567, 8.8161001, -1.7306523, 1.7592971
5: -8.9045162, -5.9158564, -8.9197712, -5.9002419, -2.3858943, 2.3683929
6: -12.0150452, -8.2602558, -12.0179567, -8.1891632, -3.3222857, 3.2671871
7: -3.2182646, -0.5745702, -3.2615128, -0.5138963, -2.4874606, 2.5195689
8: -6.9675961, -3.5079317, -7.0121222, -3.4872270, -2.5072727, 2.5304098
9: -5.5372639, -3.0319772, -5.5735240, -3.0261359, -2.0626545, 2.0932913

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579157, upper bound: 1.1597076
time: 7.89 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597079, upper bound: 1.1597078
time: 4.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.50 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 25.50
Output dim: 4, lower bound: -1.1476161, upper bound: 1.1476164
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 25.50
Output dim: 4, lower bound: -1.1476161, upper bound: 1.1476186
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 25.50
Output dim: 4, lower bound: -1.1579157, upper bound: 1.1597076
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 25.50
Output dim: 4, lower bound: -1.1597079, upper bound: 1.1597078

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -8.9197674, -5.3591337, -8.9197674, -5.3591337, -3.0532722, 3.0532725
1: -7.3834295, -4.1582737, -7.3834295, -4.1582737, -2.4770570, 2.4770563
2: -7.4752836, -4.5928955, -7.4752836, -4.5928955, -2.3521204, 2.3521204
3: -11.2590923, -7.7627215, -11.2590923, -7.7627215, -2.7101393, 2.7101398
4: 6.5971775, 8.8024073, 6.5971775, 8.8024073, -1.6781940, 1.6781938
5: -8.9024849, -5.9274836, -8.9024849, -5.9274836, -2.3421950, 2.3421941
6: -11.9991140, -8.2676954, -11.9991140, -8.2676954, -3.2386799, 3.2386799
7: -3.1996956, -0.5760213, -3.1996956, -0.5760213, -2.4393082, 2.4393082
8: -6.9664278, -3.5248680, -6.9664278, -3.5248680, -2.4664555, 2.4664552
9: -5.5144129, -3.0330338, -5.5144129, -3.0330338, -2.0305941, 2.0305941

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1475980, upper bound: 1.1457716
time: 5.54 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1475980, upper bound: 1.1476000
time: 5.09 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -8.9539986, -5.3135123, -8.9197674, -5.3591337, -3.0857096, 3.0978608
1: -7.4121752, -4.1022892, -7.3834295, -4.1582737, -2.5032430, 2.5179834
2: -7.5391188, -4.5643411, -7.4752836, -4.5928955, -2.3949862, 2.3790071
3: -11.3158779, -7.7315907, -11.2590923, -7.7627215, -2.7514038, 2.7394893
4: 6.5179892, 8.8160839, 6.5971775, 8.8024073, -1.7544322, 1.6922710
5: -8.9193459, -5.9011364, -8.9024849, -5.9274836, -2.3558307, 2.3690677
6: -12.0178480, -8.1929493, -11.9991140, -8.2676954, -3.2601185, 3.2999167
7: -3.2605205, -0.5149789, -3.1996956, -0.5760213, -2.5157518, 2.4665294
8: -7.0120916, -3.4879344, -6.9664278, -3.5248680, -2.5109205, 2.5027354
9: -5.5723653, -3.0261884, -5.5144129, -3.0330338, -2.0883701, 2.0371234

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1475980, upper bound: 1.1457718
time: 8.13 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1475980, upper bound: 1.1476007
time: 6.45 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -8.9353848, -5.3531547, -8.9528160, -5.3118868, -3.1315584, 3.0940249
1: -7.3978682, -4.1556625, -7.4088488, -4.1006479, -2.5322738, 2.5014758
2: -7.4789782, -4.5742464, -7.5375710, -4.5662374, -2.3847966, 2.4124138
3: -11.2633362, -7.7442031, -11.3154974, -7.7327323, -2.7445803, 2.7697732
4: 6.5621371, 8.8026114, 6.5185933, 8.8143959, -1.7288053, 1.7575841
5: -8.9045162, -5.9158564, -8.9159679, -5.9013958, -2.3844237, 2.3642311
6: -12.0150452, -8.2602558, -12.0141487, -8.1911993, -3.3207283, 3.2633600
7: -3.2182646, -0.5745702, -3.2590554, -0.5182445, -2.4822888, 2.5168867
8: -6.9675961, -3.5079317, -7.0046072, -3.4890087, -2.5050097, 2.5228350
9: -5.5372639, -3.0319772, -5.5712852, -3.0294909, -2.0591059, 2.0910618

Time for backsubstitution: 15.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579157, upper bound: 1.1579147
time: 5.02 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579157, upper bound: 1.1597071
time: 4.86 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -8.9353800, -5.3531590, -8.9712706, -5.3039875, -3.1419673, 3.1140811
1: -7.3978572, -4.1556625, -7.4325848, -4.0746641, -2.5397706, 2.5323744
2: -7.4789724, -4.5742517, -7.5590262, -4.5574288, -2.3946762, 2.4337232
3: -11.2633305, -7.7442064, -11.3241234, -7.6937857, -2.7838621, 2.7811372
4: 6.5621424, 8.8026066, 6.4425602, 8.8190098, -1.7351468, 1.8297470
5: -8.9045029, -5.9158597, -8.9224167, -5.8391047, -2.4315424, 2.3719187
6: -12.0150347, -8.2602615, -12.0299196, -8.1428289, -3.3534636, 3.2814994
7: -3.2182570, -0.5745823, -3.3082204, -0.5097123, -2.4903121, 2.5619278
8: -6.9675741, -3.5079372, -7.0201788, -3.4181724, -2.5640173, 2.5434859
9: -5.5372596, -3.0319886, -5.6464529, -3.0208187, -2.0697062, 2.1505995

Time for backsubstitution: 12.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1475976, upper bound: 1.1597095
time: 6.27 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1475976, upper bound: 1.1482919
time: 7.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.89 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 4, lower bound: -1.1475980, upper bound: 1.1457716
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 4, lower bound: -1.1475980, upper bound: 1.1476000
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 4, lower bound: -1.1475980, upper bound: 1.1457718
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 4, lower bound: -1.1475980, upper bound: 1.1476007
IS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 4, lower bound: -1.1579157, upper bound: 1.1579147
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 4, lower bound: -1.1579157, upper bound: 1.1597071
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 4, lower bound: -1.1475976, upper bound: 1.1597095
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.89
Output dim: 4, lower bound: -1.1475976, upper bound: 1.1482919

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9197674, -5.3591337, -3.0498886, 3.0499344
1: -7.3795280, -4.1591473, -7.3834295, -4.1582737, -2.4726396, 2.4748232
2: -7.4732456, -4.5950751, -7.4752836, -4.5928955, -2.3503346, 2.3495388
3: -11.2570133, -7.7643375, -11.2590923, -7.7627215, -2.7074938, 2.7082131
4: 6.5990024, 8.8007011, 6.5971775, 8.8024073, -1.6764126, 1.6763477
5: -8.8986826, -5.9286919, -8.9024849, -5.9274836, -2.3380294, 2.3407283
6: -11.9953022, -8.2697430, -11.9991140, -8.2676954, -3.2348480, 3.2370634
7: -3.1971812, -0.5803651, -3.1996956, -0.5760213, -2.4365292, 2.4341230
8: -6.9589167, -3.5266910, -6.9664278, -3.5248680, -2.4588652, 2.4641716
9: -5.5121059, -3.0363898, -5.5144129, -3.0330338, -2.0284431, 2.0270436

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1457712
time: 5.83 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1457746
time: 9.72 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.9361038, -5.3533535, -8.9197636, -5.3591380, -3.0697937, 3.0603378
1: -7.4026504, -4.1331730, -7.3834167, -4.1582785, -2.5032611, 2.4990242
2: -7.4963140, -4.5862331, -7.4752784, -4.5928998, -2.3760586, 2.3594680
3: -11.2655725, -7.7257738, -11.2590857, -7.7627254, -2.7190619, 2.7473271
4: 6.5237598, 8.8053169, 6.5971818, 8.8024015, -1.7454116, 1.6826890
5: -8.9051266, -5.8665733, -8.9024715, -5.9274864, -2.3456969, 2.3927147
6: -12.0110626, -8.2197924, -11.9991016, -8.2677021, -3.2529716, 3.2836661
7: -3.2463923, -0.5718788, -3.1996884, -0.5760353, -2.4787645, 2.4420910
8: -6.9745007, -3.4557991, -6.9664078, -3.5248713, -2.4795175, 2.5222740
9: -5.5868039, -3.0277123, -5.5144067, -3.0330429, -2.0874519, 2.0376501

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1475997
time: 6.49 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1476032
time: 7.10 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.9520168, -5.3156281, -8.9197674, -5.3591337, -3.0823860, 3.0945287
1: -7.4082818, -4.1031513, -7.3834295, -4.1582737, -2.4988427, 2.5157418
2: -7.5370908, -4.5665193, -7.4752836, -4.5928955, -2.3931017, 2.3764393
3: -11.3138103, -7.7331457, -11.2590923, -7.7627215, -2.7487402, 2.7376027
4: 6.5197268, 8.8143778, 6.5971775, 8.8024073, -1.7527187, 1.6904242
5: -8.9155445, -5.9022932, -8.9024849, -5.9274836, -2.3516679, 2.3676600
6: -12.0140362, -8.1950083, -11.9991140, -8.2676954, -3.2562904, 3.2983561
7: -3.2580624, -0.5193303, -3.1996956, -0.5760213, -2.5130687, 2.4613569
8: -7.0045757, -3.4897151, -6.9664278, -3.5248680, -2.5033460, 2.5004716
9: -5.5701270, -3.0295424, -5.5144129, -3.0330338, -2.0861404, 2.0335753

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1457742
time: 5.32 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1457742
time: 5.19 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -8.9704742, -5.3077273, -8.9197636, -5.3591380, -3.1024399, 3.1049509
1: -7.4319992, -4.0771666, -7.3834167, -4.1582785, -2.5297351, 2.5232399
2: -7.5585585, -4.5577102, -7.4752784, -4.5928998, -2.4143920, 2.3863249
3: -11.3224373, -7.6942043, -11.2590857, -7.7627254, -2.7597656, 2.7768846
4: 6.4437046, 8.8189936, 6.5971818, 8.8024015, -1.8247869, 1.6967652
5: -8.9219971, -5.8399715, -8.9024715, -5.9274864, -2.3593574, 2.4201016
6: -12.0298061, -8.1465912, -11.9991016, -8.2677021, -3.2744293, 3.3310788
7: -3.3072181, -0.5107973, -3.1996884, -0.5760353, -2.5578122, 2.4693811
8: -7.0201473, -3.4188290, -6.9664078, -3.5248713, -2.5239973, 2.5595105
9: -5.6452942, -3.0208716, -5.5144067, -3.0330429, -2.1455765, 2.0441771

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1476006
time: 7.02 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1475996
time: 4.90 seconds

## BFS IS instance: IS_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.9334068, -5.3552747, -8.9528160, -5.3118868, -3.1281805, 3.0906878
1: -7.3939838, -4.1565323, -7.4088488, -4.1006479, -2.5278497, 2.4992423
2: -7.4769182, -4.5764279, -7.5375710, -4.5662374, -2.3830032, 2.4098165
3: -11.2612591, -7.7458248, -11.3154974, -7.7327323, -2.7419710, 2.7678535
4: 6.5639625, 8.8009062, 6.5185933, 8.8143959, -1.7270248, 1.7557375
5: -8.9007111, -5.9170771, -8.9159679, -5.9013958, -2.3802590, 2.3627386
6: -12.0112371, -8.2622280, -12.0141487, -8.1911993, -3.3168430, 3.2617202
7: -3.2157514, -0.5789130, -3.2590554, -0.5182445, -2.4795022, 2.5117011
8: -6.9600806, -3.5097528, -7.0046072, -3.4890087, -2.4974184, 2.5205321
9: -5.5349426, -3.0353355, -5.5712852, -3.0294909, -2.0569539, 2.0875118

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_B2_B1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1579137
time: 6.62 seconds

## Relational analysis of IS_B2_B1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1464397
time: 5.46 seconds

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9517450, -5.3475275, -8.9528160, -5.3118868, -3.1478510, 3.1008103
1: -7.4170418, -4.1314440, -7.4088488, -4.1006479, -2.5560250, 2.5227222
2: -7.4995031, -4.5675931, -7.5375710, -4.5662374, -2.4083066, 2.4196897
3: -11.2698097, -7.7075620, -11.3154974, -7.7327323, -2.7532225, 2.7953298
4: 6.4895415, 8.8054314, 6.5185933, 8.8143959, -1.7835279, 1.7609599
5: -8.9071293, -5.8554344, -8.9159679, -5.9013958, -2.3868957, 2.4084921
6: -12.0269337, -8.2131844, -12.0141487, -8.1911993, -3.3340034, 3.3074582
7: -3.2651172, -0.5705254, -3.2590554, -0.5182445, -2.5066404, 2.5196085
8: -6.9754915, -3.4387050, -7.0046072, -3.4890087, -2.5140476, 2.5380299
9: -5.6089287, -3.0266628, -5.5712852, -3.0294909, -2.1109028, 2.0969393

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1597061
time: 7.93 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1482936
time: 6.24 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.9197636, -5.3591380, -8.9704742, -5.3077273, -3.1049509, 3.1024399
1: -7.3834167, -4.1582785, -7.4319992, -4.0771666, -2.5232399, 2.5297351
2: -7.4752784, -4.5928998, -7.5585585, -4.5577102, -2.3863249, 2.4143922
3: -11.2590857, -7.7627254, -11.3224373, -7.6942043, -2.7768850, 2.7597661
4: 6.5971818, 8.8024015, 6.4437046, 8.8189936, -1.6967652, 1.8247870
5: -8.9024715, -5.9274864, -8.9219971, -5.8399715, -2.4201016, 2.3593574
6: -11.9991016, -8.2677021, -12.0298061, -8.1465912, -3.3310785, 3.2744298
7: -3.1996884, -0.5760353, -3.3072181, -0.5107973, -2.4693809, 2.5578120
8: -6.9664078, -3.5248713, -7.0201473, -3.4188290, -2.5595107, 2.5239973
9: -5.5144067, -3.0330429, -5.6452942, -3.0208716, -2.0441773, 2.1455767

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1579161
time: 5.61 seconds

## Relational analysis of IS_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1597072
time: 8.16 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -8.9553585, -5.3070850, -8.9718456, -5.3012981, -3.1616669, 3.1712103
1: -7.4131432, -4.0979881, -7.4330049, -4.0728636, -2.5536289, 2.5743756
2: -7.5399342, -4.5638604, -7.5593634, -4.5572271, -2.4088755, 2.4260399
3: -11.3187666, -7.7308893, -11.3253317, -7.6934857, -2.8108988, 2.7932224
4: 6.5160499, 8.8161058, 6.4417410, 8.8190193, -1.7655716, 1.8313345
5: -8.9200592, -5.8996010, -8.9227209, -5.8384805, -2.4372418, 2.3956966
6: -12.0180244, -8.1864290, -12.0299988, -8.1401215, -3.3607378, 3.3439074
7: -3.2622163, -0.5131278, -3.3089371, -0.5089326, -2.5521698, 2.5763841
8: -7.0121231, -3.4867258, -7.0201998, -3.4177027, -2.5663052, 2.5228388
9: -5.5743508, -3.0261102, -5.6472859, -3.0207810, -2.1023049, 2.1524518

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1464407
time: 6.79 seconds

## Relational analysis of IS_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1482938
time: 8.23 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.77 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1457712
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1457746
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1475997
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1476032
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1457742
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1457742
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1476006
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1475996
IS_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1579137
IS_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1464397
IS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1597061
IS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1482936
IS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1579161
IS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1597072
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1464407
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 27.77
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1482938

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9177856, -5.3612566, -3.0465508, 3.0465500
1: -7.3795280, -4.1591473, -7.3795280, -4.1591473, -2.4704061, 2.4704058
2: -7.4732456, -4.5950751, -7.4732456, -4.5950751, -2.3477530, 2.3477528
3: -11.2570133, -7.7643375, -11.2570133, -7.7643375, -2.7055669, 2.7055669
4: 6.5990024, 8.8007011, 6.5990024, 8.8007011, -1.6745665, 1.6745665
5: -8.8986826, -5.9286919, -8.8986826, -5.9286919, -2.3365626, 2.3365631
6: -11.9953022, -8.2697430, -11.9953022, -8.2697430, -3.2332315, 3.2332315
7: -3.1971812, -0.5803651, -3.1971812, -0.5803651, -2.4313440, 2.4313440
8: -6.9589167, -3.5266910, -6.9589167, -3.5266910, -2.4565821, 2.4565818
9: -5.5121059, -3.0363898, -5.5121059, -3.0363898, -2.0248928, 2.0248926

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455207, upper bound: 1.1416134
time: 5.97 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457707, upper bound: 1.1457710
time: 6.08 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9360638, -5.3535156, -3.0566621, 3.0663903
1: -7.3795280, -4.1591473, -7.4025807, -4.1340647, -2.4938831, 2.4985139
2: -7.4732456, -4.5950751, -7.4955788, -4.5862384, -2.3576713, 2.3727701
3: -11.2570133, -7.7643375, -11.2655201, -7.7264118, -2.7440386, 2.7167702
4: 6.5990024, 8.8007011, 6.5250444, 8.8052273, -1.6797879, 1.7425355
5: -8.8986826, -5.9286919, -8.9050941, -5.8672638, -2.3880858, 2.3432021
6: -11.9953022, -8.2697430, -12.0110035, -8.2205877, -3.2790542, 3.2502794
7: -3.1971812, -0.5803651, -3.2462196, -0.5719786, -2.4392519, 2.4723415
8: -6.9589167, -3.5266910, -6.9743338, -3.4559054, -2.5145717, 2.4732175
9: -5.5121059, -3.0363898, -5.5855365, -3.0277166, -2.0343223, 2.0829298

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1416135, upper bound: 1.1455219
time: 5.14 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457705, upper bound: 1.1457703
time: 7.16 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9360638, -5.3535156, -8.9177856, -5.3612566, -3.0663910, 3.0566618
1: -7.4025807, -4.1340647, -7.3795280, -4.1591473, -2.4985137, 2.4938829
2: -7.4955788, -4.5862384, -7.4732456, -4.5950751, -2.3727703, 2.3576713
3: -11.2655201, -7.7264118, -11.2570133, -7.7643375, -2.7167706, 2.7440388
4: 6.5250444, 8.8052273, 6.5990024, 8.8007011, -1.7425356, 1.6797884
5: -8.9050941, -5.8672638, -8.8986826, -5.9286919, -2.3432021, 2.3880858
6: -12.0110035, -8.2205877, -11.9953022, -8.2697430, -3.2502794, 3.2790542
7: -3.2462196, -0.5719786, -3.1971812, -0.5803651, -2.4723415, 2.4392519
8: -6.9743338, -3.4559054, -6.9589167, -3.5266910, -2.4732170, 2.5145719
9: -5.5855365, -3.0277166, -5.5121059, -3.0363898, -2.0829296, 2.0343220

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455204, upper bound: 1.1434570
time: 7.03 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457704, upper bound: 1.1476008
time: 5.39 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9362364, -5.3528647, -8.9362364, -5.3528647, -3.0776920, 3.0776927
1: -7.4028764, -4.1304970, -7.4028764, -4.1304970, -2.5105934, 2.5105939
2: -7.4986153, -4.5862231, -7.4986153, -4.5862231, -2.3854342, 2.3854342
3: -11.2656326, -7.7237730, -11.2656326, -7.7237730, -2.7536569, 2.7536566
4: 6.5197315, 8.8054256, 6.5197315, 8.8054256, -1.7523423, 1.7523422
5: -8.9051723, -5.8644085, -8.9051723, -5.8644085, -2.3956971, 2.3956976
6: -12.0111532, -8.2172947, -12.0111532, -8.2172947, -3.2923193, 3.2923188
7: -3.2469287, -0.5716588, -3.2469287, -0.5716588, -2.4828229, 2.4828227
8: -6.9750233, -3.4554715, -6.9750233, -3.4554715, -2.4955845, 2.4955840
9: -5.5903368, -3.0276990, -5.5903368, -3.0276990, -2.0960417, 2.0960417

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455204, upper bound: 1.1434590
time: 6.22 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457704, upper bound: 1.1475989
time: 6.18 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9520168, -5.3156281, -8.9177856, -5.3612566, -3.0790482, 3.0911441
1: -7.4082818, -4.1031513, -7.3795280, -4.1591473, -2.4966102, 2.5113111
2: -7.5370908, -4.5665193, -7.4732456, -4.5950751, -2.3905039, 2.3746533
3: -11.3138103, -7.7331457, -11.2570133, -7.7643375, -2.7468233, 2.7349567
4: 6.5197268, 8.8143778, 6.5990024, 8.8007011, -1.7508726, 1.6886432
5: -8.9155445, -5.9022932, -8.8986826, -5.9286919, -2.3502021, 2.3634949
6: -12.0140362, -8.1950083, -11.9953022, -8.2697430, -3.2546740, 3.2944732
7: -3.2580624, -0.5193303, -3.1971812, -0.5803651, -2.5078840, 2.4585812
8: -7.0045757, -3.4897151, -6.9589167, -3.5266910, -2.5010395, 2.4928818
9: -5.5701270, -3.0295424, -5.5121059, -3.0363898, -2.0825896, 2.0314243

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1538026, upper bound: 1.1455203
time: 6.85 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579121, upper bound: 1.1457696
time: 7.92 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9520168, -5.3156281, -8.9360638, -5.3535156, -3.0891595, 3.1109846
1: -7.4082818, -4.1031513, -7.4025807, -4.1340647, -2.5200872, 2.5394635
2: -7.5370908, -4.5665193, -7.4955788, -4.5862384, -2.4004147, 2.3996708
3: -11.3138103, -7.7331457, -11.2655201, -7.7264118, -2.7741313, 2.7461600
4: 6.5197268, 8.8143778, 6.5250444, 8.8052273, -1.7560942, 1.7454754
5: -8.9155445, -5.9022932, -8.9050941, -5.8672638, -2.3963041, 2.3701344
6: -12.0140362, -8.1950083, -12.0110035, -8.2205877, -3.3005462, 3.3116302
7: -3.2580624, -0.5193303, -3.2462196, -0.5719786, -2.5157914, 2.4858444
8: -7.0045757, -3.4897151, -6.9743338, -3.4559054, -2.5185046, 2.5095177
9: -5.5701270, -3.0295424, -5.5855365, -3.0277166, -2.0920186, 2.0851111

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1538026, upper bound: 1.1455194
time: 5.01 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579121, upper bound: 1.1457691
time: 5.23 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9704266, -5.3078909, -8.9177856, -5.3612566, -3.0990334, 3.1012740
1: -7.4319181, -4.0780630, -7.3795280, -4.1591473, -2.5252490, 2.5184054
2: -7.5578704, -4.5577159, -7.4732456, -4.5950751, -2.4113426, 2.3845279
3: -11.3223877, -7.6948514, -11.2570133, -7.7643375, -2.7578211, 2.7735913
4: 6.4450035, 8.8189030, 6.5990024, 8.8007011, -1.8218830, 1.6938610
5: -8.9219627, -5.8406639, -8.8986826, -5.9286919, -2.3568621, 2.4154639
6: -12.0297480, -8.1473618, -11.9953022, -8.2697430, -3.2717371, 3.3264627
7: -3.3070374, -0.5108945, -3.1971812, -0.5803651, -2.5513921, 2.4665351
8: -7.0199804, -3.4189520, -6.9589167, -3.5266910, -2.5177088, 2.5517952
9: -5.6440072, -3.0208740, -5.5121059, -3.0363898, -2.1410413, 2.0408485

Time for backsubstitution: 12.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1576956, upper bound: 1.1434565
time: 5.77 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1476004
time: 6.39 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9706211, -5.3072371, -8.9362364, -5.3528647, -3.1103497, 3.1223066
1: -7.4322629, -4.0744839, -7.4028764, -4.1304970, -2.5370884, 2.5490060
2: -7.5607991, -4.5577002, -7.4986153, -4.5862231, -2.4231369, 2.4122915
3: -11.3224983, -7.6921792, -11.2656326, -7.7237730, -2.7873006, 2.7832272
4: 6.4396315, 8.8191013, 6.5197315, 8.8054256, -1.8310037, 1.7552805
5: -8.9220409, -5.8378019, -8.9051723, -5.8644085, -2.4049540, 2.4225974
6: -12.0298986, -8.1441822, -12.0111532, -8.2172947, -3.3137770, 3.3467457
7: -3.3077831, -0.5105767, -3.2469287, -0.5716588, -2.5618563, 2.4963207
8: -7.0206690, -3.4184527, -6.9750233, -3.4554715, -2.5362825, 2.5327723
9: -5.6488848, -3.0208573, -5.5903368, -3.0276990, -2.1542072, 2.0982294

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1576956, upper bound: 1.1434583
time: 5.40 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1476007
time: 5.62 seconds

## BFS IS instance: IS_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9520168, -5.3156281, -3.0911446, 3.0790477
1: -7.3795280, -4.1591473, -7.4082818, -4.1031513, -2.5113108, 2.4966099
2: -7.4732456, -4.5950751, -7.5370908, -4.5665193, -2.3746533, 2.3905036
3: -11.2570133, -7.7643375, -11.3138103, -7.7331457, -2.7349567, 2.7468233
4: 6.5990024, 8.8007011, 6.5197268, 8.8143778, -1.6886435, 1.7508729
5: -8.8986826, -5.9286919, -8.9155445, -5.9022932, -2.3634953, 2.3502016
6: -11.9953022, -8.2697430, -12.0140362, -8.1950083, -3.2944732, 3.2546740
7: -3.1971812, -0.5803651, -3.2580624, -0.5193303, -2.4585814, 2.5078835
8: -6.9589167, -3.5266910, -7.0045757, -3.4897151, -2.4928818, 2.5010390
9: -5.5121059, -3.0363898, -5.5701270, -3.0295424, -2.0314245, 2.0825896

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B2_B1_A1_A1_A1

### Relational analysis result of IS_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455179, upper bound: 1.1538031
time: 9.79 seconds

## Relational analysis of IS_B2_B1_A1_A1_A2

### Relational analysis result of IS_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457679, upper bound: 1.1579122
time: 5.91 seconds

## BFS IS instance: IS_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.9533844, -5.3091950, -8.9533844, -5.3091950, -3.1478806, 3.1478798
1: -7.4092550, -4.0988493, -7.4092550, -4.0988493, -2.5417187, 2.5417190
2: -7.5379148, -4.5660362, -7.5379148, -4.5660362, -2.3972745, 2.3972745
3: -11.3167067, -7.7324371, -11.3167067, -7.7324371, -2.7799363, 2.7799363
4: 6.5177813, 8.8144073, 6.5177813, 8.8144073, -1.7575154, 1.7575154
5: -8.9162674, -5.9007535, -8.9162674, -5.9007535, -2.3865652, 2.3865652
6: -12.0142260, -8.1884604, -12.0142260, -8.1884604, -3.3241072, 3.3241074
7: -3.2597790, -0.5174644, -3.2597790, -0.5174644, -2.5414710, 2.5414710
8: -7.0046282, -3.4885011, -7.0046282, -3.4885011, -2.4998999, 2.4998994
9: -5.5721159, -3.0294533, -5.5721159, -3.0294533, -2.0894616, 2.0894611

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B2_B1_A1_A2_B1

### Relational analysis result of IS_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1416107, upper bound: 1.1461869
time: 6.30 seconds

## Relational analysis of IS_B2_B1_A1_A2_B2

### Relational analysis result of IS_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457677, upper bound: 1.1457672
time: 5.29 seconds

## BFS IS instance: IS_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.9360638, -5.3535156, -8.9520168, -5.3156281, -3.1109848, 3.0891590
1: -7.4025807, -4.1340647, -7.4082818, -4.1031513, -2.5394638, 2.5200875
2: -7.4955788, -4.5862384, -7.5370908, -4.5665193, -2.3996711, 2.4004152
3: -11.2655201, -7.7264118, -11.3138103, -7.7331457, -2.7461596, 2.7741313
4: 6.5250444, 8.8052273, 6.5197268, 8.8143778, -1.7454753, 1.7560942
5: -8.9050941, -5.8672638, -8.9155445, -5.9022932, -2.3701339, 2.3963041
6: -12.0110035, -8.2205877, -12.0140362, -8.1950083, -3.3116298, 3.3005459
7: -3.2462196, -0.5719786, -3.2580624, -0.5193303, -2.4858441, 2.5157914
8: -6.9743338, -3.4559054, -7.0045757, -3.4897151, -2.5095177, 2.5185049
9: -5.5855365, -3.0277166, -5.5701270, -3.0295424, -2.0851114, 2.0920188

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B2_B1_A2_A1_A1

### Relational analysis result of IS_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455176, upper bound: 1.1556082
time: 6.64 seconds

## Relational analysis of IS_B2_B1_A2_A1_A2

### Relational analysis result of IS_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457676, upper bound: 1.1597046
time: 6.47 seconds

## BFS IS instance: IS_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -8.9717989, -5.3014607, -8.9533844, -5.3091950, -3.1677999, 3.1579981
1: -7.4329185, -4.0737591, -7.4092550, -4.0988493, -2.5704248, 2.5488145
2: -7.5587149, -4.5572333, -7.5379148, -4.5660362, -2.4227586, 2.4071398
3: -11.3252811, -7.6941319, -11.3167067, -7.7324371, -2.7909274, 2.8077226
4: 6.4430394, 8.8189306, 6.5177813, 8.8144073, -1.8284245, 1.7627358
5: -8.9226856, -5.8391700, -8.9162674, -5.9007535, -2.3931985, 2.4326124
6: -12.0299377, -8.1408863, -12.0142260, -8.1884604, -3.3412724, 3.3561218
7: -3.3087556, -0.5090301, -3.2597790, -0.5174644, -2.5699682, 2.5494242
8: -7.0200343, -3.4178295, -7.0046282, -3.4885011, -2.5165606, 2.5585899
9: -5.6459980, -3.0207844, -5.5721159, -3.0294533, -2.1479146, 2.0988982

Time for backsubstitution: 12.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B2_B1_A2_A2_A1

### Relational analysis result of IS_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455176, upper bound: 1.1441343
time: 6.34 seconds

## Relational analysis of IS_B2_B1_A2_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457676, upper bound: 1.1482920
time: 6.95 seconds

## BFS IS instance: IS_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9704266, -5.3078909, -3.1012745, 3.0990331
1: -7.3795280, -4.1591473, -7.4319181, -4.0780630, -2.5184052, 2.5252488
2: -7.4732456, -4.5950751, -7.5578704, -4.5577159, -2.3845282, 2.4113426
3: -11.2570133, -7.7643375, -11.3223877, -7.6948514, -2.7735910, 2.7578211
4: 6.5990024, 8.8007011, 6.4450035, 8.8189030, -1.6938610, 1.8218832
5: -8.8986826, -5.9286919, -8.9219627, -5.8406639, -2.4154639, 2.3568625
6: -11.9953022, -8.2697430, -12.0297480, -8.1473618, -3.3264632, 3.2717371
7: -3.1971812, -0.5803651, -3.3070374, -0.5108945, -2.4665351, 2.5513921
8: -6.9589167, -3.5266910, -7.0199804, -3.4189520, -2.5517950, 2.5177090
9: -5.5121059, -3.0363898, -5.6440072, -3.0208740, -2.0408485, 2.1410415

Time for backsubstitution: 12.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B2_B2_A1_A1_B1

### Relational analysis result of IS_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1416128, upper bound: 1.1576983
time: 5.65 seconds

## Relational analysis of IS_B2_B2_A1_A1_B2

### Relational analysis result of IS_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457698, upper bound: 1.1579150
time: 6.10 seconds

## BFS IS instance: IS_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.9362364, -5.3528647, -8.9706211, -5.3072371, -3.1223059, 3.1103497
1: -7.4028764, -4.1304970, -7.4322629, -4.0744839, -2.5490060, 2.5370877
2: -7.4986153, -4.5862231, -7.5607991, -4.5577002, -2.4122915, 2.4231369
3: -11.2656326, -7.7237730, -11.3224983, -7.6921792, -2.7832270, 2.7873006
4: 6.5197315, 8.8054256, 6.4396315, 8.8191013, -1.7552805, 1.8310037
5: -8.9051723, -5.8644085, -8.9220409, -5.8378019, -2.4225974, 2.4049544
6: -12.0111532, -8.2172947, -12.0298986, -8.1441822, -3.3467455, 3.3137765
7: -3.2469287, -0.5716588, -3.3077831, -0.5105767, -2.4963207, 2.5618563
8: -6.9750233, -3.4554715, -7.0206690, -3.4184527, -2.5327721, 2.5362828
9: -5.5903368, -3.0276990, -5.6488848, -3.0208573, -2.0982289, 2.1542070

Time for backsubstitution: 12.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B2_B2_A1_A2_B1

### Relational analysis result of IS_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1416128, upper bound: 1.1594980
time: 7.35 seconds

## Relational analysis of IS_B2_B2_A1_A2_B2

### Relational analysis result of IS_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457698, upper bound: 1.1597051
time: 15.80 seconds

## BFS IS instance: IS_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -8.9533844, -5.3091950, -8.9717989, -5.3014607, -3.1579981, 3.1677997
1: -7.4092550, -4.0988493, -7.4329185, -4.0737591, -2.5488145, 2.5704253
2: -7.5379148, -4.5660362, -7.5587149, -4.5572333, -2.4071398, 2.4227583
3: -11.3167067, -7.7324371, -11.3252811, -7.6941319, -2.8077226, 2.7909274
4: 6.5177813, 8.8144073, 6.4430394, 8.8189306, -1.7627358, 1.8284243
5: -8.9162674, -5.9007535, -8.9226856, -5.8391700, -2.4326122, 2.3931990
6: -12.0142260, -8.1884604, -12.0299377, -8.1408863, -3.3561215, 3.3412721
7: -3.2597790, -0.5174644, -3.3087556, -0.5090301, -2.5494242, 2.5699685
8: -7.0046282, -3.4885011, -7.0200343, -3.4178295, -2.5585904, 2.5165606
9: -5.5721159, -3.0294533, -5.6459980, -3.0207844, -2.0988984, 2.1479149

Time for backsubstitution: 12.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B2_B2_A2_A1_B1

### Relational analysis result of IS_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1538024, upper bound: 1.1461863
time: 5.55 seconds

## Relational analysis of IS_B2_B2_A2_A1_B2

### Relational analysis result of IS_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579120, upper bound: 1.1464403
time: 4.95 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.49 seconds
IS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1455207, upper bound: 1.1416134
IS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1457707, upper bound: 1.1457710
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1416135, upper bound: 1.1455219
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1457705, upper bound: 1.1457703
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1455204, upper bound: 1.1434570
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1457704, upper bound: 1.1476008
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1455204, upper bound: 1.1434590
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1457704, upper bound: 1.1475989
IS_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1538026, upper bound: 1.1455203
IS_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1579121, upper bound: 1.1457696
IS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1538026, upper bound: 1.1455194
IS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1579121, upper bound: 1.1457691
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1576956, upper bound: 1.1434565
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1476004
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1576956, upper bound: 1.1434583
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1476007
IS_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1455179, upper bound: 1.1538031
IS_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1457679, upper bound: 1.1579122
IS_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1416107, upper bound: 1.1461869
IS_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1457677, upper bound: 1.1457672
IS_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1455176, upper bound: 1.1556082
IS_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1457676, upper bound: 1.1597046
IS_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1455176, upper bound: 1.1441343
IS_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1457676, upper bound: 1.1482920
IS_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1416128, upper bound: 1.1576983
IS_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1457698, upper bound: 1.1579150
IS_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1416128, upper bound: 1.1594980
IS_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1457698, upper bound: 1.1597051
IS_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1538024, upper bound: 1.1461863
IS_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 23.49
Output dim: 4, lower bound: -1.1579120, upper bound: 1.1464403
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 23.49
Output dim: 4, lower bound: -1.1457694, upper bound: 1.1482938
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.7199985980987549
rel_dist={4: [-1.1597516071645249, 1.1597543492710418]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0515290, upper bound: 1.0415948
time: 5.93 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516541, upper bound: 1.0516528
time: 7.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.65 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 13.65
Output dim: 4, lower bound: -1.0515290, upper bound: 1.0415948
IS_B2, status: Status.UNKNOWN, split count: 1, time: 13.65
Output dim: 4, lower bound: -1.0516541, upper bound: 1.0516528

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -8.9325085, -5.3542285, -8.9197674, -5.3591337, -2.9822130, 2.9736247
1: -7.3952055, -4.1561289, -7.3834295, -4.1582737, -2.4047513, 2.3947177
2: -7.4782953, -4.5776968, -7.4752836, -4.5928955, -2.2872758, 2.2961633
3: -11.2625790, -7.7476139, -11.2590923, -7.7627215, -2.6309509, 2.6412959
4: 6.5686035, 8.8025713, 6.5971775, 8.8024073, -1.6513438, 1.6230619
5: -8.9041462, -5.9180174, -8.9024849, -5.9274836, -2.2675891, 2.2699542
6: -12.0121126, -8.2615852, -11.9991140, -8.2676954, -3.1550283, 3.1460962
7: -3.2148366, -0.5748340, -3.1996956, -0.5760213, -2.3809810, 2.3665950
8: -6.9673834, -3.5110519, -6.9664278, -3.5248680, -2.3760662, 2.3891397
9: -5.5330510, -3.0321689, -5.5144129, -3.0330338, -1.9987972, 1.9809420

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0415904, upper bound: 1.0415930
time: 5.63 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0415904, upper bound: 1.0415903
time: 8.11 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -8.9353790, -5.3531547, -8.9540720, -5.3130360, -3.0405302, 3.0080988
1: -7.3978667, -4.1556625, -7.4122486, -4.1019783, -2.4459319, 2.4212663
2: -7.4789758, -4.5742502, -7.5391769, -4.5643058, -2.3133650, 2.3414295
3: -11.2633352, -7.7442079, -11.3160095, -7.7315402, -2.6607265, 2.6848221
4: 6.5621452, 8.8026114, 6.5178452, 8.8160505, -1.6724560, 1.6993644
5: -8.9045143, -5.9158607, -8.9193325, -5.9010239, -2.3005276, 2.2858090
6: -12.0150394, -8.2602577, -12.0178518, -8.1924686, -3.2192674, 3.1684690
7: -3.2182608, -0.5745707, -3.2606454, -0.5148420, -2.4113913, 2.4425821
8: -6.9675961, -3.5079393, -7.0120811, -3.4878447, -2.4111414, 2.4350729
9: -5.5372553, -3.0319786, -5.5725117, -3.0261893, -2.0100098, 2.0387962

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0497399, upper bound: 1.0516361
time: 7.70 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516363, upper bound: 1.0516380
time: 7.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 29.96 seconds
IS_B1_A1, status: Status.VERIFIED, split count: 2, time: 29.96
Output dim: 4, lower bound: -1.0415904, upper bound: 1.0415930
IS_B1_A2, status: Status.VERIFIED, split count: 2, time: 29.96
Output dim: 4, lower bound: -1.0415904, upper bound: 1.0415903
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 29.96
Output dim: 4, lower bound: -1.0497399, upper bound: 1.0516361
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 29.96
Output dim: 4, lower bound: -1.0516363, upper bound: 1.0516380

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -8.9353790, -5.3531547, -8.9520912, -5.3151517, -3.0371952, 3.0047746
1: -7.3978667, -4.1556625, -7.4083529, -4.1028433, -2.4436920, 2.4168670
2: -7.4789758, -4.5742502, -7.5371475, -4.5664854, -2.3107967, 2.3395452
3: -11.2633352, -7.7442079, -11.3139410, -7.7330952, -2.6588402, 2.6821589
4: 6.5621452, 8.8026114, 6.5195808, 8.8143463, -1.6706095, 1.6976519
5: -8.9045143, -5.9158607, -8.9155254, -5.9021778, -2.2990570, 2.2816463
6: -12.0150394, -8.2602577, -12.0140400, -8.1945267, -3.2177072, 3.1646409
7: -3.2182608, -0.5745707, -3.2581871, -0.5191908, -2.4062200, 2.4398997
8: -6.9675961, -3.5079393, -7.0045662, -3.4896240, -2.4088783, 2.4274981
9: -5.5372553, -3.0319786, -5.5702734, -3.0295439, -2.0064607, 2.0365667

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B2_B1_A1

### Relational analysis result of IS_B2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0497399, upper bound: 1.0497387
time: 7.59 seconds

## Relational analysis of IS_B2_B1_A2

### Relational analysis result of IS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0497399, upper bound: 1.0516381
time: 6.37 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -8.9353714, -5.3531623, -8.9702892, -5.3074436, -3.0471935, 3.0245385
1: -7.3978519, -4.1556635, -7.4318361, -4.0778503, -2.4506574, 2.4463847
2: -7.4789705, -4.5742569, -7.5578356, -4.5577478, -2.3205929, 2.3602605
3: -11.2633286, -7.7442126, -11.3224239, -7.6950331, -2.6972136, 2.6930902
4: 6.5621490, 8.8026056, 6.4450049, 8.8188009, -1.6761458, 1.7677165
5: -8.9044971, -5.9158635, -8.9219198, -5.8406062, -2.3441300, 2.2886195
6: -12.0150251, -8.2602634, -12.0291395, -8.1469898, -3.2495995, 3.1814756
7: -3.2182529, -0.5745857, -3.3067012, -0.5109181, -2.4140382, 2.4832153
8: -6.9675674, -3.5079451, -7.0198917, -3.4194520, -2.4651871, 2.4454885
9: -5.5372486, -3.0319924, -5.6438284, -3.0212154, -2.0159800, 2.0926805

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415726, upper bound: 1.0515104
time: 8.75 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0415726, upper bound: 1.0421895
time: 5.97 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.41 seconds
IS_B2_B1_A1, status: Status.VERIFIED, split count: 3, time: 29.41
Output dim: 4, lower bound: -1.0497399, upper bound: 1.0497387
IS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.41
Output dim: 4, lower bound: -1.0497399, upper bound: 1.0516381
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.41
Output dim: 4, lower bound: -1.0415726, upper bound: 1.0515104
IS_B2_B2_A2, status: Status.VERIFIED, split count: 3, time: 29.41
Output dim: 4, lower bound: -1.0415726, upper bound: 1.0421895

## BFS IS instance: IS_B2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9512892, -5.3477535, -8.9520912, -5.3151517, -3.0530386, 3.0110993
1: -7.4166656, -4.1325512, -7.4083529, -4.1028433, -2.4669449, 2.4356978
2: -7.4985256, -4.5677285, -7.5371475, -4.5664854, -2.3331909, 2.3466737
3: -11.2696114, -7.7087088, -11.3139410, -7.7330952, -2.6670923, 2.7067647
4: 6.4912181, 8.8052502, 6.5195808, 8.8143463, -1.7232115, 1.7007527
5: -8.9070625, -5.8562832, -8.9155254, -5.9021778, -2.3014250, 2.3235266
6: -12.0257015, -8.2142506, -12.0140400, -8.1945267, -3.2297816, 3.2069178
7: -3.2640169, -0.5709376, -3.2581871, -0.5191908, -2.4293447, 2.4423194
8: -6.9751587, -3.4399574, -7.0045662, -3.4896240, -2.4176226, 2.4417393
9: -5.6070347, -3.0273271, -5.5702734, -3.0295439, -2.0546720, 2.0418005

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_B2_B1_A2_A1

### Relational analysis result of IS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396738, upper bound: 1.0515107
time: 9.46 seconds

## Relational analysis of IS_B2_B1_A2_A2

### Relational analysis result of IS_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0396738, upper bound: 1.0421870
time: 6.44 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.9197626, -5.3591385, -8.9693232, -5.3115182, -3.0120993, 3.0127339
1: -7.3834143, -4.1582766, -7.4311867, -4.0806074, -2.4339180, 2.4437952
2: -7.4752779, -4.5929008, -7.5572882, -4.5580583, -2.3149633, 2.3408439
3: -11.2590847, -7.7627273, -11.3203325, -7.6954956, -2.6911583, 2.6717646
4: 6.5971813, 8.8024006, 6.4462566, 8.8186750, -1.6377368, 1.7601868
5: -8.9024677, -5.9274864, -8.9212379, -5.8415532, -2.3331451, 2.2758660
6: -11.9990997, -8.2676992, -12.0289869, -8.1510868, -3.2267842, 3.1747427
7: -3.1996865, -0.5760386, -3.3056042, -0.5120972, -2.3929911, 2.4777336
8: -6.9664011, -3.5248725, -7.0198145, -3.4201684, -2.4602818, 2.4259729
9: -5.5144072, -3.0330462, -5.6425586, -3.0212951, -1.9904222, 2.0863254

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0396738, upper bound: 1.0496163
time: 8.07 seconds

## Relational analysis of IS_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396738, upper bound: 1.0515108
time: 8.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 30.95 seconds
IS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 30.95
Output dim: 4, lower bound: -1.0396738, upper bound: 1.0515107
IS_B2_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 30.95
Output dim: 4, lower bound: -1.0396738, upper bound: 1.0421870
IS_B2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 30.95
Output dim: 4, lower bound: -1.0396738, upper bound: 1.0496163
IS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 30.95
Output dim: 4, lower bound: -1.0396738, upper bound: 1.0515108

## BFS IS instance: IS_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -8.9356127, -5.3537412, -8.9511261, -5.3192239, -3.0180392, 2.9992838
1: -7.4022045, -4.1351681, -7.4077177, -4.1055975, -2.4501774, 2.4308579
2: -7.4946012, -4.5863709, -7.5366130, -4.5667968, -2.3272810, 2.3273156
3: -11.2653227, -7.7275467, -11.3118496, -7.7335501, -2.6609535, 2.6853931
4: 6.5267010, 8.8050461, 6.5208225, 8.8142204, -1.6851509, 1.6966782
5: -8.9050274, -5.8681030, -8.9148407, -5.9031558, -2.2868476, 2.3111815
6: -12.0097742, -8.2216482, -12.0138874, -8.1986427, -3.2069778, 3.1971729
7: -3.2451196, -0.5723906, -3.2570975, -0.5203701, -2.4084315, 2.4396062
8: -6.9740033, -3.4571512, -7.0044899, -3.4903989, -2.4153891, 2.4221880
9: -5.5836802, -3.0283809, -5.5690064, -3.0296235, -2.0288584, 2.0376682

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_B2_B1_A2_A1_A1

### Relational analysis result of IS_B2_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0392379, upper bound: 1.0481046
time: 8.58 seconds

## Relational analysis of IS_B2_B1_A2_A1_A2

### Relational analysis result of IS_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396723, upper bound: 1.0515122
time: 12.05 seconds

## BFS IS instance: IS_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.9362364, -5.3528647, -8.9697247, -5.3108335, -3.0297384, 3.0208039
1: -7.4028764, -4.1304970, -7.4316826, -4.0769320, -2.4602160, 2.4515324
2: -7.4986153, -4.5862231, -7.5603347, -4.5579767, -2.3410077, 2.3501647
3: -11.2656326, -7.7237730, -11.3205366, -7.6925926, -2.6981940, 2.6995158
4: 6.5197315, 8.8054256, 6.4407396, 8.8189449, -1.6962823, 1.7683594
5: -8.9051723, -5.8644085, -8.9213400, -5.8386331, -2.3377242, 2.3204606
6: -12.0111532, -8.2172947, -12.0297518, -8.1477976, -3.2432804, 3.2147274
7: -3.2469287, -0.5716588, -3.3068094, -0.5116186, -2.4201360, 2.4828303
8: -6.9750233, -3.4554715, -7.0205832, -3.4190922, -2.4364595, 2.4409192
9: -5.5903368, -3.0276990, -5.6477647, -3.0209398, -2.0434151, 2.0961800

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_B2_B2_A1_A2_B1

### Relational analysis result of IS_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0362614, upper bound: 1.0510844
time: 5.97 seconds

## Relational analysis of IS_B2_B2_A1_A2_B2

### Relational analysis result of IS_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396737, upper bound: 1.0515111
time: 5.94 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.46 seconds
IS_B2_B1_A2_A1_A1, status: Status.VERIFIED, split count: 5, time: 26.46
Output dim: 4, lower bound: -1.0392379, upper bound: 1.0481046
IS_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 26.46
Output dim: 4, lower bound: -1.0396723, upper bound: 1.0515122
IS_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 26.46
Output dim: 4, lower bound: -1.0362614, upper bound: 1.0510844
IS_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 26.46
Output dim: 4, lower bound: -1.0396737, upper bound: 1.0515111

## BFS IS instance: IS_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -8.9356108, -5.3537445, -8.9511261, -5.3192244, -3.0184207, 2.9992747
1: -7.4022050, -4.1351690, -7.4077215, -4.1055984, -2.4500914, 2.4336081
2: -7.4945984, -4.5863709, -7.5366125, -4.5667973, -2.3271570, 2.3259094
3: -11.2653179, -7.7275467, -11.3118477, -7.7335510, -2.6548400, 2.6824417
4: 6.5267015, 8.8050432, 6.5208235, 8.8142214, -1.6842160, 1.7006345
5: -8.9050293, -5.8681026, -8.9148407, -5.9031539, -2.2854729, 2.3105800
6: -12.0097752, -8.2216530, -12.0138884, -8.1986418, -3.2108064, 3.1968880
7: -3.2451184, -0.5723919, -3.2570970, -0.5203702, -2.4074788, 2.4396019
8: -6.9739995, -3.4571548, -7.0044880, -3.4904015, -2.4200516, 2.4215479
9: -5.5836797, -3.0283813, -5.5690055, -3.0296230, -2.0300775, 2.0376523

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 1684
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2136
type: B, layer: 3, pos: 2136
type: A, layer: 3, pos: 760
type: B, layer: 3, pos: 760
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 676
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 3112
type: B, layer: 3, pos: 3112
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1257
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: A, layer: 3, pos: 1486
type: B, layer: 3, pos: 1486
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2531
type: A, layer: 3, pos: 2130
type: B, layer: 3, pos: 759
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 759
type: A, layer: 3, pos: 1404
type: B, layer: 3, pos: 1404
type: B, layer: 3, pos: 1982
type: A, layer: 3, pos: 1982
type: A, layer: 3, pos: 572
type: B, layer: 3, pos: 572
type: A, layer: 3, pos: 170
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 418
type: B, layer: 3, pos: 418
type: B, layer: 3, pos: 1244
type: A, layer: 3, pos: 1244
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 907
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 176
type: A, layer: 3, pos: 176
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2237
type: B, layer: 3, pos: 2378
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 1685
type: B, layer: 3, pos: 2237
type: B, layer: 3, pos: 1685
type: B, layer: 3, pos: 1933
type: A, layer: 3, pos: 1933
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 1943
type: A, layer: 3, pos: 2336
type: B, layer: 3, pos: 1943
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1971
type: B, layer: 3, pos: 1971
type: B, layer: 3, pos: 2336
type: B, layer: 3, pos: 2328
type: A, layer: 3, pos: 2328
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 1839
type: B, layer: 3, pos: 416
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1247
type: B, layer: 3, pos: 1247
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2244
type: A, layer: 3, pos: 2244
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1459
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2922
type: B, layer: 3, pos: 2922
type: A, layer: 3, pos: 894
type: B, layer: 3, pos: 2390
type: B, layer: 3, pos: 894
type: A, layer: 3, pos: 2390
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: B, layer: 3, pos: 1802
type: A, layer: 3, pos: 1802
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 3105
type: B, layer: 3, pos: 2391
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1153
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1153
type: B, layer: 3, pos: 2852
type: A, layer: 3, pos: 2852
type: B, layer: 3, pos: 2608
type: A, layer: 3, pos: 2608
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 1253
type: B, layer: 3, pos: 1253
type: A, layer: 3, pos: 397
type: B, layer: 3, pos: 397
type: B, layer: 3, pos: 1449
type: A, layer: 3, pos: 1449
type: B, layer: 3, pos: 1778
type: A, layer: 3, pos: 1778

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 766

## Relational analysis of IS_B2_B1_A2_A1_A2_B1

### Relational analysis result of IS_B2_B1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0309427, upper bound: 1.0320413
time: 6.06 seconds

## Relational analysis of IS_B2_B1_A2_A1_A2_B2

### Relational analysis result of IS_B2_B1_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0309427, upper bound: 1.0428033
time: 9.78 seconds

## BFS IS instance: IS_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9341545, -5.3594189, -8.9624062, -5.3257284, -3.0122042, 3.0067124
1: -7.3997488, -4.1311865, -7.4241848, -4.0795851, -2.4519863, 2.4430521
2: -7.4932442, -4.5870996, -7.5476227, -4.5616555, -2.3325067, 2.3368642
3: -11.2564907, -7.7247443, -11.2997074, -7.6985550, -2.6829410, 2.6775868
4: 6.5203352, 8.8028383, 6.4431286, 8.8129892, -1.6909542, 1.7620512
5: -8.9035549, -5.8654451, -8.9175787, -5.8420377, -2.3315039, 2.3155932
6: -12.0087643, -8.2255640, -12.0201588, -8.1665993, -3.2224407, 3.1930101
7: -3.2453678, -0.5755365, -3.3018773, -0.5206711, -2.4089594, 2.4703531
8: -6.9722056, -3.4566884, -7.0140138, -3.4226832, -2.4349012, 2.4348738
9: -5.5896158, -3.0290971, -5.6453848, -3.0242438, -2.0394235, 2.0915685

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 2321
type: B, layer: 3, pos: 1684
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 2136
type: B, layer: 3, pos: 2136
type: A, layer: 3, pos: 760
type: B, layer: 3, pos: 760
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 3112
type: A, layer: 3, pos: 3112
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 2531
type: A, layer: 3, pos: 2531
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1486
type: A, layer: 3, pos: 1486
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 572
type: B, layer: 3, pos: 572
type: A, layer: 3, pos: 2130
type: B, layer: 3, pos: 2130
type: A, layer: 3, pos: 759
type: B, layer: 3, pos: 759
type: B, layer: 3, pos: 1404
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1982
type: B, layer: 3, pos: 170
type: B, layer: 3, pos: 1982
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 418
type: A, layer: 3, pos: 2528
type: B, layer: 3, pos: 418
type: A, layer: 3, pos: 1244
type: B, layer: 3, pos: 1244
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 907
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 176
type: A, layer: 3, pos: 2237
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 176
type: B, layer: 3, pos: 2237
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 1685
type: A, layer: 3, pos: 1685
type: A, layer: 3, pos: 1933
type: B, layer: 3, pos: 1933
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1943
type: A, layer: 3, pos: 1943
type: B, layer: 3, pos: 1846
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1971
type: A, layer: 3, pos: 1971
type: B, layer: 3, pos: 2328
type: A, layer: 3, pos: 2328
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 1247
type: B, layer: 3, pos: 1247
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2244
type: A, layer: 3, pos: 2244
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2922
type: B, layer: 3, pos: 2922
type: A, layer: 3, pos: 894
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 2390
type: A, layer: 3, pos: 2390
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: B, layer: 3, pos: 1802
type: A, layer: 3, pos: 1802
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 2391
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1153
type: B, layer: 3, pos: 1153
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2852
type: B, layer: 3, pos: 2852
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 2608
type: A, layer: 3, pos: 2608
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 397
type: B, layer: 3, pos: 397
type: A, layer: 3, pos: 1253
type: A, layer: 3, pos: 1449
type: B, layer: 3, pos: 1449
type: B, layer: 3, pos: 1778
type: A, layer: 3, pos: 1778

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of IS_B2_B2_A1_A2_B1_A1

### Relational analysis result of IS_B2_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0167943, upper bound: 1.0423594
time: 6.65 seconds

## Relational analysis of IS_B2_B2_A1_A2_B1_A2

### Relational analysis result of IS_B2_B2_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0274956, upper bound: 1.0423560
time: 6.70 seconds

## BFS IS instance: IS_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9362373, -5.3528643, -8.9697227, -5.3108368, -3.0297303, 3.0211866
1: -7.4028759, -4.1304970, -7.4316792, -4.0769320, -2.4629688, 2.4514952
2: -7.4986148, -4.5862241, -7.5603304, -4.5579777, -2.3410068, 2.3499670
3: -11.2656326, -7.7237740, -11.3205299, -7.6925945, -2.6981931, 2.6932502
4: 6.5197330, 8.8054256, 6.4407396, 8.8189411, -1.6933064, 1.7674246
5: -8.9051733, -5.8644109, -8.9213390, -5.8386345, -2.3371258, 2.3190536
6: -12.0111513, -8.2172966, -12.0297470, -8.1478014, -3.2429962, 3.2222404
7: -3.2469285, -0.5716596, -3.3068082, -0.5116208, -2.4200459, 2.4818957
8: -6.9750223, -3.4554715, -7.0205803, -3.4190936, -2.4364176, 2.4396679
9: -5.5903387, -3.0276985, -5.6477652, -3.0209394, -2.0433745, 2.0974069

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 2321
type: A, layer: 3, pos: 2321
type: B, layer: 3, pos: 1684
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 2333
type: A, layer: 3, pos: 2333
type: B, layer: 3, pos: 2136
type: A, layer: 3, pos: 2136
type: B, layer: 3, pos: 760
type: A, layer: 3, pos: 760
type: B, layer: 3, pos: 1395
type: A, layer: 3, pos: 1395
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 1789
type: A, layer: 3, pos: 1789
type: B, layer: 3, pos: 3112
type: A, layer: 3, pos: 3112
type: A, layer: 3, pos: 676
type: B, layer: 3, pos: 2531
type: A, layer: 3, pos: 2531
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 424
type: B, layer: 3, pos: 424
type: B, layer: 3, pos: 1486
type: A, layer: 3, pos: 1486
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 572
type: B, layer: 3, pos: 572
type: B, layer: 3, pos: 2528
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 2130
type: B, layer: 3, pos: 2130
type: A, layer: 3, pos: 759
type: B, layer: 3, pos: 759
type: B, layer: 3, pos: 1404
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1982
type: B, layer: 3, pos: 1982
type: B, layer: 3, pos: 170
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 418
type: B, layer: 3, pos: 418
type: A, layer: 3, pos: 1244
type: B, layer: 3, pos: 1244
type: A, layer: 3, pos: 1452
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1934
type: B, layer: 3, pos: 1934
type: B, layer: 3, pos: 403
type: A, layer: 3, pos: 403
type: B, layer: 3, pos: 907
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 2378
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 176
type: B, layer: 3, pos: 176
type: A, layer: 3, pos: 2237
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2237
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 1685
type: A, layer: 3, pos: 1685
type: A, layer: 3, pos: 1933
type: B, layer: 3, pos: 1933
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1992
type: B, layer: 3, pos: 1992
type: B, layer: 3, pos: 2336
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1943
type: B, layer: 3, pos: 1846
type: A, layer: 3, pos: 1943
type: A, layer: 3, pos: 1839
type: B, layer: 3, pos: 1839
type: A, layer: 3, pos: 1846
type: B, layer: 3, pos: 1971
type: A, layer: 3, pos: 1971
type: B, layer: 3, pos: 2328
type: A, layer: 3, pos: 2328
type: B, layer: 3, pos: 416
type: A, layer: 3, pos: 416
type: B, layer: 3, pos: 1247
type: A, layer: 3, pos: 1247
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2244
type: A, layer: 3, pos: 2244
type: B, layer: 3, pos: 1459
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2922
type: A, layer: 3, pos: 2922
type: A, layer: 3, pos: 894
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 2390
type: A, layer: 3, pos: 2390
type: A, layer: 3, pos: 1753
type: B, layer: 3, pos: 1753
type: B, layer: 3, pos: 1802
type: A, layer: 3, pos: 1802
type: B, layer: 3, pos: 3105
type: A, layer: 3, pos: 3105
type: B, layer: 3, pos: 2391
type: A, layer: 3, pos: 2216
type: B, layer: 3, pos: 2216
type: A, layer: 3, pos: 1248
type: B, layer: 3, pos: 1248
type: A, layer: 3, pos: 1153
type: B, layer: 3, pos: 1153
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 2852
type: B, layer: 3, pos: 2852
type: B, layer: 3, pos: 172
type: A, layer: 3, pos: 172
type: B, layer: 3, pos: 2608
type: A, layer: 3, pos: 2608
type: B, layer: 3, pos: 2867
type: A, layer: 3, pos: 2867
type: B, layer: 3, pos: 1253
type: A, layer: 3, pos: 1253
type: A, layer: 3, pos: 397
type: B, layer: 3, pos: 397
type: A, layer: 3, pos: 1449
type: B, layer: 3, pos: 1449
type: B, layer: 3, pos: 1778
type: A, layer: 3, pos: 1778

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of IS_B2_B2_A1_A2_B2_A1

### Relational analysis result of IS_B2_B2_A1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0202206, upper bound: 1.0428042
time: 10.76 seconds

## Relational analysis of IS_B2_B2_A1_A2_B2_A2

### Relational analysis result of IS_B2_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0309424, upper bound: 1.0428030
time: 8.68 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 34.07 seconds
IS_B2_B1_A2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 34.07
Output dim: 4, lower bound: -1.0309427, upper bound: 1.0320413
IS_B2_B1_A2_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 34.07
Output dim: 4, lower bound: -1.0309427, upper bound: 1.0428033
IS_B2_B2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 34.07
Output dim: 4, lower bound: -1.0167943, upper bound: 1.0423594
IS_B2_B2_A1_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 34.07
Output dim: 4, lower bound: -1.0274956, upper bound: 1.0423560
IS_B2_B2_A1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 34.07
Output dim: 4, lower bound: -1.0202206, upper bound: 1.0428042
IS_B2_B2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 34.07
Output dim: 4, lower bound: -1.0309424, upper bound: 1.0428030
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.6619789600372314
rel_dist={4: [-1.051660938425652, 1.051661396029865]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2011.12 seconds
