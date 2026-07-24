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
execution time: IAR + LP analysis = 13.56 + 33.19 = 46.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.7321513, upper bound: 1.7321501


# Binary Search by BASE starts (time budget: 3553.25 seconds, max iter: 100)

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
Binary search time: 193.63 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3359.62 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077453, upper bound: 1.4184900
time: 7.20 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184881, upper bound: 1.4184894
time: 7.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.64 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 14.64
Output dim: 4, lower bound: -1.4077453, upper bound: 1.4184900
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.64
Output dim: 4, lower bound: -1.4184881, upper bound: 1.4184894

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.9197674, -5.3591337, -8.9354134, -5.3531485, -3.3299432, 3.3405356
1: -7.3834295, -4.1582737, -7.3978786, -4.1556597, -2.7315741, 2.7439022
2: -7.4752836, -4.5928955, -7.4789820, -4.5742297, -2.5836210, 2.5727408
3: -11.2590923, -7.7627215, -11.2633400, -7.7441711, -2.9827385, 2.9700165
4: 6.5971775, 8.8024073, 6.5621042, 8.8026104, -1.8558023, 1.8905063
5: -8.9024849, -5.9274836, -8.9045181, -5.9158378, -2.6014929, 2.5986290
6: -11.9991140, -8.2676954, -12.0150757, -8.2602482, -3.5400667, 3.5510764
7: -3.1996956, -0.5760213, -3.2182775, -0.5745678, -2.6251278, 2.6422563
8: -6.9664278, -3.5248680, -6.9675961, -3.5078919, -2.7652025, 2.7491472
9: -5.5144129, -3.0330338, -5.5373082, -3.0319777, -2.1919842, 2.2139020

Time for backsubstitution: 12.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4077410
time: 9.38 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4184890
time: 5.21 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.9553642, -5.3070812, -8.9353971, -5.3531528, -3.3636150, 3.4124753
1: -7.4131517, -4.0979872, -7.3978720, -4.1556597, -2.7588444, 2.7963674
2: -7.5399389, -4.5638552, -7.4789791, -4.5742402, -2.6313014, 2.6088829
3: -11.3187714, -7.7308869, -11.2633362, -7.7441897, -3.0318427, 3.0030141
4: 6.5160456, 8.8161116, 6.5621233, 8.8026104, -1.9368997, 1.9048622
5: -8.9200726, -5.8995981, -8.9045162, -5.9158492, -2.6155267, 2.6395259
6: -12.0180368, -8.1864262, -12.0150566, -8.2602539, -3.5629945, 3.6226568
7: -3.2622232, -0.5131162, -3.2182701, -0.5745701, -2.6876531, 2.7051539
8: -7.0121460, -3.4867215, -6.9675951, -3.5079165, -2.8103390, 2.7942522
9: -5.5743561, -3.0260987, -5.5372796, -3.0319781, -2.2545891, 2.2204754

Time for backsubstitution: 12.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184892, upper bound: 1.4077410
time: 6.42 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184892, upper bound: 1.4184889
time: 5.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.07 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.07
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4077410
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.07
Output dim: 4, lower bound: -1.4077410, upper bound: 1.4184890
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.07
Output dim: 4, lower bound: -1.4184892, upper bound: 1.4077410
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.07
Output dim: 4, lower bound: -1.4184892, upper bound: 1.4184889

## BFS IS instance: IS_A1_B1

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

Time for backsubstitution: 12.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077334, upper bound: 1.4063801
time: 5.28 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077331, upper bound: 1.4077304
time: 5.09 seconds

## BFS IS instance: IS_A1_B2

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

Time for backsubstitution: 12.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077334, upper bound: 1.4171031
time: 5.18 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4077331, upper bound: 1.4184756
time: 5.11 seconds

## BFS IS instance: IS_A2_B1

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

Time for backsubstitution: 12.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184748, upper bound: 1.4063801
time: 4.32 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184745, upper bound: 1.4077304
time: 4.57 seconds

## BFS IS instance: IS_A2_B2

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

Time for backsubstitution: 12.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184759, upper bound: 1.4105820
time: 4.78 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4184756, upper bound: 1.4121008
time: 4.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.87 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 4, lower bound: -1.4077334, upper bound: 1.4063801
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 4, lower bound: -1.4077331, upper bound: 1.4077304
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 4, lower bound: -1.4077334, upper bound: 1.4171031
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 4, lower bound: -1.4077331, upper bound: 1.4184756
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 4, lower bound: -1.4184748, upper bound: 1.4063801
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 4, lower bound: -1.4184745, upper bound: 1.4077304
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 4, lower bound: -1.4184759, upper bound: 1.4105820
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 4, lower bound: -1.4184756, upper bound: 1.4121008

## BFS IS instance: IS_A1_B1_A1

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

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4063824
time: 6.43 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4063828
time: 5.90 seconds

## BFS IS instance: IS_A1_B1_A2

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

Time for backsubstitution: 12.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4077350
time: 5.08 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4077329
time: 6.59 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9553642, -5.3070812, -3.3670826, 3.3493567
1: -7.3795280, -4.1591473, -7.4131517, -4.0979872, -2.7774372, 2.7541280
2: -7.4732456, -4.5950751, -7.5399389, -4.5638552, -2.5906482, 2.6102247
3: -11.2570133, -7.7643375, -11.3187714, -7.7308869, -2.9908562, 3.0106883
4: 6.5990024, 8.8007011, 6.5160456, 8.8161116, -1.8648586, 1.9286170
5: -8.8986826, -5.9286919, -8.9200726, -5.8995981, -2.6134052, 2.6017299
6: -11.9953022, -8.2697430, -12.0180368, -8.1864262, -3.6009612, 3.5533419
7: -3.1971812, -0.5803651, -3.2622232, -0.5131162, -2.6840649, 2.6818581
8: -6.9589167, -3.5266910, -7.0121460, -3.4867215, -2.7757673, 2.7886729
9: -5.5121059, -3.0363898, -5.5743561, -3.0260987, -2.1928182, 2.2445736

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4171005
time: 5.22 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4171004
time: 5.95 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9362144, -5.3529425, -8.9553623, -5.3070850, -3.3871861, 3.3603806
1: -7.4028411, -4.1309299, -7.4131451, -4.0979881, -2.8105438, 2.7801416
2: -7.4982338, -4.5862241, -7.5399365, -4.5638580, -2.6182389, 2.6201856
3: -11.2656307, -7.7241049, -11.3187675, -7.7308884, -3.0031071, 3.0398297
4: 6.5204015, 8.8054256, 6.5160475, 8.8161087, -1.9278846, 1.9364870
5: -8.9051714, -5.8647685, -8.9200649, -5.8995991, -2.6230111, 2.6547835
6: -12.0111465, -8.2177105, -12.0180311, -8.1864281, -3.6212592, 3.6044769
7: -3.2468393, -0.5716875, -3.2622199, -0.5131224, -2.7207289, 2.6905324
8: -6.9749389, -3.4555233, -7.0121317, -3.4867249, -2.8041525, 2.8124001
9: -5.5897975, -3.0277028, -5.5743518, -3.0261054, -2.2561965, 2.2574430

Time for backsubstitution: 12.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4184763
time: 4.46 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4184763
time: 4.57 seconds

## BFS IS instance: IS_A2_B1_A1

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

Time for backsubstitution: 12.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4063827
time: 6.27 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4063827
time: 6.24 seconds

## BFS IS instance: IS_A2_B1_A2

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

Time for backsubstitution: 12.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4077349
time: 4.49 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4077350
time: 5.01 seconds

## BFS IS instance: IS_A2_B2_A1

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

Time for backsubstitution: 12.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4105798
time: 7.61 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4105801
time: 7.04 seconds

## BFS IS instance: IS_A2_B2_A2

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

Time for backsubstitution: 12.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4121007
time: 4.70 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4121008
time: 5.61 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.69 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4063824
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4063828
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4077350
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4077329
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4171005
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4171004
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4184763
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4063827, upper bound: 1.4184763
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4063827
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4063827
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4077349
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4077350
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4105798
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4105801
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4121007
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.69
Output dim: 4, lower bound: -1.4171003, upper bound: 1.4121008

## BFS IS instance: IS_A1_B1_A1_B1

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

Time for backsubstitution: 13.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062790, upper bound: 1.3996542
time: 4.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4062800
time: 4.65 seconds

## BFS IS instance: IS_A1_B1_A1_B2

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

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062790, upper bound: 1.3996527
time: 6.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4062784
time: 6.38 seconds

## BFS IS instance: IS_A1_B1_A2_B1

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

Time for backsubstitution: 12.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4010365
time: 4.23 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4076078
time: 6.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2

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

Time for backsubstitution: 12.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4010347
time: 6.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4076077
time: 5.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1

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

Time for backsubstitution: 12.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4103557
time: 4.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4170024
time: 4.28 seconds

## BFS IS instance: IS_A1_B2_A1_B2

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

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4103538
time: 7.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4170008
time: 6.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1

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

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4117549
time: 4.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4183543
time: 6.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2

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

Time for backsubstitution: 12.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4117525
time: 6.15 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4183542
time: 6.42 seconds

## BFS IS instance: IS_A2_B1_A1_B1

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

Time for backsubstitution: 12.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.3996542
time: 5.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4062780
time: 8.67 seconds

## BFS IS instance: IS_A2_B1_A1_B2

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

Time for backsubstitution: 12.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.3996542
time: 4.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4062803
time: 4.34 seconds

## BFS IS instance: IS_A2_B1_A2_B1

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

Time for backsubstitution: 12.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4010365
time: 3.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4076073
time: 6.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2

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

Time for backsubstitution: 13.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4010343
time: 6.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4076074
time: 5.46 seconds

## BFS IS instance: IS_A2_B2_A1_B1

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

Time for backsubstitution: 12.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4037824
time: 5.27 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4105019
time: 9.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2

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

Time for backsubstitution: 13.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4037824
time: 4.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4105034
time: 4.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1

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

Time for backsubstitution: 12.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4052609
time: 3.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4120166
time: 7.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9719906, -5.3008099, -8.9719906, -5.3008099, -3.4498196, 3.4498196
1: -7.4332638, -4.0701799, -7.4332638, -4.0701799, -2.8403730, 2.8403730
2: -7.5616016, -4.5572181, -7.5616016, -4.5572181, -2.6567774, 2.6567776
3: -11.3253918, -7.6914606, -11.3253918, -7.6914606, -3.0790334, 3.0790339
4: 6.4376669, 8.8191299, 6.4376669, 8.8191299, -2.0150473, 2.0150471
5: -8.9227667, -5.8363128, -8.9227667, -5.8363128, -2.6936204, 2.6936204
6: -12.0300884, -8.1377153, -12.0300884, -8.1377153, -3.6735106, 3.6735113
7: -3.3095057, -0.5087115, -3.3095057, -0.5087115, -2.8007941, 2.8007941
8: -7.0207233, -3.4173274, -7.0207233, -3.4173274, -2.8336535, 2.8336532
9: -5.6508780, -3.0207691, -5.6508780, -3.0207691, -2.3253345, 2.3253348

Time for backsubstitution: 12.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4052607
time: 6.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4120172
time: 6.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.36 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062790, upper bound: 1.3996542
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4062800
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062790, upper bound: 1.3996527
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4062784
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4010365
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4076078
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4010347
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4076077
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4103557
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4170024
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4103538
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4170008
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4117549
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4183543
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4117525
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4183542
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170013, upper bound: 1.3996542
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4062780
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170013, upper bound: 1.3996542
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4062803
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4010365
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4076073
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4010343
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4076074
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4037824
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4105019
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4037824
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4105034
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4052609
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4120166
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4052607
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4120172

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9103146, -5.3761315, -8.9166660, -5.3643513, -3.3020344, 3.2956436
1: -7.3720083, -4.1617994, -7.3780484, -4.1594706, -2.7147236, 2.7167211
2: -7.4605055, -4.5987673, -7.4707003, -4.5954852, -2.5477686, 2.5550988
3: -11.2361774, -7.7703094, -11.2527008, -7.7647934, -2.9375687, 2.9485784
4: 6.6014686, 8.7947397, 6.5992856, 8.7994747, -1.8483934, 1.8470013
5: -8.8949356, -5.9321175, -8.8979168, -5.9291787, -2.5791807, 2.5794318
6: -11.9857388, -8.2885303, -11.9941864, -8.2736530, -3.5123796, 3.5080953
7: -3.1921513, -0.5893819, -3.1964273, -0.5821950, -2.6099563, 2.6070454
8: -6.9523339, -3.5303407, -6.9575777, -3.5272684, -2.7326951, 2.7348104
9: -5.5097089, -3.0397077, -5.5117655, -3.0370522, -2.1790745, 2.1790564

Time for backsubstitution: 13.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3996532, upper bound: 1.3996549
time: 4.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.3996532, upper bound: 1.3996528
time: 5.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 23.14 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.14
Output dim: 4, lower bound: -1.3996532, upper bound: 1.3996549
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.14
Output dim: 4, lower bound: -1.3996532, upper bound: 1.3996528
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4062800
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062790, upper bound: 1.3996527
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4062784
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4010365
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4076078
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4010347
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4076077
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4103557
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4170024
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4103538
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062790, upper bound: 1.4170008
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4117549
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4183543
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4117525
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4062786, upper bound: 1.4183542
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170013, upper bound: 1.3996542
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4062780
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170013, upper bound: 1.3996542
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4062803
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4010365
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4076073
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4010343
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4076074
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4037824
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4105019
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4037824
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170013, upper bound: 1.4105034
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4052609
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4120166
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4052607
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.14
Output dim: 4, lower bound: -1.4170010, upper bound: 1.4120172
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.8940565586090088
rel_dist={4: [-1.418532051192658, 1.418531738496264]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1476185, upper bound: 1.1597273
time: 7.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597256, upper bound: 1.1597297
time: 5.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.44 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.44
Output dim: 4, lower bound: -1.1476185, upper bound: 1.1597273
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.44
Output dim: 4, lower bound: -1.1597256, upper bound: 1.1597297

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.9197674, -5.3591337, -8.9354134, -5.3531485, -3.0642056, 3.0747976
1: -7.3834295, -4.1582737, -7.3978786, -4.1556597, -2.4792252, 2.4915528
2: -7.4752836, -4.5928955, -7.4789820, -4.5742297, -2.3705935, 2.3597136
3: -11.2590923, -7.7627215, -11.2633400, -7.7441711, -2.7293205, 2.7165992
4: 6.5971775, 8.8024073, 6.5621042, 8.8026104, -1.6817443, 1.7164483
5: -8.9024849, -5.9274836, -8.9045181, -5.9158378, -2.3545351, 2.3516712
6: -11.9991140, -8.2676954, -12.0150757, -8.2602482, -3.2455177, 3.2565279
7: -3.1996956, -0.5760213, -3.2182775, -0.5745678, -2.4413095, 2.4589639
8: -6.9664278, -3.5248680, -6.9675961, -3.5078919, -2.4858580, 2.4698024
9: -5.5144129, -3.0330338, -5.5373082, -3.0319777, -2.0342050, 2.0561228

Time for backsubstitution: 12.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1476161, upper bound: 1.1476155
time: 7.66 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1476161, upper bound: 1.1597265
time: 7.02 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.9547949, -5.3097715, -8.9353848, -5.3531547, -3.0973473, 3.1348925
1: -7.4127445, -4.0997839, -7.3978682, -4.1556625, -2.5058756, 2.5345151
2: -7.5395980, -4.5640569, -7.4789782, -4.5742464, -2.4142966, 2.3873639
3: -11.3175611, -7.7311797, -11.2633362, -7.7442031, -2.7724352, 2.7464659
4: 6.5168567, 8.8161001, 6.5621371, 8.8026114, -1.7592969, 1.7306523
5: -8.9197712, -5.9002419, -8.9045162, -5.9158564, -2.3683934, 2.3858938
6: -12.0179567, -8.1891632, -12.0150452, -8.2602558, -3.2671871, 3.3222859
7: -3.2615128, -0.5138963, -3.2182646, -0.5745702, -2.5195689, 2.4874604
8: -7.0121222, -3.4872270, -6.9675961, -3.5079317, -2.5304098, 2.5072732
9: -5.5735240, -3.0261359, -5.5372639, -3.0319772, -2.0932913, 2.0626545

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597270, upper bound: 1.1476169
time: 4.49 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597270, upper bound: 1.1597280
time: 4.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.66 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.66
Output dim: 4, lower bound: -1.1476161, upper bound: 1.1476155
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.66
Output dim: 4, lower bound: -1.1476161, upper bound: 1.1597265
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.66
Output dim: 4, lower bound: -1.1597270, upper bound: 1.1476169
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.66
Output dim: 4, lower bound: -1.1597270, upper bound: 1.1597280

## BFS IS instance: IS_A1_B1

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

Time for backsubstitution: 12.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1476004, upper bound: 1.1457693
time: 6.26 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1476004, upper bound: 1.1475976
time: 4.99 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.9197674, -5.3591337, -8.9539986, -5.3135123, -3.0978603, 3.0857098
1: -7.3834295, -4.1582737, -7.4121752, -4.1022892, -2.5179832, 2.5032427
2: -7.4752836, -4.5928955, -7.5391188, -4.5643411, -2.3790073, 2.3949857
3: -11.2590923, -7.7627215, -11.3158779, -7.7315907, -2.7394896, 2.7514040
4: 6.5971775, 8.8024073, 6.5179892, 8.8160839, -1.6922708, 1.7544322
5: -8.9024849, -5.9274836, -8.9193459, -5.9011364, -2.3690677, 2.3558311
6: -11.9991140, -8.2676954, -12.0178480, -8.1929493, -3.2999167, 3.2601190
7: -3.1996956, -0.5760213, -3.2605205, -0.5149789, -2.4665294, 2.5157518
8: -6.9664278, -3.5248680, -7.0120916, -3.4879344, -2.5027351, 2.5109205
9: -5.5144129, -3.0330338, -5.5723653, -3.0261884, -2.0371232, 2.0883701

Time for backsubstitution: 12.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1476004, upper bound: 1.1579159
time: 6.65 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1476004, upper bound: 1.1597081
time: 5.00 seconds

## BFS IS instance: IS_A2_B1

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

Time for backsubstitution: 12.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597069, upper bound: 1.1457718
time: 5.75 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597069, upper bound: 1.1475982
time: 5.51 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.9553642, -5.3070812, -8.9553642, -5.3070812, -3.1545916, 3.1545913
1: -7.4131517, -4.0979872, -7.4131517, -4.0979872, -2.5483713, 2.5483713
2: -7.5399389, -4.5638552, -7.5399389, -4.5638552, -2.4015670, 2.4015665
3: -11.3187714, -7.7308869, -11.3187714, -7.7308869, -2.7845254, 2.7845249
4: 6.5160456, 8.8161116, 6.5160456, 8.8161116, -1.7610738, 1.7610741
5: -8.9200726, -5.8995981, -8.9200726, -5.8995981, -2.3921976, 2.3921981
6: -12.0180368, -8.1864262, -12.0180368, -8.1864262, -3.3295484, 3.3295481
7: -3.2622232, -0.5131162, -3.2622232, -0.5131162, -2.5493178, 2.5493181
8: -7.0121460, -3.4867215, -7.0121460, -3.4867215, -2.5097528, 2.5097528
9: -5.5743561, -3.0260987, -5.5743561, -3.0260987, -2.0952413, 2.0952411

Time for backsubstitution: 12.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597083, upper bound: 1.1464421
time: 5.95 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1597083, upper bound: 1.1482933
time: 5.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.17 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 4, lower bound: -1.1476004, upper bound: 1.1457693
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 4, lower bound: -1.1476004, upper bound: 1.1475976
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 4, lower bound: -1.1476004, upper bound: 1.1579159
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 4, lower bound: -1.1476004, upper bound: 1.1597081
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 4, lower bound: -1.1597069, upper bound: 1.1457718
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 4, lower bound: -1.1597069, upper bound: 1.1475982
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 4, lower bound: -1.1597083, upper bound: 1.1464421
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 4, lower bound: -1.1597083, upper bound: 1.1482933

## BFS IS instance: IS_A1_B1_A1

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

Time for backsubstitution: 12.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1457712
time: 5.81 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1457746
time: 9.64 seconds

## BFS IS instance: IS_A1_B1_A2

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

Time for backsubstitution: 12.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1475997
time: 6.44 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1476032
time: 6.93 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9177856, -5.3612566, -8.9539986, -5.3135123, -3.0944767, 3.0823715
1: -7.3795280, -4.1591473, -7.4121752, -4.1022892, -2.5135524, 2.5010092
2: -7.4732456, -4.5950751, -7.5391188, -4.5643411, -2.3772211, 2.3923874
3: -11.2570133, -7.7643375, -11.3158779, -7.7315907, -2.7368431, 2.7494872
4: 6.5990024, 8.8007011, 6.5179892, 8.8160839, -1.6904898, 1.7525861
5: -8.8986826, -5.9286919, -8.9193459, -5.9011364, -2.3649020, 2.3543644
6: -11.9953022, -8.2697430, -12.0178480, -8.1929493, -3.2960334, 3.2585015
7: -3.1971812, -0.5803651, -3.2605205, -0.5149789, -2.4637537, 2.5105672
8: -6.9589167, -3.5266910, -7.0120916, -3.4879344, -2.4951458, 2.5086138
9: -5.5121059, -3.0363898, -5.5723653, -3.0261884, -2.0349731, 2.0848193

Time for backsubstitution: 12.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457718, upper bound: 1.1579133
time: 5.99 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457718, upper bound: 1.1579143
time: 5.44 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9361038, -5.3533535, -8.9539948, -5.3135161, -3.1143818, 3.0927751
1: -7.4026504, -4.1331730, -7.4121656, -4.1022882, -2.5434332, 2.5252116
2: -7.4963140, -4.5862331, -7.5391140, -4.5643463, -2.4029460, 2.4023125
3: -11.2655725, -7.7257738, -11.3158703, -7.7315955, -2.7484121, 2.7772996
4: 6.5237598, 8.8053169, 6.5179915, 8.8160782, -1.7483516, 1.7589281
5: -8.9051266, -5.8665733, -8.9193344, -5.9011378, -2.3725691, 2.4009304
6: -12.0110626, -8.2197924, -12.0178337, -8.1929531, -3.3142681, 3.3051579
7: -3.2463923, -0.5718788, -3.2605138, -0.5149934, -2.4922643, 2.5185370
8: -6.9745007, -3.4557991, -7.0120697, -3.4879379, -2.5157952, 2.5262063
9: -5.5868039, -3.0277123, -5.5723600, -3.0261984, -2.0896347, 2.0954258

Time for backsubstitution: 13.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457718, upper bound: 1.1597057
time: 7.08 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457718, upper bound: 1.1597095
time: 5.02 seconds

## BFS IS instance: IS_A2_B1_A1

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

Time for backsubstitution: 12.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1457742
time: 5.23 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1457742
time: 5.13 seconds

## BFS IS instance: IS_A2_B1_A2

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

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1476006
time: 6.97 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1475996
time: 4.83 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.9533844, -5.3091950, -8.9553642, -5.3070812, -3.1512127, 3.1512582
1: -7.4092550, -4.0988493, -7.4131517, -4.0979872, -2.5439596, 2.5461307
2: -7.5379148, -4.5660362, -7.5399389, -4.5638552, -2.3998427, 2.3989987
3: -11.3167067, -7.7324371, -11.3187714, -7.7308869, -2.7818627, 2.7825987
4: 6.5177813, 8.8144073, 6.5160456, 8.8161116, -1.7593622, 1.7592278
5: -8.9162674, -5.9007535, -8.9200726, -5.8995981, -2.3880358, 2.3907280
6: -12.0142260, -8.1884604, -12.0180368, -8.1864262, -3.3256650, 3.3279910
7: -3.2597790, -0.5174644, -3.2622232, -0.5131162, -2.5466423, 2.5441465
8: -7.0046282, -3.4885011, -7.0121460, -3.4867215, -2.5021625, 2.5074897
9: -5.5721159, -3.0294533, -5.5743561, -3.0260987, -2.0930126, 2.0916901

Time for backsubstitution: 12.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1464420
time: 5.28 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1464388
time: 5.00 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.9718456, -5.3012981, -8.9553585, -5.3070850, -3.1712103, 3.1616673
1: -7.4330049, -4.0728636, -7.4131432, -4.0979881, -2.5743756, 2.5536292
2: -7.5593634, -4.5572271, -7.5399342, -4.5638604, -2.4260397, 2.4088752
3: -11.3253317, -7.6934857, -11.3187666, -7.7308893, -2.7932229, 2.8108988
4: 6.4417410, 8.8190193, 6.5160499, 8.8161058, -1.8313346, 1.7655718
5: -8.9227209, -5.8384805, -8.9200592, -5.8996010, -2.3956966, 2.4372416
6: -12.0299988, -8.1401215, -12.0180244, -8.1864290, -3.3439074, 3.3607378
7: -3.3089371, -0.5089326, -3.2622163, -0.5131278, -2.5763841, 2.5521696
8: -7.0201998, -3.4177027, -7.0121231, -3.4867258, -2.5228386, 2.5663052
9: -5.6472859, -3.0207810, -5.5743508, -3.0261102, -2.1524520, 2.1023054

Time for backsubstitution: 12.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1482960
time: 5.43 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1482960
time: 5.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 23.24 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1457712
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1457746
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1475997
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1457722, upper bound: 1.1476032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1457718, upper bound: 1.1579133
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1457718, upper bound: 1.1579143
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1457718, upper bound: 1.1597057
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1457718, upper bound: 1.1597095
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1457742
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1457742
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1476006
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1475996
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1464420
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1464388
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1482960
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.24
Output dim: 4, lower bound: -1.1579143, upper bound: 1.1482960

## BFS IS instance: IS_A1_B1_A1_B1

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

Time for backsubstitution: 12.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455207, upper bound: 1.1416134
time: 5.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457707, upper bound: 1.1457710
time: 6.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2

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

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455207, upper bound: 1.1416155
time: 5.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457707, upper bound: 1.1457696
time: 6.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1

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

Time for backsubstitution: 12.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455204, upper bound: 1.1434570
time: 7.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457704, upper bound: 1.1476008
time: 5.38 seconds

## BFS IS instance: IS_A1_B1_A2_B2

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

Time for backsubstitution: 12.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455204, upper bound: 1.1434590
time: 6.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457704, upper bound: 1.1475989
time: 6.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1

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

Time for backsubstitution: 12.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455203, upper bound: 1.1538034
time: 8.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457703, upper bound: 1.1579121
time: 6.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2

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

Time for backsubstitution: 12.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455203, upper bound: 1.1538027
time: 6.45 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457703, upper bound: 1.1579127
time: 5.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1

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

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455200, upper bound: 1.1556082
time: 6.30 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457700, upper bound: 1.1597068
time: 6.26 seconds

## BFS IS instance: IS_A1_B2_A2_B2

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

Time for backsubstitution: 13.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1455200, upper bound: 1.1556059
time: 8.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1457700, upper bound: 1.1597072
time: 5.67 seconds

## BFS IS instance: IS_A2_B1_A1_B1

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

Time for backsubstitution: 13.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1576957, upper bound: 1.1416135
time: 6.08 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579123, upper bound: 1.1457722
time: 5.52 seconds

## BFS IS instance: IS_A2_B1_A1_B2

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

Time for backsubstitution: 13.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1576957, upper bound: 1.1416151
time: 4.90 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579123, upper bound: 1.1457690
time: 4.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1

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

Time for backsubstitution: 13.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1576956, upper bound: 1.1434565
time: 5.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1476004
time: 6.31 seconds

## BFS IS instance: IS_A2_B1_A2_B2

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

Time for backsubstitution: 13.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1576956, upper bound: 1.1434583
time: 5.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1476007
time: 5.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1

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

Time for backsubstitution: 13.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1576970, upper bound: 1.1422788
time: 4.95 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579135, upper bound: 1.1464401
time: 7.11 seconds

## BFS IS instance: IS_A2_B2_A1_B2

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

Time for backsubstitution: 13.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1576970, upper bound: 1.1422820
time: 5.06 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579135, upper bound: 1.1464368
time: 4.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1

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

Time for backsubstitution: 13.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1576956, upper bound: 1.1441344
time: 6.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1482937
time: 5.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9719906, -5.3008099, -8.9719906, -5.3008099, -3.1789198, 3.1789196
1: -7.4332638, -4.0701799, -7.4332638, -4.0701799, -2.5799813, 2.5799813
2: -7.5616016, -4.5572181, -7.5616016, -4.5572181, -2.4354019, 2.4354019
3: -11.3253918, -7.6914606, -11.3253918, -7.6914606, -2.8209209, 2.8209214
4: 6.4376669, 8.8191299, 6.4376669, 8.8191299, -1.8377008, 1.8377008
5: -8.9227667, -5.8363128, -8.9227667, -5.8363128, -2.4412735, 2.4412732
6: -12.0300884, -8.1377153, -12.0300884, -8.1377153, -3.3764143, 3.3764150
7: -3.3095057, -0.5087115, -3.3095057, -0.5087115, -2.5804276, 2.5804276
8: -7.0207233, -3.4173274, -7.0207233, -3.4173274, -2.5398674, 2.5398674
9: -5.6508780, -3.0207691, -5.6508780, -3.0207691, -2.1610918, 2.1610918

Time for backsubstitution: 13.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1576969, upper bound: 1.1441367
time: 5.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1482940
time: 5.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.47 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1455207, upper bound: 1.1416134
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1457707, upper bound: 1.1457710
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1455207, upper bound: 1.1416155
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1457707, upper bound: 1.1457696
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1455204, upper bound: 1.1434570
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1457704, upper bound: 1.1476008
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1455204, upper bound: 1.1434590
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1457704, upper bound: 1.1475989
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1455203, upper bound: 1.1538034
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1457703, upper bound: 1.1579121
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1455203, upper bound: 1.1538027
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1457703, upper bound: 1.1579127
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1455200, upper bound: 1.1556082
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1457700, upper bound: 1.1597068
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1455200, upper bound: 1.1556059
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1457700, upper bound: 1.1597072
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1576957, upper bound: 1.1416135
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1579123, upper bound: 1.1457722
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1576957, upper bound: 1.1416151
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1579123, upper bound: 1.1457690
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1576956, upper bound: 1.1434565
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1476004
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1576956, upper bound: 1.1434583
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1476007
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1576970, upper bound: 1.1422788
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1579135, upper bound: 1.1464401
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1576970, upper bound: 1.1422820
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1579135, upper bound: 1.1464368
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1576956, upper bound: 1.1441344
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1482937
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1576969, upper bound: 1.1441367
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.47
Output dim: 4, lower bound: -1.1579122, upper bound: 1.1482940
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.7199985980987549
rel_dist={4: [-1.1597516071645249, 1.1597543492710418]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415920, upper bound: 1.0515316
time: 4.91 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0516527, upper bound: 1.0516546
time: 6.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.60
Output dim: 4, lower bound: -1.0415920, upper bound: 1.0515316
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.60
Output dim: 4, lower bound: -1.0516527, upper bound: 1.0516546

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.9197674, -5.3591337, -8.9325085, -5.3542285, -2.9736247, 2.9822125
1: -7.3834295, -4.1582737, -7.3952055, -4.1561289, -2.3947167, 2.4047508
2: -7.4752836, -4.5928955, -7.4782953, -4.5776968, -2.2961631, 2.2872760
3: -11.2590923, -7.7627215, -11.2625790, -7.7476139, -2.6412954, 2.6309502
4: 6.5971775, 8.8024073, 6.5686035, 8.8025713, -1.6230614, 1.6513441
5: -8.9024849, -5.9274836, -8.9041462, -5.9180174, -2.2699537, 2.2675886
6: -11.9991140, -8.2676954, -12.0121126, -8.2615852, -3.1460962, 3.1550283
7: -3.1996956, -0.5760213, -3.2148366, -0.5748340, -2.3665953, 2.3809812
8: -6.9664278, -3.5248680, -6.9673834, -3.5110519, -2.3891392, 2.3760660
9: -5.5144129, -3.0330338, -5.5330510, -3.0321689, -1.9809415, 1.9987974

Time for backsubstitution: 13.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0415904, upper bound: 1.0415930
time: 6.13 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415904, upper bound: 1.0515316
time: 6.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -8.9540720, -5.3130360, -8.9353790, -5.3531547, -3.0080986, 3.0405302
1: -7.4122486, -4.1019783, -7.3978667, -4.1556625, -2.4212666, 2.4459319
2: -7.5391769, -4.5643058, -7.4789758, -4.5742502, -2.3414292, 2.3133645
3: -11.3160095, -7.7315402, -11.2633352, -7.7442079, -2.6848221, 2.6607270
4: 6.5178452, 8.8160505, 6.5621452, 8.8026114, -1.6993642, 1.6724560
5: -8.9193325, -5.9010239, -8.9045143, -5.9158607, -2.2858090, 2.3005276
6: -12.0178518, -8.1924686, -12.0150394, -8.2602577, -3.1684680, 3.2192674
7: -3.2606454, -0.5148420, -3.2182608, -0.5745707, -2.4425821, 2.4113913
8: -7.0120811, -3.4878447, -6.9675961, -3.5079393, -2.4350729, 2.4111412
9: -5.5725117, -3.0261893, -5.5372553, -3.0319786, -2.0387964, 2.0100100

Time for backsubstitution: 14.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0515290, upper bound: 1.0415932
time: 5.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0515290, upper bound: 1.0415903
time: 7.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.74 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 27.74
Output dim: 4, lower bound: -1.0415904, upper bound: 1.0415930
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.74
Output dim: 4, lower bound: -1.0415904, upper bound: 1.0515316
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.74
Output dim: 4, lower bound: -1.0515290, upper bound: 1.0415932
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.74
Output dim: 4, lower bound: -1.0515290, upper bound: 1.0415903

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.9197674, -5.3591337, -8.9531059, -5.3171096, -3.0054240, 2.9962959
1: -7.3834295, -4.1582737, -7.4116178, -4.1047344, -2.4291940, 2.4186816
2: -7.4752836, -4.5928955, -7.5386443, -4.5646186, -2.3077283, 2.3220356
3: -11.2590923, -7.7627215, -11.3139153, -7.7319946, -2.6546736, 2.6636152
4: 6.5971775, 8.8024073, 6.5190821, 8.8159266, -1.6340468, 1.6952913
5: -8.9024849, -5.9274836, -8.9186459, -5.9019957, -2.2858839, 2.2730541
6: -11.9991140, -8.2676954, -12.0176983, -8.1965799, -3.1964703, 3.1617360
7: -3.1996956, -0.5760213, -3.2595558, -0.5160191, -2.3903446, 2.4398713
8: -6.9664278, -3.5248680, -7.0120049, -3.4886155, -2.4088993, 2.4155579
9: -5.5144129, -3.0330338, -5.5712409, -3.0262709, -1.9844494, 2.0346634

Time for backsubstitution: 13.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0415746, upper bound: 1.0496181
time: 7.26 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0415746, upper bound: 1.0515109
time: 9.20 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -8.9531059, -5.3171096, -8.9197674, -5.3591337, -2.9962959, 3.0054240
1: -7.4116178, -4.1047344, -7.3834295, -4.1582737, -2.4186816, 2.4291940
2: -7.5386443, -4.5646186, -7.4752836, -4.5928955, -2.3220353, 2.3077283
3: -11.3139153, -7.7319946, -11.2590923, -7.7627215, -2.6636152, 2.6546738
4: 6.5190821, 8.8159266, 6.5971775, 8.8024073, -1.6952913, 1.6340470
5: -8.9186459, -5.9019957, -8.9024849, -5.9274836, -2.2730536, 2.2858844
6: -12.0176983, -8.1965799, -11.9991140, -8.2676954, -3.1617355, 3.1964703
7: -3.2595558, -0.5160191, -3.1996956, -0.5760213, -2.4398713, 2.3903444
8: -7.0120049, -3.4886155, -6.9664278, -3.5248680, -2.4155579, 2.4088995
9: -5.5712409, -3.0262709, -5.5144129, -3.0330338, -2.0346634, 1.9844496

Time for backsubstitution: 13.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0515101, upper bound: 1.0396744
time: 6.78 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0515101, upper bound: 1.0415755
time: 5.82 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -8.9553642, -5.3070812, -8.9553642, -5.3070812, -3.0630493, 3.0630493
1: -7.4131517, -4.0979872, -7.4131517, -4.0979872, -2.4614935, 2.4614935
2: -7.5399389, -4.5638552, -7.5399389, -4.5638552, -2.3277845, 2.3277841
3: -11.3187714, -7.7308869, -11.3187714, -7.7308869, -2.6977057, 2.6977053
4: 6.5160456, 8.8161116, 6.5160456, 8.8161116, -1.7021482, 1.7021484
5: -8.9200726, -5.8995981, -8.9200726, -5.8995981, -2.3079672, 2.3079672
6: -12.0180368, -8.1864262, -12.0180368, -8.1864262, -3.2302098, 3.2302094
7: -3.2622232, -0.5131162, -3.2622232, -0.5131162, -2.4733586, 2.4733586
8: -7.0121460, -3.4867215, -7.0121460, -3.4867215, -2.4142718, 2.4142714
9: -5.5743561, -3.0260987, -5.5743561, -3.0260987, -2.0417507, 2.0417509

Time for backsubstitution: 13.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 513

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0515115, upper bound: 1.0396766
time: 5.81 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0515144, upper bound: 1.0421877
time: 7.39 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.67 seconds
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 26.67
Output dim: 4, lower bound: -1.0415746, upper bound: 1.0496181
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.67
Output dim: 4, lower bound: -1.0415746, upper bound: 1.0515109
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.67
Output dim: 4, lower bound: -1.0515101, upper bound: 1.0396744
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.67
Output dim: 4, lower bound: -1.0515101, upper bound: 1.0415755
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.67
Output dim: 4, lower bound: -1.0515115, upper bound: 1.0396766
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.67
Output dim: 4, lower bound: -1.0515144, upper bound: 1.0421877

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9358540, -5.3535461, -8.9530983, -5.3171124, -3.0216498, 3.0029461
1: -7.4024291, -4.1341619, -7.4116006, -4.1047363, -2.4533470, 2.4357774
2: -7.4954801, -4.5863047, -7.5386381, -4.5646253, -2.3307724, 2.3292742
3: -11.2654305, -7.7266431, -11.3139076, -7.7319994, -2.6631603, 2.6887937
4: 6.5251904, 8.8051548, 6.5190859, 8.8159199, -1.6882098, 1.6989822
5: -8.9050674, -5.8673158, -8.9186268, -5.9020004, -2.2886715, 2.3159127
6: -12.0103903, -8.2206936, -12.0176849, -8.1965866, -3.2095127, 3.2019224
7: -3.2457504, -0.5721507, -3.2595482, -0.5160371, -2.4150834, 2.4424565
8: -6.9742551, -3.4564869, -7.0119791, -3.4886243, -2.4192934, 2.4302950
9: -5.5852075, -3.0280561, -5.5712357, -3.0262837, -2.0335989, 2.0406368

Time for backsubstitution: 13.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396754, upper bound: 1.0515120
time: 6.62 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396754, upper bound: 1.0515126
time: 5.96 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.9511261, -5.3192239, -8.9197674, -5.3591337, -2.9929714, 3.0020905
1: -7.4077177, -4.1055975, -7.3834295, -4.1582737, -2.4142823, 2.4269538
2: -7.5366130, -4.5667968, -7.4752836, -4.5928955, -2.3201494, 2.3051598
3: -11.3118496, -7.7335501, -11.2590923, -7.7627215, -2.6609507, 2.6527863
4: 6.5208225, 8.8142204, 6.5971775, 8.8024073, -1.6935778, 1.6322000
5: -8.9148407, -5.9031558, -8.9024849, -5.9274836, -2.2688918, 2.2844772
6: -12.0138874, -8.1986427, -11.9991140, -8.2676954, -3.1579084, 3.1949081
7: -3.2570975, -0.5203701, -3.1996956, -0.5760213, -2.4371862, 2.3851721
8: -7.0044899, -3.4903989, -6.9664278, -3.5248680, -2.4079828, 2.4066360
9: -5.5690064, -3.0296235, -5.5144129, -3.0330338, -2.0324337, 1.9809010

Time for backsubstitution: 13.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0496162, upper bound: 1.0396760
time: 8.23 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0496162, upper bound: 1.0396760
time: 6.58 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9693232, -5.3115182, -8.9197626, -5.3591385, -3.0127335, 3.0120993
1: -7.4311867, -4.0806074, -7.3834143, -4.1582766, -2.4437952, 2.4339175
2: -7.5572882, -4.5580583, -7.4752779, -4.5929008, -2.3408437, 2.3149636
3: -11.3203325, -7.6954956, -11.2590847, -7.7627273, -2.6717649, 2.6911588
4: 6.4462566, 8.8186750, 6.5971813, 8.8024006, -1.7601867, 1.6377366
5: -8.9212379, -5.8415532, -8.9024677, -5.9274864, -2.2758656, 2.3331451
6: -12.0289869, -8.1510868, -11.9990997, -8.2676992, -3.1747427, 3.2267833
7: -3.3056042, -0.5120972, -3.1996865, -0.5760386, -2.4777336, 2.3929911
8: -7.0198145, -3.4201684, -6.9664011, -3.5248725, -2.4259729, 2.4602821
9: -5.6425586, -3.0212951, -5.5144072, -3.0330462, -2.0863254, 1.9904220

Time for backsubstitution: 13.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0496162, upper bound: 1.0415748
time: 7.90 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0496162, upper bound: 1.0415775
time: 5.87 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.9533844, -5.3091950, -8.9553642, -5.3070812, -3.0596714, 3.0597162
1: -7.4092550, -4.0988493, -7.4131517, -4.0979872, -2.4570818, 2.4592528
2: -7.5379148, -4.5660362, -7.5399389, -4.5638552, -2.3260603, 2.3252163
3: -11.3167067, -7.7324371, -11.3187714, -7.7308869, -2.6950431, 2.6957791
4: 6.5177813, 8.8144073, 6.5160456, 8.8161116, -1.7004366, 1.7003021
5: -8.9162674, -5.9007535, -8.9200726, -5.8995981, -2.3038044, 2.3064976
6: -12.0142260, -8.1884604, -12.0180368, -8.1864262, -3.2263255, 3.2286520
7: -3.2597790, -0.5174644, -3.2622232, -0.5131162, -2.4706826, 2.4681869
8: -7.0046282, -3.4885011, -7.0121460, -3.4867215, -2.4066815, 2.4120080
9: -5.5721159, -3.0294533, -5.5743561, -3.0260987, -2.0395219, 2.0381999

Time for backsubstitution: 13.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0497384, upper bound: 1.0402892
time: 7.70 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0497384, upper bound: 1.0402897
time: 5.91 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.9715872, -5.3014936, -8.9553585, -5.3070879, -3.0794077, 3.0697117
1: -7.4327660, -4.0738554, -7.4131365, -4.0979891, -2.4861891, 2.4662194
2: -7.5586219, -4.5572996, -7.5399323, -4.5638604, -2.3513560, 2.3350053
3: -11.3251858, -7.6943636, -11.3187656, -7.7308908, -2.7059684, 2.7233531
4: 6.4431839, 8.8188581, 6.5160508, 8.8161049, -1.7705246, 1.7058413
5: -8.9226627, -5.8392239, -8.9200573, -5.8996024, -2.3107510, 2.3510449
6: -12.0293255, -8.1409950, -12.0180235, -8.1864300, -3.2432585, 3.2605686
7: -3.3082943, -0.5091909, -3.2622151, -0.5131299, -2.4993691, 2.4760065
8: -7.0199547, -3.4184093, -7.0121160, -3.4867277, -2.4246883, 2.4686093
9: -5.6456714, -3.0211258, -5.5743494, -3.0261126, -2.0957384, 2.0477331

Time for backsubstitution: 13.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 513

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0497384, upper bound: 1.0415729
time: 14.29 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0497384, upper bound: 1.0421889
time: 7.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 34.81 seconds
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.81
Output dim: 4, lower bound: -1.0396754, upper bound: 1.0515120
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.81
Output dim: 4, lower bound: -1.0396754, upper bound: 1.0515126
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 34.81
Output dim: 4, lower bound: -1.0496162, upper bound: 1.0396760
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 34.81
Output dim: 4, lower bound: -1.0496162, upper bound: 1.0396760
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 34.81
Output dim: 4, lower bound: -1.0496162, upper bound: 1.0415748
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 34.81
Output dim: 4, lower bound: -1.0496162, upper bound: 1.0415775
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 34.81
Output dim: 4, lower bound: -1.0497384, upper bound: 1.0402892
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 34.81
Output dim: 4, lower bound: -1.0497384, upper bound: 1.0402897
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 34.81
Output dim: 4, lower bound: -1.0497384, upper bound: 1.0415729
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 34.81
Output dim: 4, lower bound: -1.0497384, upper bound: 1.0421889

## BFS IS instance: IS_A1_B2_A2_B1

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

Time for backsubstitution: 13.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0392396, upper bound: 1.0481029
time: 5.03 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396739, upper bound: 1.0515121
time: 6.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2

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

Time for backsubstitution: 13.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0392396, upper bound: 1.0481032
time: 6.07 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0396739, upper bound: 1.0515126
time: 6.66 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 26.06 seconds
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 26.06
Output dim: 4, lower bound: -1.0392396, upper bound: 1.0481029
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 26.06
Output dim: 4, lower bound: -1.0396739, upper bound: 1.0515121
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 26.06
Output dim: 4, lower bound: -1.0392396, upper bound: 1.0481032
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 26.06
Output dim: 4, lower bound: -1.0396739, upper bound: 1.0515126

## BFS IS instance: IS_A1_B2_A2_B1_A2

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

Time for backsubstitution: 13.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0362614, upper bound: 1.0510808
time: 5.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0362614, upper bound: 1.0515091
time: 5.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -8.9362354, -5.3528666, -8.9697227, -5.3108339, -3.0301218, 3.0207968
1: -7.4028745, -4.1304970, -7.4316826, -4.0769310, -2.4601300, 2.4554312
2: -7.4986115, -4.5862241, -7.5603337, -4.5579762, -2.3408842, 2.3487582
3: -11.2656269, -7.7237730, -11.3205357, -7.6925931, -2.6920815, 2.6965649
4: 6.5197330, 8.8054237, 6.4407392, 8.8189440, -1.6953470, 1.7654905
5: -8.9051714, -5.8644099, -8.9213409, -5.8386340, -2.3363175, 2.3198583
6: -12.0111485, -8.2172995, -12.0297508, -8.1477966, -3.2471094, 3.2146530
7: -3.2469282, -0.5716619, -3.3068092, -0.5116191, -2.4191837, 2.4827423
8: -6.9750214, -3.4554727, -7.0205812, -3.4190929, -2.4411240, 2.4402785
9: -5.5903397, -3.0277004, -5.6477652, -3.0209398, -2.0446334, 2.0961401

Time for backsubstitution: 13.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 884

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 884

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0362614, upper bound: 1.0510825
time: 6.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.0362614, upper bound: 1.0515099
time: 7.59 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 27.02 seconds
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.02
Output dim: 4, lower bound: -1.0362614, upper bound: 1.0510808
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.02
Output dim: 4, lower bound: -1.0362614, upper bound: 1.0515091
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 27.02
Output dim: 4, lower bound: -1.0362614, upper bound: 1.0510825
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 27.02
Output dim: 4, lower bound: -1.0362614, upper bound: 1.0515099

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9356108, -5.3537445, -8.9437981, -5.3341174, -3.0020680, 2.9923925
1: -7.4022050, -4.1351690, -7.4002070, -4.1082516, -2.4433346, 2.4220846
2: -7.4945984, -4.5863709, -7.5238619, -4.5704727, -2.3241696, 2.3131776
3: -11.2653179, -7.7275467, -11.2910185, -7.7394915, -2.6548867, 2.6615036
4: 6.5267015, 8.8050432, 6.5232048, 8.8082619, -1.6795952, 1.6960495
5: -8.9050293, -5.8681026, -8.9110756, -5.9065533, -2.2835755, 2.3066823
6: -12.0097752, -8.2216530, -12.0043106, -8.2174978, -3.1844053, 3.1724691
7: -3.2451184, -0.5723919, -3.2521288, -0.5294276, -2.3977275, 2.4343853
8: -6.9739995, -3.4571548, -6.9979110, -3.4940147, -2.4139185, 2.4170403
9: -5.5836797, -3.0283813, -5.5666413, -3.0329342, -2.0250354, 2.0350325

Time for backsubstitution: 13.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1684
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 2136
type: A, layer: 3, pos: 3112
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2531
type: A, layer: 3, pos: 1486
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1982
type: A, layer: 3, pos: 572
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 907
type: A, layer: 3, pos: 1685
type: A, layer: 3, pos: 418
type: A, layer: 3, pos: 1943
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 176
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 760
type: A, layer: 3, pos: 1933
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2130
type: A, layer: 3, pos: 2237
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1971
type: A, layer: 3, pos: 2244
type: A, layer: 3, pos: 2390
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2922
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 1244
type: A, layer: 3, pos: 759
type: A, layer: 3, pos: 2328
type: A, layer: 3, pos: 1247
type: A, layer: 3, pos: 1802
type: A, layer: 3, pos: 894
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1153
type: A, layer: 3, pos: 2608
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 2852
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1253
type: A, layer: 3, pos: 1449
type: A, layer: 3, pos: 397
type: A, layer: 3, pos: 1778

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0167943, upper bound: 1.0423578
time: 6.60 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0274956, upper bound: 1.0423561
time: 7.51 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9356108, -5.3537445, -8.9511261, -5.3192263, -3.0184174, 2.9996634
1: -7.4022050, -4.1351690, -7.4077177, -4.1055994, -2.4529297, 2.4336069
2: -7.4945984, -4.5863709, -7.5366106, -4.5667982, -2.3271561, 2.3258288
3: -11.2653179, -7.7275467, -11.3118439, -7.7335515, -2.6548390, 2.6772690
4: 6.5267015, 8.8050432, 6.5208225, 8.8142204, -1.6821728, 1.7006340
5: -8.9050293, -5.8681026, -8.9148369, -5.9031549, -2.2854719, 2.3094151
6: -12.0097752, -8.2216530, -12.0138855, -8.1986446, -3.2108059, 3.2010036
7: -3.2451184, -0.5723919, -3.2570965, -0.5203720, -2.4074788, 2.4398463
8: -6.9739995, -3.4571548, -7.0044889, -3.4904020, -2.4200506, 2.4209018
9: -5.5836797, -3.0283813, -5.5690069, -3.0296249, -2.0300765, 2.0394168

Time for backsubstitution: 13.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1684
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 2136
type: A, layer: 3, pos: 3112
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 2531
type: A, layer: 3, pos: 1486
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 1982
type: A, layer: 3, pos: 572
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 907
type: A, layer: 3, pos: 1685
type: A, layer: 3, pos: 418
type: A, layer: 3, pos: 1943
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 176
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 760
type: A, layer: 3, pos: 1933
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2130
type: A, layer: 3, pos: 2237
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1971
type: A, layer: 3, pos: 2244
type: A, layer: 3, pos: 2390
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2922
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 1244
type: A, layer: 3, pos: 759
type: A, layer: 3, pos: 2328
type: A, layer: 3, pos: 1247
type: A, layer: 3, pos: 1802
type: A, layer: 3, pos: 894
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1153
type: A, layer: 3, pos: 2608
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 2852
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 1253
type: A, layer: 3, pos: 1449
type: A, layer: 3, pos: 397
type: A, layer: 3, pos: 1778

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0167943, upper bound: 1.0428043
time: 7.91 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0274956, upper bound: 1.0428040
time: 10.57 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9362354, -5.3528666, -8.9624062, -5.3257284, -3.0137663, 3.0139155
1: -7.4028745, -4.1304970, -7.4241848, -4.0795851, -2.4533753, 2.4434245
2: -7.4986115, -4.5862241, -7.5476227, -4.5616555, -2.3378768, 2.3360579
3: -11.2656269, -7.7237730, -11.2997074, -7.6985550, -2.6858525, 2.6756327
4: 6.5197330, 8.8054237, 6.4431286, 8.8129892, -1.6907327, 1.7623879
5: -8.9051714, -5.8644099, -8.9175787, -5.8420377, -2.3318462, 2.3159573
6: -12.0111485, -8.2172995, -12.0201588, -8.1665993, -3.2207203, 3.1937156
7: -3.2469282, -0.5716619, -3.3018773, -0.5206711, -2.4094334, 2.4712768
8: -6.9750214, -3.4554727, -7.0140138, -3.4226832, -2.4348965, 2.4357960
9: -5.5903397, -3.0277004, -5.6453848, -3.0242438, -2.0395966, 2.0922341

Time for backsubstitution: 13.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1684
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 2136
type: A, layer: 3, pos: 3112
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 2531
type: A, layer: 3, pos: 1486
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 572
type: A, layer: 3, pos: 1982
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 907
type: A, layer: 3, pos: 1685
type: A, layer: 3, pos: 418
type: A, layer: 3, pos: 1943
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 176
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 760
type: A, layer: 3, pos: 1933
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2130
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2237
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1971
type: A, layer: 3, pos: 2244
type: A, layer: 3, pos: 2390
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2922
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 1244
type: A, layer: 3, pos: 759
type: A, layer: 3, pos: 2328
type: A, layer: 3, pos: 1247
type: A, layer: 3, pos: 1802
type: A, layer: 3, pos: 894
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1153
type: A, layer: 3, pos: 2608
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 2852
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1253
type: A, layer: 3, pos: 1449
type: A, layer: 3, pos: 397
type: A, layer: 3, pos: 1778

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0167943, upper bound: 1.0423571
time: 7.54 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0274956, upper bound: 1.0423559
time: 6.05 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9362354, -5.3528666, -8.9697227, -5.3108368, -3.0301161, 3.0211825
1: -7.4028745, -4.1304970, -7.4316792, -4.0769320, -2.4629688, 2.4554288
2: -7.4986115, -4.5862241, -7.5603304, -4.5579777, -2.3408837, 2.3486776
3: -11.2656269, -7.7237730, -11.3205299, -7.6925945, -2.6920815, 2.6913922
4: 6.5197330, 8.8054237, 6.4407396, 8.8189411, -1.6933062, 1.7654899
5: -8.9051714, -5.8644099, -8.9213390, -5.8386345, -2.3359594, 2.3186932
6: -12.0111485, -8.2172995, -12.0297470, -8.1478014, -3.2471085, 3.2222404
7: -3.2469282, -0.5716619, -3.3068082, -0.5116208, -2.4191837, 2.4818947
8: -6.9750214, -3.4554727, -7.0205803, -3.4190936, -2.4411230, 2.4396677
9: -5.5903397, -3.0277004, -5.6477652, -3.0209394, -2.0446329, 2.0974064

Time for backsubstitution: 13.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 1789
type: A, layer: 3, pos: 2321
type: A, layer: 3, pos: 1395
type: A, layer: 3, pos: 2333
type: A, layer: 3, pos: 1684
type: A, layer: 3, pos: 170
type: A, layer: 3, pos: 2136
type: A, layer: 3, pos: 3112
type: A, layer: 3, pos: 2336
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 2531
type: A, layer: 3, pos: 1486
type: A, layer: 3, pos: 2528
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1404
type: A, layer: 3, pos: 572
type: A, layer: 3, pos: 1982
type: A, layer: 3, pos: 424
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 907
type: A, layer: 3, pos: 1685
type: A, layer: 3, pos: 418
type: A, layer: 3, pos: 1943
type: A, layer: 3, pos: 403
type: A, layer: 3, pos: 176
type: A, layer: 3, pos: 1934
type: A, layer: 3, pos: 760
type: A, layer: 3, pos: 1933
type: A, layer: 3, pos: 1846
type: A, layer: 3, pos: 2130
type: A, layer: 3, pos: 2378
type: A, layer: 3, pos: 2237
type: A, layer: 3, pos: 1839
type: A, layer: 3, pos: 1459
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1971
type: A, layer: 3, pos: 2244
type: A, layer: 3, pos: 2390
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2922
type: A, layer: 3, pos: 1992
type: A, layer: 3, pos: 1244
type: A, layer: 3, pos: 759
type: A, layer: 3, pos: 2328
type: A, layer: 3, pos: 1247
type: A, layer: 3, pos: 1802
type: A, layer: 3, pos: 894
type: A, layer: 3, pos: 3105
type: A, layer: 3, pos: 1753
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 416
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2216
type: A, layer: 3, pos: 1858
type: A, layer: 3, pos: 1153
type: A, layer: 3, pos: 2608
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 2852
type: A, layer: 3, pos: 2867
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 1253
type: A, layer: 3, pos: 1449
type: A, layer: 3, pos: 397
type: A, layer: 3, pos: 1778

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0167943, upper bound: 1.0423573
time: 6.14 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -1.0274956, upper bound: 1.0423594
time: 8.19 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 27.91 seconds
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 27.91
Output dim: 4, lower bound: -1.0167943, upper bound: 1.0423578
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 27.91
Output dim: 4, lower bound: -1.0274956, upper bound: 1.0423561
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 27.91
Output dim: 4, lower bound: -1.0167943, upper bound: 1.0428043
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 27.91
Output dim: 4, lower bound: -1.0274956, upper bound: 1.0428040
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 27.91
Output dim: 4, lower bound: -1.0167943, upper bound: 1.0423571
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 27.91
Output dim: 4, lower bound: -1.0274956, upper bound: 1.0423559
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 27.91
Output dim: 4, lower bound: -1.0167943, upper bound: 1.0423573
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 27.91
Output dim: 4, lower bound: -1.0274956, upper bound: 1.0423594
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.6619789600372314
rel_dist={4: [-1.051660938425652, 1.051661396029865]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2164.70 seconds
