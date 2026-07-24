## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.0301740019999999
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.1949067, -1.9940324, -6.1949067, -1.9940324, -4.2008743, 4.2008743)
1: (-12.2345352, -8.9912519, -12.2345352, -8.9912519, -3.2432833, 3.2432833)
2: (-5.6206846, -2.2354484, -5.6206846, -2.2354484, -3.3852363, 3.3852363)
3: (-5.3708906, -1.5042069, -5.3708906, -1.5042069, -3.8666837, 3.8666837)
4: (-11.5238037, -7.5678163, -11.5238037, -7.5678163, -3.9559875, 3.9559875)
5: (-6.2900958, -3.0817809, -6.2900958, -3.0817809, -3.2083149, 3.2083149)
6: (-12.4278812, -8.6957130, -12.4278812, -8.6957130, -3.7321682, 3.7321682)
7: (-8.1703596, -4.6722708, -8.1703596, -4.6722708, -3.4980888, 3.4980888)
8: (7.7388387, 10.0601540, 7.7388387, 10.0601540, -2.3213153, 2.3213153)
9: (-6.3480244, -2.8028445, -6.3480244, -2.8028445, -3.5451798, 3.5451798)

## BASE Result
execution time: IAR + LP analysis = 14.82 + 40.05 = 54.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.7692351, upper bound: 1.7692330


# Binary Search by BASE starts (time budget: 3545.13 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.321315288543701
rel_dist={8: [-1.3149698422491287, 1.314969004521421]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.225489616394043
rel_dist={8: [-1.040580051898944, 1.0405791313826693]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.099544048309326
rel_dist={8: [-0.8121649022104638, 0.8121633074202457]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.1625168323516846
rel_dist={8: [-0.9337397585648102, 0.9337386492559308]}

## Binary Search Result
Binary search time: 204.48 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3340.65 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3975430, upper bound: 1.3998457
time: 7.08 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998458, upper bound: 1.3998480
time: 6.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.58
Output dim: 8, lower bound: -1.3975430, upper bound: 1.3998457
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.58
Output dim: 8, lower bound: -1.3998458, upper bound: 1.3998480

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -6.1548796, -2.0182879, -6.1907473, -1.9995725, -3.6406112, 3.6925511
1: -12.2045288, -9.0081425, -12.2286587, -8.9920654, -3.1954403, 3.2205162
2: -5.5950623, -2.2592869, -5.6197281, -2.2392721, -3.0741434, 3.1090555
3: -5.3431683, -1.5320253, -5.3696628, -1.5100133, -3.8331549, 3.8376374
4: -11.4873419, -7.5901585, -11.5207491, -7.5726442, -3.5558319, 3.7422228
5: -6.2218180, -3.1273403, -6.2747593, -3.0830684, -2.8905096, 2.8927870
6: -12.3938370, -8.7286739, -12.4207287, -8.6971512, -3.3221521, 3.2952757
7: -8.1438236, -4.6864657, -8.1661339, -4.6731272, -3.4706964, 3.4796681
8: 7.7507811, 10.0393200, 7.7402463, 10.0580120, -2.3072309, 2.2990737
9: -6.3052926, -2.8363342, -6.3445244, -2.8109336, -3.2122927, 3.2247896

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3975430, upper bound: 1.3975426
time: 7.75 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3975430, upper bound: 1.3998465
time: 5.71 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.1948943, -1.9940510, -6.1949048, -1.9940357, -3.7113338, 3.7036233
1: -12.2345200, -8.9912529, -12.2345314, -8.9912539, -3.2160292, 3.2340021
2: -5.6206799, -2.2354584, -5.6206841, -2.2354517, -3.1093655, 3.1000972
3: -5.3708854, -1.5042269, -5.3708897, -1.5042117, -3.8666737, 3.8666627
4: -11.5237951, -7.5678372, -11.5238018, -7.5678205, -3.7588739, 3.7540245
5: -6.2900400, -3.0817838, -6.2900801, -3.0817823, -2.9459953, 2.9675026
6: -12.4278564, -8.6957150, -12.4278736, -8.6957121, -3.3566008, 3.3671026
7: -8.1703444, -4.6722717, -8.1703548, -4.6722727, -3.4980717, 3.4980831
8: 7.7388434, 10.0601482, 7.7388411, 10.0601521, -2.3213086, 2.3213072
9: -6.3480148, -2.8028774, -6.3480225, -2.8028526, -3.2682600, 3.2564197

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998458, upper bound: 1.3975430
time: 4.88 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998459, upper bound: 1.3998458
time: 4.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.83 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.83
Output dim: 8, lower bound: -1.3975430, upper bound: 1.3975426
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.83
Output dim: 8, lower bound: -1.3975430, upper bound: 1.3998465
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 23.83
Output dim: 8, lower bound: -1.3998458, upper bound: 1.3975430
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 23.83
Output dim: 8, lower bound: -1.3998459, upper bound: 1.3998458

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -6.1548796, -2.0182879, -6.1548796, -2.0182879, -3.6564388, 3.6564384
1: -12.2045288, -9.0081425, -12.2045288, -9.0081425, -3.1774664, 3.1774669
2: -5.5950623, -2.2592869, -5.5950623, -2.2592869, -3.0803022, 3.0803022
3: -5.3431683, -1.5320253, -5.3431683, -1.5320253, -3.8111429, 3.8111429
4: -11.4873419, -7.5901585, -11.4873419, -7.5901585, -3.5399332, 3.5399332
5: -6.2218180, -3.1273403, -6.2218180, -3.1273403, -2.8417068, 2.8417068
6: -12.3938370, -8.7286739, -12.3938370, -8.7286739, -3.2682085, 3.2682085
7: -8.1438236, -4.6864657, -8.1438236, -4.6864657, -3.4573579, 3.4573579
8: 7.7507811, 10.0393200, 7.7507811, 10.0393200, -2.2885389, 2.2885389
9: -6.3052926, -2.8363342, -6.3052926, -2.8363342, -3.1846576, 3.1846571

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3975417, upper bound: 1.3953549
time: 7.86 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3975394, upper bound: 1.3975410
time: 7.57 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -6.1548796, -2.0182879, -6.1948051, -1.9940507, -3.6460643, 3.6962519
1: -12.2045288, -9.0081425, -12.2345161, -8.9912558, -3.1953783, 3.2263737
2: -5.5950623, -2.2592869, -5.6206799, -2.2354763, -3.0795517, 3.1063633
3: -5.3431683, -1.5320253, -5.3708653, -1.5042288, -3.8389394, 3.8388400
4: -11.4873419, -7.5901585, -11.5237923, -7.5678511, -3.5603552, 3.7426419
5: -6.2218180, -3.1273403, -6.2900343, -3.0818195, -2.8919997, 2.9084167
6: -12.3938370, -8.7286739, -12.4278450, -8.6957493, -3.3234835, 3.3026600
7: -8.1438236, -4.6864657, -8.1703186, -4.6722727, -3.4715509, 3.4838529
8: 7.7507811, 10.0393200, 7.7388453, 10.0601120, -2.3093309, 2.3004746
9: -6.3052926, -2.8363342, -6.3479738, -2.8028793, -3.2203188, 3.2277780

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3975394, upper bound: 1.3976631
time: 5.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3975394, upper bound: 1.3998416
time: 5.95 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.1948943, -1.9940510, -6.1548796, -2.0182879, -3.7191725, 3.6460786
1: -12.2345200, -8.9912529, -12.2045288, -9.0081425, -3.2263775, 3.1788478
2: -5.6206799, -2.2354584, -5.5950623, -2.2592869, -3.0992393, 3.0795751
3: -5.3708854, -1.5042269, -5.3431683, -1.5320253, -3.8388600, 3.8389413
4: -11.5237951, -7.5678372, -11.4873419, -7.5901585, -3.7426767, 3.5594029
5: -6.2900400, -3.0817838, -6.2218180, -3.1273403, -2.9084225, 2.9004440
6: -12.4278564, -8.6957150, -12.3938370, -8.7286739, -3.3026924, 3.3328023
7: -8.1703444, -4.6722717, -8.1438236, -4.6864657, -3.4838786, 3.4715519
8: 7.7388434, 10.0601482, 7.7507811, 10.0393200, -2.3004766, 2.3093672
9: -6.3480148, -2.8028774, -6.3052926, -2.8363342, -3.2325315, 3.2203193

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998421, upper bound: 1.3953529
time: 7.77 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998421, upper bound: 1.3975392
time: 8.14 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.1948943, -1.9940510, -6.1948943, -1.9940510, -3.7036142, 3.7036138
1: -12.2345200, -8.9912529, -12.2345200, -8.9912529, -3.2160254, 3.2160258
2: -5.6206799, -2.2354584, -5.6206799, -2.2354584, -3.1093569, 3.1093564
3: -5.3708854, -1.5042269, -5.3708854, -1.5042269, -3.8666584, 3.8666584
4: -11.5237951, -7.5678372, -11.5237951, -7.5678372, -3.7540131, 3.7540121
5: -6.2900400, -3.0817838, -6.2900400, -3.0817838, -2.9459929, 2.9459925
6: -12.4278564, -8.6957150, -12.4278564, -8.6957150, -3.3565974, 3.3565974
7: -8.1703444, -4.6722717, -8.1703444, -4.6722717, -3.4980726, 3.4980726
8: 7.7388434, 10.0601482, 7.7388434, 10.0601482, -2.3213048, 2.3213048
9: -6.3480148, -2.8028774, -6.3480148, -2.8028774, -3.2564135, 3.2564135

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998423, upper bound: 1.3953548
time: 11.53 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998423, upper bound: 1.3975416
time: 7.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 33.30 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 8, lower bound: -1.3975417, upper bound: 1.3953549
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 8, lower bound: -1.3975394, upper bound: 1.3975410
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 8, lower bound: -1.3975394, upper bound: 1.3976631
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 8, lower bound: -1.3975394, upper bound: 1.3998416
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 8, lower bound: -1.3998421, upper bound: 1.3953529
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 8, lower bound: -1.3998421, upper bound: 1.3975392
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 8, lower bound: -1.3998423, upper bound: 1.3953548
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 8, lower bound: -1.3998423, upper bound: 1.3975416

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1326585, -2.0187602, -6.1531439, -2.0183253, -3.6342154, 3.6540933
1: -12.1895523, -9.0087080, -12.2033548, -9.0081854, -3.1624632, 3.1756821
2: -5.5764723, -2.2627537, -5.5936089, -2.2595568, -3.0610838, 3.0758147
3: -5.3413172, -1.5354044, -5.3430257, -1.5322912, -3.8090260, 3.8076212
4: -11.4839725, -7.6006575, -11.4870796, -7.5909810, -3.5356541, 3.5292420
5: -6.2195330, -3.1305327, -6.2216415, -3.1275926, -2.8350744, 2.8332620
6: -12.3920040, -8.7305202, -12.3936958, -8.7288151, -3.2601967, 3.2593269
7: -8.1394825, -4.6933384, -8.1434879, -4.6870041, -3.4524784, 3.4501495
8: 7.7532110, 10.0342607, 7.7509699, 10.0389223, -2.2857113, 2.2832909
9: -6.3017912, -2.8385584, -6.3050184, -2.8365092, -3.1781673, 3.1793137

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3953545, upper bound: 1.3953543
time: 5.83 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3953545, upper bound: 1.3953554
time: 5.97 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.1636796, -1.9719963, -6.1548653, -2.0182891, -3.6629968, 3.7027369
1: -12.2169275, -8.9913034, -12.2045174, -9.0081425, -3.1907201, 3.1949706
2: -5.5972962, -2.2221589, -5.5950546, -2.2592897, -3.0815392, 3.1172173
3: -5.3585348, -1.5285611, -5.3431683, -1.5320275, -3.8265073, 3.8146071
4: -11.5102978, -7.5843287, -11.4873390, -7.5901632, -3.5625620, 3.5446410
5: -6.2342663, -3.1250095, -6.2218180, -3.1273427, -2.8507566, 2.8524475
6: -12.4050369, -8.7265186, -12.3938370, -8.7286730, -3.2722874, 3.2818279
7: -8.1610918, -4.6861424, -8.1438217, -4.6864691, -3.4746227, 3.4576793
8: 7.7421999, 10.0472679, 7.7507839, 10.0393162, -2.2971163, 2.2964840
9: -6.3083148, -2.8212578, -6.3052917, -2.8363361, -3.1847095, 3.2045636

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3973630, upper bound: 1.3951970
time: 6.02 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3975384, upper bound: 1.3975374
time: 7.13 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1326585, -2.0187602, -6.1930695, -1.9941096, -3.6238971, 3.6939087
1: -12.1895523, -9.0087080, -12.2333279, -8.9912968, -3.1803966, 3.2246199
2: -5.5764723, -2.2627537, -5.6192293, -2.2357736, -3.0604715, 3.1018763
3: -5.3413172, -1.5354044, -5.3707194, -1.5045178, -3.8367994, 3.8330326
4: -11.4839725, -7.6006575, -11.5234146, -7.5686703, -3.5560741, 3.7318263
5: -6.2195330, -3.1305327, -6.2898555, -3.0820808, -2.8852139, 2.8999696
6: -12.3920040, -8.7305202, -12.4277010, -8.6959047, -3.3152990, 3.2937784
7: -8.1394825, -4.6933384, -8.1699047, -4.6728096, -3.4666729, 3.4765663
8: 7.7532110, 10.0342607, 7.7390442, 10.0597153, -2.3065042, 2.2952166
9: -6.3017912, -2.8385584, -6.3476996, -2.8030524, -3.2137928, 3.2224350

Time for backsubstitution: 15.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3973611, upper bound: 1.3952374
time: 8.26 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3975364, upper bound: 1.3976591
time: 6.71 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.1636796, -1.9719963, -6.1947885, -1.9940515, -3.6531458, 3.7341247
1: -12.2169275, -8.9913034, -12.2345066, -8.9912548, -3.2083488, 3.2432032
2: -5.5972962, -2.2221589, -5.6206708, -2.2354794, -3.0808830, 3.1278381
3: -5.3585348, -1.5285611, -5.3708625, -1.5042329, -3.8543019, 3.8423014
4: -11.5102978, -7.5843287, -11.5237885, -7.5678563, -3.5829792, 3.7486143
5: -6.2342663, -3.1250095, -6.2900314, -3.0818210, -2.8970180, 2.9191594
6: -12.4050369, -8.7265186, -12.4278460, -8.6957502, -3.3275051, 3.3162794
7: -8.1610918, -4.6861424, -8.1703176, -4.6722755, -3.4888163, 3.4841752
8: 7.7421999, 10.0472679, 7.7388468, 10.0601063, -2.3179064, 2.3084211
9: -6.3083148, -2.8212578, -6.3479719, -2.8028808, -3.2203407, 3.2476840

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3973611, upper bound: 1.3974384
time: 9.08 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3975364, upper bound: 1.3998384
time: 14.06 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.1727061, -1.9947910, -6.1531439, -2.0183253, -3.6969728, 3.6433854
1: -12.2193680, -8.9918194, -12.2033548, -9.0081854, -3.2111826, 3.1770630
2: -5.6021352, -2.2392907, -5.5936089, -2.2595568, -3.0800171, 3.0746841
3: -5.3690128, -1.5079150, -5.3430257, -1.5322912, -3.8367217, 3.8351107
4: -11.5189600, -7.5783281, -11.4870796, -7.5909810, -3.7372417, 3.5487080
5: -6.2877455, -3.0851088, -6.2216415, -3.1275926, -2.9017782, 2.8916631
6: -12.4259882, -8.6976967, -12.3936958, -8.7288151, -3.2946806, 3.3235612
7: -8.1650248, -4.6791430, -8.1434879, -4.6870041, -3.4780207, 3.4643450
8: 7.7414203, 10.0550957, 7.7509699, 10.0389223, -2.2975020, 2.3041258
9: -6.3445187, -2.8051057, -6.3050184, -2.8365092, -3.2260442, 3.2149243

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976620, upper bound: 1.3953522
time: 5.42 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976620, upper bound: 1.3953520
time: 5.07 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.2037034, -1.9477634, -6.1548653, -2.0182891, -3.7255554, 3.6871424
1: -12.2469349, -8.9744205, -12.2045174, -9.0081425, -3.2387924, 3.1963506
2: -5.6229372, -2.1984165, -5.5950546, -2.2592897, -3.1004515, 3.1089802
3: -5.3863192, -1.5007141, -5.3431683, -1.5320275, -3.8542917, 3.8424542
4: -11.5466518, -7.5620289, -11.4873390, -7.5901632, -3.7651749, 3.5640182
5: -6.3024464, -3.0794811, -6.2218180, -3.1273427, -2.9174271, 2.9038560
6: -12.4390230, -8.6935825, -12.3938370, -8.7286730, -3.3068151, 3.3427658
7: -8.1877050, -4.6719499, -8.1438217, -4.6864691, -3.5012360, 3.4718719
8: 7.7302933, 10.0681105, 7.7507839, 10.0393162, -2.3090229, 2.3173265
9: -6.3509479, -2.7878442, -6.3052917, -2.8363361, -3.2325001, 3.2402496

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3996256, upper bound: 1.3951937
time: 5.21 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998392, upper bound: 1.3975359
time: 7.93 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.1727061, -1.9947910, -6.1931620, -1.9941092, -3.6814809, 3.7009206
1: -12.2193680, -8.9918194, -12.2333336, -8.9912977, -3.2006903, 3.2141633
2: -5.6021352, -2.2392907, -5.6192303, -2.2357559, -3.0902729, 3.1044669
3: -5.3690128, -1.5079150, -5.3707399, -1.5045149, -3.8644979, 3.8628249
4: -11.5189600, -7.5783281, -11.5234213, -7.5686564, -3.7486992, 3.7431960
5: -6.2877455, -3.0851088, -6.2898607, -3.0820456, -2.9391918, 2.9372115
6: -12.4259882, -8.6976967, -12.4277124, -8.6958675, -3.3484144, 3.3473573
7: -8.1650248, -4.6791430, -8.1699295, -4.6728077, -3.4922171, 3.4907866
8: 7.7414203, 10.0550957, 7.7390442, 10.0597515, -2.3183312, 2.3160515
9: -6.3445187, -2.8051057, -6.3477397, -2.8030517, -3.2498908, 3.2510176

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3996257, upper bound: 1.3929858
time: 5.46 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998393, upper bound: 1.3953486
time: 6.03 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.2037034, -1.9477634, -6.1948791, -1.9940519, -3.7105103, 3.7412341
1: -12.2469349, -8.9744205, -12.2345114, -8.9912548, -3.2290535, 3.2335877
2: -5.6229372, -2.1984165, -5.6206717, -2.2354598, -3.1106644, 3.1233535
3: -5.3863192, -1.5007141, -5.3708854, -1.5042281, -3.8820910, 3.8701713
4: -11.5466518, -7.5620289, -11.5237923, -7.5678396, -3.7766962, 3.7598696
5: -6.3024464, -3.0794811, -6.2900381, -3.0817866, -2.9544454, 2.9570794
6: -12.4390230, -8.6935825, -12.4278564, -8.6957169, -3.3608379, 3.3704271
7: -8.1877050, -4.6719499, -8.1703424, -4.6722755, -3.5154295, 3.4983926
8: 7.7302933, 10.0681105, 7.7388458, 10.0601444, -2.3298512, 2.3292646
9: -6.3509479, -2.7878442, -6.3480110, -2.8028791, -3.2563515, 3.2763433

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3996257, upper bound: 1.3951935
time: 9.95 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3998393, upper bound: 1.3975364
time: 7.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 32.09 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3953545, upper bound: 1.3953543
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3953545, upper bound: 1.3953554
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3973630, upper bound: 1.3951970
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3975384, upper bound: 1.3975374
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3973611, upper bound: 1.3952374
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3975364, upper bound: 1.3976591
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3973611, upper bound: 1.3974384
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3975364, upper bound: 1.3998384
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3976620, upper bound: 1.3953522
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3976620, upper bound: 1.3953520
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3996256, upper bound: 1.3951937
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3998392, upper bound: 1.3975359
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3996257, upper bound: 1.3929858
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3998393, upper bound: 1.3953486
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3996257, upper bound: 1.3951935
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.09
Output dim: 8, lower bound: -1.3998393, upper bound: 1.3975364

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1326585, -2.0187602, -6.1326585, -2.0187602, -3.6336517, 3.6336517
1: -12.1895523, -9.0087080, -12.1895523, -9.0087080, -3.1618996, 3.1618996
2: -5.5764723, -2.2627537, -5.5764723, -2.2627537, -3.0583143, 3.0583138
3: -5.3413172, -1.5354044, -5.3413172, -1.5354044, -3.8059127, 3.8059127
4: -11.4839725, -7.6006575, -11.4839725, -7.6006575, -3.5260429, 3.5260429
5: -6.2195330, -3.1305327, -6.2195330, -3.1305327, -2.8277187, 2.8277187
6: -12.3920040, -8.7305202, -12.3920040, -8.7305202, -3.2525377, 3.2525382
7: -8.1394825, -4.6933384, -8.1394825, -4.6933384, -3.4461441, 3.4461441
8: 7.7532110, 10.0342607, 7.7532110, 10.0342607, -2.2810497, 2.2810497
9: -6.3017912, -2.8385584, -6.3017912, -2.8385584, -3.1736817, 3.1736813

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3929880, upper bound: 1.3951572
time: 9.10 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3953512, upper bound: 1.3953517
time: 5.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1326585, -2.0187602, -6.1636796, -1.9719963, -3.6805701, 3.6637969
1: -12.1895523, -9.0087080, -12.2169275, -8.9913034, -3.1800251, 3.1902781
2: -5.5764723, -2.2627537, -5.5972962, -2.2221589, -3.0981627, 3.0801883
3: -5.3413172, -1.5354044, -5.3585348, -1.5285611, -3.8127561, 3.8231304
4: -11.4839725, -7.6006575, -11.5102978, -7.5843287, -3.5417118, 3.5521393
5: -6.2195330, -3.1305327, -6.2342663, -3.1250095, -2.8337212, 2.8427849
6: -12.3920040, -8.7305202, -12.4050369, -8.7265186, -3.2580018, 3.2639856
7: -8.1394825, -4.6933384, -8.1610918, -4.6861424, -3.4533401, 3.4677534
8: 7.7532110, 10.0342607, 7.7421999, 10.0472679, -2.2940569, 2.2920609
9: -6.3017912, -2.8385584, -6.3083148, -2.8212578, -3.1908278, 3.1798482

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3929880, upper bound: 1.3951594
time: 5.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3953512, upper bound: 1.3953513
time: 10.53 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1628723, -1.9776068, -6.1443291, -2.0420170, -3.6382818, 3.6840377
1: -12.2141924, -8.9925451, -12.1909332, -9.0144758, -3.1811056, 3.1796422
2: -5.5966702, -2.2269361, -5.5889854, -2.2802243, -3.0591249, 3.1003804
3: -5.3546576, -1.5296497, -5.3248448, -1.5409472, -3.8137105, 3.7951951
4: -11.5072680, -7.5853844, -11.4718199, -7.5989733, -3.5477800, 3.5279527
5: -6.2335210, -3.1300130, -6.2131395, -3.1486368, -2.8273859, 2.8371773
6: -12.4039450, -8.7294416, -12.3860149, -8.7415228, -3.2577586, 3.2713070
7: -8.1573086, -4.6866994, -8.1251516, -4.6898036, -3.4675050, 3.4384522
8: 7.7447777, 10.0461206, 7.7624106, 10.0320148, -2.2872372, 2.2837100
9: -6.3069935, -2.8237147, -6.2936749, -2.8489580, -3.1700153, 3.1905680

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3959888, upper bound: 1.3951879
time: 7.38 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3973557, upper bound: 1.3951877
time: 13.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1636796, -1.9719963, -6.1548657, -2.0182943, -3.6501737, 3.7027364
1: -12.2169275, -8.9913034, -12.2045164, -9.0081425, -3.1930680, 3.1949682
2: -5.5972962, -2.2221589, -5.5950546, -2.2592931, -3.0777807, 3.1144660
3: -5.3585348, -1.5285611, -5.3431644, -1.5320289, -3.8265059, 3.8146033
4: -11.5102978, -7.5843287, -11.4873390, -7.5901642, -3.5622330, 3.5443444
5: -6.2342663, -3.1250095, -6.2218156, -3.1273470, -2.8377523, 2.8524485
6: -12.4050369, -8.7265186, -12.3938370, -8.7286797, -3.2719364, 3.2818260
7: -8.1610918, -4.6861424, -8.1438189, -4.6864715, -3.4746203, 3.4576764
8: 7.7421999, 10.0472679, 7.7507849, 10.0393143, -2.2971144, 2.2964830
9: -6.3083148, -2.8212578, -6.3052893, -2.8363383, -3.1813474, 3.2045622

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3951957, upper bound: 1.3973624
time: 8.95 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3951956, upper bound: 1.3975405
time: 6.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1318545, -2.0243876, -6.1824675, -2.0180483, -3.5989389, 3.6751456
1: -12.1867981, -9.0099449, -12.2196188, -8.9976387, -3.1707811, 3.2096739
2: -5.5758524, -2.2675116, -5.6131358, -2.2570243, -3.0379963, 3.0893679
3: -5.3374052, -1.5364904, -5.3523359, -1.5137260, -3.8236792, 3.8128819
4: -11.4809265, -7.6017036, -11.5067291, -7.5774488, -3.5413599, 3.7136574
5: -6.2187891, -3.1355476, -6.2811947, -3.1034884, -2.8617401, 2.8846965
6: -12.3909130, -8.7334480, -12.4198971, -8.7088432, -3.3006592, 3.2832470
7: -8.1356812, -4.6938944, -8.1504421, -4.6761622, -3.4595189, 3.4565477
8: 7.7557945, 10.0331135, 7.7508097, 10.0523739, -2.2965794, 2.2823038
9: -6.3004732, -2.8410728, -6.3360090, -2.8157008, -3.1990590, 3.2082958

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3959869, upper bound: 1.3952303
time: 7.91 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3973537, upper bound: 1.3952297
time: 8.50 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1326585, -2.0187602, -6.1930695, -1.9941125, -3.6104422, 3.6939077
1: -12.1895523, -9.0087080, -12.2333241, -8.9912996, -3.1827431, 3.2246161
2: -5.5764723, -2.2627537, -5.6192303, -2.2357755, -3.0583591, 3.1018753
3: -5.3413172, -1.5354044, -5.3707156, -1.5045180, -3.8367991, 3.8278527
4: -11.4839725, -7.6006575, -11.5234146, -7.5686736, -3.5557432, 3.7299294
5: -6.2195330, -3.1305327, -6.2898540, -3.0820856, -2.8728676, 2.8999691
6: -12.3920040, -8.7305202, -12.4277000, -8.6959066, -3.3152947, 3.2937775
7: -8.1394825, -4.6933384, -8.1699009, -4.6728072, -3.4666753, 3.4765625
8: 7.7532110, 10.0342607, 7.7390461, 10.0597143, -2.3065033, 2.2952147
9: -6.3017912, -2.8385584, -6.3476977, -2.8030546, -3.2105384, 3.2224331

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3951936, upper bound: 1.3974305
time: 9.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3951936, upper bound: 1.3976612
time: 5.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.1628723, -1.9776068, -6.1841869, -2.0179896, -3.6281872, 3.7100790
1: -12.2141924, -8.9925451, -12.2208014, -8.9975929, -3.1987505, 3.2282562
2: -5.5966702, -2.2269361, -5.6145811, -2.2567251, -3.0583954, 3.1109934
3: -5.3546576, -1.5296497, -5.3524790, -1.5134382, -3.8412194, 3.8228292
4: -11.5072680, -7.5853844, -11.5071087, -7.5766335, -3.5682440, 3.7304540
5: -6.2335210, -3.1300130, -6.2813711, -3.1032238, -2.8735533, 2.9038992
6: -12.4039450, -8.7294416, -12.4200382, -8.7086916, -3.3128476, 3.3057690
7: -8.1573086, -4.6866994, -8.1508570, -4.6756287, -3.4816799, 3.4641576
8: 7.7447777, 10.0461206, 7.7506104, 10.0527649, -2.3079872, 2.2955103
9: -6.3069935, -2.8237147, -6.3362799, -2.8155293, -3.2056007, 3.2335987

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3959870, upper bound: 1.3974309
time: 9.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3973538, upper bound: 1.3974309
time: 6.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1636796, -1.9719963, -6.1947889, -1.9940553, -3.6396966, 3.7290964
1: -12.2169275, -8.9913034, -12.2345037, -8.9912548, -3.2106938, 3.2432003
2: -5.5972962, -2.2221589, -5.6206703, -2.2354822, -3.0787716, 3.1250825
3: -5.3585348, -1.5285611, -5.3708601, -1.5042338, -3.8543010, 3.8422990
4: -11.5102978, -7.5843287, -11.5237865, -7.5678568, -3.5826492, 3.7467160
5: -6.2342663, -3.1250095, -6.2900324, -3.0818272, -2.8846512, 2.9191580
6: -12.4050369, -8.7265186, -12.4278431, -8.6957531, -3.3275023, 3.3162789
7: -8.1610918, -4.6861424, -8.1703119, -4.6722760, -3.4888158, 3.4841695
8: 7.7421999, 10.0472679, 7.7388487, 10.0601063, -2.3179064, 2.3084192
9: -6.3083148, -2.8212578, -6.3479691, -2.8028843, -3.2170849, 3.2476835

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3951936, upper bound: 1.3996251
time: 10.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3951936, upper bound: 1.3998412
time: 6.75 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1727061, -1.9947910, -6.1326585, -2.0187602, -3.6964083, 3.6230202
1: -12.2193680, -8.9918194, -12.1895523, -9.0087080, -3.2106600, 3.1632791
2: -5.6021352, -2.2392907, -5.5764723, -2.2627537, -3.0772476, 3.0573406
3: -5.3690128, -1.5079150, -5.3413172, -1.5354044, -3.8336084, 3.8334022
4: -11.5189600, -7.5783281, -11.4839725, -7.6006575, -3.7276087, 3.5455108
5: -6.2877455, -3.0851088, -6.2195330, -3.1305327, -2.8944225, 2.8860016
6: -12.4259882, -8.6976967, -12.3920040, -8.7305202, -3.2870216, 3.3166447
7: -8.1650248, -4.6791430, -8.1394825, -4.6933384, -3.4716864, 3.4603395
8: 7.7414203, 10.0550957, 7.7532110, 10.0342607, -2.2928405, 2.3018847
9: -6.3445187, -2.8051057, -6.3017912, -2.8385584, -3.2215586, 3.2092619

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952380, upper bound: 1.3951549
time: 5.01 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976588, upper bound: 1.3953492
time: 5.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1727061, -1.9947910, -6.1636796, -1.9719963, -3.7341461, 3.6535592
1: -12.2193680, -8.9918194, -12.2169275, -8.9913034, -3.2280645, 3.1916585
2: -5.6021352, -2.2392907, -5.5972962, -2.2221589, -3.1017437, 3.0790887
3: -5.3690128, -1.5079150, -5.3585348, -1.5285611, -3.8404517, 3.8506198
4: -11.5189600, -7.5783281, -11.5102978, -7.5843287, -3.7443485, 3.5716062
5: -6.2877455, -3.0851088, -6.2342663, -3.1250095, -2.9004254, 2.8975017
6: -12.4259882, -8.6976967, -12.4050369, -8.7265186, -3.2924857, 3.3283505
7: -8.1650248, -4.6791430, -8.1610918, -4.6861424, -3.4788823, 3.4819489
8: 7.7414203, 10.0550957, 7.7421999, 10.0472679, -2.3058476, 2.3128958
9: -6.3445187, -2.8051057, -6.3083148, -2.8212578, -3.2387042, 3.2154307

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3952379, upper bound: 1.3951551
time: 4.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3976588, upper bound: 1.3953483
time: 5.87 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.2028775, -1.9533710, -6.1443291, -2.0420170, -3.7008004, 3.6632307
1: -12.2442198, -8.9756622, -12.1909332, -9.0144758, -3.2297440, 3.1809702
2: -5.6223068, -2.2031801, -5.5889854, -2.2802243, -3.0780363, 3.0924046
3: -5.3824334, -1.5017996, -5.3248448, -1.5409472, -3.8414862, 3.8230453
4: -11.5436087, -7.5630770, -11.4718199, -7.5989733, -3.7516785, 3.5473328
5: -6.3017092, -3.0844865, -6.2131395, -3.1486368, -2.8940630, 2.8825288
6: -12.4379368, -8.6965008, -12.3860149, -8.7415228, -3.2922912, 3.3289382
7: -8.1839523, -4.6725121, -8.1251516, -4.6898036, -3.4941487, 3.4526396
8: 7.7328653, 10.0669527, 7.7624106, 10.0320148, -2.2991495, 2.3045421
9: -6.3495936, -2.7903023, -6.2936749, -2.8489580, -3.2177806, 3.2262492

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3983584, upper bound: 1.3951884
time: 6.99 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3996178, upper bound: 1.3951860
time: 6.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 28.22 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3929880, upper bound: 1.3951572
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3953512, upper bound: 1.3953517
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3929880, upper bound: 1.3951594
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3953512, upper bound: 1.3953513
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3959888, upper bound: 1.3951879
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3973557, upper bound: 1.3951877
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3951957, upper bound: 1.3973624
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3951956, upper bound: 1.3975405
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3959869, upper bound: 1.3952303
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3973537, upper bound: 1.3952297
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3951936, upper bound: 1.3974305
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3951936, upper bound: 1.3976612
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3959870, upper bound: 1.3974309
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3973538, upper bound: 1.3974309
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3951936, upper bound: 1.3996251
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3951936, upper bound: 1.3998412
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3952380, upper bound: 1.3951549
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3976588, upper bound: 1.3953492
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3952379, upper bound: 1.3951551
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3976588, upper bound: 1.3953483
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3983584, upper bound: 1.3951884
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 8, lower bound: -1.3996178, upper bound: 1.3951860
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 8, lower bound: -1.3998392, upper bound: 1.3975359
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 8, lower bound: -1.3996257, upper bound: 1.3929858
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 8, lower bound: -1.3998393, upper bound: 1.3953486
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 8, lower bound: -1.3996257, upper bound: 1.3951935
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.22
Output dim: 8, lower bound: -1.3998393, upper bound: 1.3975364
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.321315288543701
rel_dist={8: [-1.3998489967280836, 1.3998484636799233]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1348737, upper bound: 1.1366273
time: 5.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366280, upper bound: 1.1366270
time: 6.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.94
Output dim: 8, lower bound: -1.1348737, upper bound: 1.1366273
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.94
Output dim: 8, lower bound: -1.1366280, upper bound: 1.1366270

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -6.1548796, -2.0182879, -6.1874065, -2.0039027, -3.2779675, 3.3245530
1: -12.2045288, -9.0081425, -12.2239027, -8.9925785, -2.8955965, 2.9219131
2: -5.5950623, -2.2592869, -5.6190901, -2.2423472, -2.8527355, 2.8853383
3: -5.3431683, -1.5320253, -5.3687630, -1.5147655, -3.6170292, 3.6071582
4: -11.4873419, -7.5901585, -11.5182457, -7.5765052, -3.2275047, 3.4186034
5: -6.2218180, -3.1273403, -6.2622070, -3.0839958, -2.6715651, 2.6584663
6: -12.3938370, -8.7286739, -12.4149065, -8.6982517, -2.9992580, 2.9654822
7: -8.1438236, -4.6864657, -8.1627283, -4.6736841, -3.4701395, 3.4762626
8: 7.7507811, 10.0393200, 7.7413902, 10.0563517, -2.2494745, 2.2604208
9: -6.3052926, -2.8363342, -6.3417192, -2.8175600, -2.9553952, 2.9712348

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336823, upper bound: 1.1366256
time: 4.96 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1348719, upper bound: 1.1366255
time: 5.15 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.1948943, -1.9940510, -6.1949005, -1.9940403, -3.3525662, 3.3434968
1: -12.2345200, -8.9912529, -12.2345276, -8.9912539, -2.9102459, 2.9314260
2: -5.6206799, -2.2354584, -5.6206827, -2.2354534, -2.8864250, 2.8772473
3: -5.3708854, -1.5042269, -5.3708882, -1.5042179, -3.6458426, 3.6344719
4: -11.5237951, -7.5678372, -11.5237970, -7.5678267, -3.4363871, 3.4275279
5: -6.2900400, -3.0817838, -6.2900662, -3.0817828, -2.7242789, 2.7480552
6: -12.4278564, -8.6957150, -12.4278698, -8.6957130, -3.0314026, 3.0437446
7: -8.1703444, -4.6722717, -8.1703520, -4.6722708, -3.4980736, 3.4980803
8: 7.7388434, 10.0601482, 7.7388420, 10.0601511, -2.2858758, 2.2881560
9: -6.3480148, -2.8028774, -6.3480186, -2.8028624, -3.0185604, 3.0046439

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354365, upper bound: 1.1366253
time: 4.92 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366261, upper bound: 1.1366248
time: 5.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.28 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.28
Output dim: 8, lower bound: -1.1336823, upper bound: 1.1366256
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.28
Output dim: 8, lower bound: -1.1348719, upper bound: 1.1366255
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.28
Output dim: 8, lower bound: -1.1354365, upper bound: 1.1366253
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.28
Output dim: 8, lower bound: -1.1366261, upper bound: 1.1366248

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -6.1461077, -2.0184727, -6.1652107, -2.0046425, -3.2682829, 3.3021617
1: -12.1986008, -9.0083637, -12.2087288, -8.9931440, -2.8890638, 2.9064789
2: -5.5877204, -2.2606478, -5.6005068, -2.2461913, -2.8418598, 2.8651676
3: -5.3424435, -1.5333645, -5.3668900, -1.5184624, -3.6073980, 3.5995169
4: -11.4860210, -7.5943022, -11.5134392, -7.5869980, -3.2157269, 3.4099884
5: -6.2209196, -3.1286097, -6.2599115, -3.0873213, -2.6608400, 2.6493034
6: -12.3931160, -8.7293997, -12.4130402, -8.7002287, -2.9876795, 2.9548497
7: -8.1421175, -4.6891775, -8.1574087, -4.6805544, -3.4615631, 3.4682312
8: 7.7517381, 10.0373192, 7.7439709, 10.0512972, -2.2385044, 2.2508094
9: -6.3039069, -2.8372099, -6.3382139, -2.8197942, -2.9480515, 2.9632030

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1319940, upper bound: 1.1362006
time: 7.05 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336805, upper bound: 1.1366237
time: 5.69 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -6.1548586, -2.0182896, -6.1962094, -1.9576132, -3.3183384, 3.3264508
1: -12.2045135, -9.0081434, -12.2363319, -8.9757423, -2.9130936, 2.9341469
2: -5.5950470, -2.2592907, -5.6213193, -2.2052886, -2.8857465, 2.8809867
3: -5.3431664, -1.5320284, -5.3841805, -1.5112538, -3.6236477, 3.6171813
4: -11.4873409, -7.5901651, -11.5410995, -7.5706930, -3.2303724, 3.4411812
5: -6.2218170, -3.1273427, -6.2746334, -3.0816836, -2.6802797, 2.6674924
6: -12.3938379, -8.7286758, -12.4260979, -8.6961288, -3.0108585, 2.9696388
7: -8.1438179, -4.6864719, -8.1800394, -4.6733584, -3.4704595, 3.4935675
8: 7.7507849, 10.0393143, 7.7328386, 10.0643063, -2.2543836, 2.2759659
9: -6.3052893, -2.8363373, -6.3446789, -2.8025162, -2.9743323, 2.9712362

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1331893, upper bound: 1.1361990
time: 6.47 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1348701, upper bound: 1.1366238
time: 5.10 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.1861358, -1.9943411, -6.1727138, -1.9947829, -3.3428907, 3.3210602
1: -12.2285242, -8.9914761, -12.2193775, -8.9918175, -2.9035330, 2.9159670
2: -5.6133571, -2.2369642, -5.6021371, -2.2392864, -2.8755760, 2.8570843
3: -5.3701510, -1.5056870, -5.3690147, -1.5079048, -3.6362505, 3.6259365
4: -11.5218983, -7.5719767, -11.5189648, -7.5783176, -3.4240971, 3.4189043
5: -6.2891378, -3.0831041, -6.2877746, -3.0851064, -2.7135496, 2.7380288
6: -12.4271193, -8.6964951, -12.4260006, -8.6976976, -3.0197840, 3.0328012
7: -8.1682510, -4.6749830, -8.1650324, -4.6791415, -3.4891095, 3.4900494
8: 7.7398572, 10.0581493, 7.7414174, 10.0551004, -2.2744641, 2.2785249
9: -6.3466287, -2.8037522, -6.3445234, -2.8050888, -3.0112209, 2.9965687

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337277, upper bound: 1.1361978
time: 6.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354347, upper bound: 1.1366232
time: 7.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.1948724, -1.9940505, -6.2037096, -1.9477518, -3.3798037, 3.3458586
1: -12.2345057, -8.9912548, -12.2469416, -8.9744177, -2.9278030, 2.9437881
2: -5.6206646, -2.2354631, -5.6229391, -2.1984115, -2.8966098, 2.8731599
3: -5.3708858, -1.5042307, -5.3863230, -1.5007045, -3.6525574, 3.6441069
4: -11.5237913, -7.5678396, -11.5466557, -7.5620184, -3.4411430, 3.4502091
5: -6.2900372, -3.0817866, -6.3024750, -3.0794792, -2.7330055, 2.7519288
6: -12.4278545, -8.6957178, -12.4390354, -8.6935787, -3.0429831, 3.0457044
7: -8.1703415, -4.6722789, -8.1877127, -4.6719484, -3.4983931, 3.5154338
8: 7.7388468, 10.0601425, 7.7302904, 10.0681133, -2.2909012, 2.3036947
9: -6.3480110, -2.8028808, -6.3509531, -2.7878258, -3.0374870, 3.0045786

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349224, upper bound: 1.1362006
time: 5.90 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366243, upper bound: 1.1366236
time: 5.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.29 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.1319940, upper bound: 1.1362006
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.1336805, upper bound: 1.1366237
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.1331893, upper bound: 1.1361990
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.1348701, upper bound: 1.1366238
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.1337277, upper bound: 1.1361978
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.1354347, upper bound: 1.1366232
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.1349224, upper bound: 1.1362006
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 8, lower bound: -1.1366243, upper bound: 1.1366236

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1355772, -2.0422018, -6.1638308, -2.0140934, -3.2458143, 3.2767544
1: -12.1849995, -9.0147028, -12.2041330, -8.9952240, -2.8731070, 2.8945675
2: -5.5816593, -2.2815893, -5.5994568, -2.2541823, -2.8263264, 2.8422337
3: -5.3241215, -1.5422845, -5.3603125, -1.5202937, -3.5863781, 3.5830550
4: -11.4704733, -7.6031117, -11.5083008, -7.5887470, -3.1980524, 3.3943396
5: -6.2122455, -3.1499157, -6.2586737, -3.0957551, -2.6420975, 2.6253181
6: -12.3852978, -8.7422523, -12.4112053, -8.7051468, -2.9749246, 2.9397135
7: -8.1234407, -4.6925106, -8.1510639, -4.6814923, -3.4419484, 3.4585533
8: 7.7633648, 10.0300255, 7.7483010, 10.0493507, -2.2243147, 2.2363038
9: -6.2923007, -2.8498235, -6.3359709, -2.8240166, -2.9322424, 2.9475336

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1319940, upper bound: 1.1345690
time: 8.48 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1319940, upper bound: 1.1362006
time: 7.25 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.1461067, -2.0184765, -6.1652098, -2.0046432, -3.2682834, 3.2859478
1: -12.1985989, -9.0083647, -12.2087269, -8.9931431, -2.8890343, 2.9098306
2: -5.5877194, -2.2606521, -5.6005049, -2.2461905, -2.8418589, 2.8607488
3: -5.3424401, -1.5333650, -5.3668900, -1.5184619, -3.5999470, 3.5995164
4: -11.4860182, -7.5943046, -11.5134382, -7.5869970, -3.2139874, 3.4099884
5: -6.2209196, -3.1286130, -6.2599134, -3.0873227, -2.6608396, 2.6340041
6: -12.3931141, -8.7294016, -12.4130383, -8.7002296, -2.9880500, 2.9544392
7: -8.1421127, -4.6891775, -8.1574097, -4.6805544, -3.4615583, 3.4682322
8: 7.7517405, 10.0373182, 7.7439718, 10.0512972, -2.2384763, 2.2541258
9: -6.3039064, -2.8372102, -6.3382139, -2.8197932, -2.9480519, 2.9592447

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336805, upper bound: 1.1349755
time: 5.73 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336805, upper bound: 1.1366237
time: 5.92 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1443205, -2.0420184, -6.1948323, -1.9670396, -3.2910647, 3.3010416
1: -12.1909275, -9.0144768, -12.2317638, -8.9778299, -2.8971372, 2.9222798
2: -5.5889816, -2.2802279, -5.6202583, -2.2133040, -2.8660784, 2.8580351
3: -5.3248453, -1.5409489, -5.3776588, -1.5130858, -3.6026487, 3.6007710
4: -11.4718189, -7.5989757, -11.5359869, -7.5724592, -3.2127323, 3.4255571
5: -6.2131395, -3.1486368, -6.2733908, -3.0900946, -2.6615601, 2.6435204
6: -12.3860159, -8.7415228, -12.4242592, -8.7010365, -2.9981275, 2.9544892
7: -8.1251507, -4.6898055, -8.1737337, -4.6742954, -3.4508553, 3.4839282
8: 7.7624121, 10.0320129, 7.7371597, 10.0623684, -2.2401686, 2.2614818
9: -6.2936754, -2.8489611, -6.3424306, -2.8066485, -2.9586086, 2.9555492

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1331893, upper bound: 1.1345681
time: 9.54 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1331893, upper bound: 1.1361990
time: 8.12 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.1548553, -2.0182924, -6.1962094, -1.9576139, -3.3133368, 3.3102493
1: -12.2045116, -9.0081434, -12.2363300, -8.9757423, -2.9130650, 2.9374981
2: -5.5950470, -2.2592936, -5.6213183, -2.2052879, -2.8830132, 2.8765664
3: -5.3431654, -1.5320289, -5.3841805, -1.5112536, -3.6161966, 3.6171799
4: -11.4873371, -7.5901661, -11.5411005, -7.5706930, -3.2285490, 3.4411802
5: -6.2218161, -3.1273460, -6.2746334, -3.0816832, -2.6802788, 2.6521921
6: -12.3938351, -8.7286777, -12.4260979, -8.6961308, -3.0112305, 2.9692273
7: -8.1438169, -4.6864729, -8.1800394, -4.6733565, -3.4704604, 3.4935665
8: 7.7507877, 10.0393124, 7.7328386, 10.0643082, -2.2543535, 2.2792823
9: -6.3052878, -2.8363380, -6.3446779, -2.8025165, -2.9743333, 2.9672780

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1348701, upper bound: 1.1349758
time: 6.04 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1348701, upper bound: 1.1366238
time: 5.94 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.1755376, -2.0182810, -6.1713219, -2.0042276, -3.3203354, 3.2953815
1: -12.2148027, -8.9978209, -12.2147989, -8.9938974, -2.8873343, 2.9040208
2: -5.6072669, -2.2582211, -5.6010838, -2.2472732, -2.8577604, 2.8340960
3: -5.3517694, -1.5148952, -5.3624339, -1.5097408, -3.6152296, 3.6093125
4: -11.5051823, -7.5807548, -11.5138321, -7.5800700, -3.4050131, 3.4033155
5: -6.2804818, -3.1045227, -6.2865343, -3.0935397, -2.6948271, 2.7139378
6: -12.4193201, -8.7094393, -12.4241648, -8.7026138, -3.0069523, 3.0175638
7: -8.1487770, -4.6783342, -8.1587038, -4.6800857, -3.4686913, 3.4803696
8: 7.7516236, 10.0508127, 7.7457414, 10.0531445, -2.2599473, 2.2639875
9: -6.3349466, -2.8163984, -6.3422589, -2.8093109, -2.9953232, 2.9808393

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337275, upper bound: 1.1344677
time: 4.97 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337278, upper bound: 1.1344684
time: 6.54 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.1861339, -1.9943452, -6.1727152, -1.9947822, -3.3428907, 3.3044658
1: -12.2285213, -8.9914761, -12.2193747, -8.9918175, -2.9034853, 2.9193182
2: -5.6133566, -2.2369673, -5.6021366, -2.2392859, -2.8747129, 2.8545995
3: -5.3701482, -1.5056872, -5.3690138, -1.5079067, -3.6288004, 3.6259356
4: -11.5218954, -7.5719781, -11.5189648, -7.5783176, -3.4218655, 3.4189024
5: -6.2891364, -3.0831113, -6.2877741, -3.0851059, -2.7135487, 2.7233915
6: -12.4271202, -8.6964970, -12.4259987, -8.6976948, -3.0201540, 3.0327930
7: -8.1682463, -4.6749830, -8.1650314, -4.6791415, -3.4891047, 3.4900484
8: 7.7398596, 10.0581455, 7.7414179, 10.0551004, -2.2744174, 2.2818413
9: -6.3466291, -2.8037558, -6.3445225, -2.8050890, -3.0112181, 2.9927359

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354345, upper bound: 1.1348689
time: 7.59 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354348, upper bound: 1.1348688
time: 9.37 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.1842699, -2.0179913, -6.2023163, -1.9571784, -3.3524270, 3.3201809
1: -12.2207975, -8.9975929, -12.2423954, -8.9765053, -2.9115834, 2.9319401
2: -5.6145763, -2.2567101, -5.6218762, -2.2064168, -2.8769541, 2.8501544
3: -5.3524995, -1.5134373, -5.3797979, -1.5025434, -3.6315556, 3.6275349
4: -11.5071135, -7.5766182, -11.5415516, -7.5637827, -3.4221144, 3.4346771
5: -6.2813768, -3.1031904, -6.3012323, -3.0878897, -2.7143073, 2.7278614
6: -12.4200506, -8.7086544, -12.4371967, -8.6984854, -3.0283599, 3.0304413
7: -8.1508808, -4.6756315, -8.1814213, -4.6728902, -3.4779906, 3.5057898
8: 7.7506104, 10.0527983, 7.7346048, 10.0661650, -2.2763700, 2.2891784
9: -6.3363209, -2.8155303, -6.3486757, -2.7919579, -3.0216756, 2.9888296

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349222, upper bound: 1.1344704
time: 6.14 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1349225, upper bound: 1.1344704
time: 5.88 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.1948695, -1.9940548, -6.2037096, -1.9477532, -3.3747854, 3.3292823
1: -12.2345009, -8.9912567, -12.2469435, -8.9744186, -2.9277554, 2.9471388
2: -5.6206646, -2.2354665, -5.6229391, -2.1984122, -2.8938723, 2.8706775
3: -5.3708820, -1.5042300, -5.3863220, -1.5007036, -3.6451063, 3.6441059
4: -11.5237865, -7.5678449, -11.5466557, -7.5620184, -3.4389133, 3.4502096
5: -6.2900372, -3.0817924, -6.3024755, -3.0794802, -2.7330046, 2.7372906
6: -12.4278545, -8.6957207, -12.4390373, -8.6935806, -3.0421448, 3.0455041
7: -8.1703367, -4.6722794, -8.1877136, -4.6719484, -3.4983883, 3.5154343
8: 7.7388496, 10.0601416, 7.7302895, 10.0681133, -2.2908545, 2.3070111
9: -6.3480105, -2.8028817, -6.3509531, -2.7878265, -3.0374866, 3.0007467

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366243, upper bound: 1.1348694
time: 5.75 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1366244, upper bound: 1.1348693
time: 5.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.22 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1319940, upper bound: 1.1345690
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1319940, upper bound: 1.1362006
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1336805, upper bound: 1.1349755
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1336805, upper bound: 1.1366237
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1331893, upper bound: 1.1345681
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1331893, upper bound: 1.1361990
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1348701, upper bound: 1.1349758
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1348701, upper bound: 1.1366238
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1337275, upper bound: 1.1344677
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1337278, upper bound: 1.1344684
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1354345, upper bound: 1.1348689
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1354348, upper bound: 1.1348688
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1349222, upper bound: 1.1344704
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1349225, upper bound: 1.1344704
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1366243, upper bound: 1.1348694
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.22
Output dim: 8, lower bound: -1.1366244, upper bound: 1.1348693

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1355772, -2.0422018, -6.1313019, -2.0282092, -3.2588716, 3.2436833
1: -12.1849995, -9.0147028, -12.1849394, -9.0107851, -2.8547573, 2.8501558
2: -5.5816593, -2.2815893, -5.5754323, -2.2707472, -2.8310490, 2.8142524
3: -5.3241215, -1.5422845, -5.3347573, -1.5372398, -3.5433044, 3.5489659
4: -11.4704733, -7.6031117, -11.4788685, -7.6024227, -3.1811209, 3.1867642
5: -6.2122455, -3.1499157, -6.2182803, -3.1389580, -2.5904045, 2.5863934
6: -12.3852978, -8.7422523, -12.3901615, -8.7354364, -2.9206610, 2.9186959
7: -8.1234407, -4.6925106, -8.1331224, -4.6942701, -3.4291706, 3.4406118
8: 7.7633648, 10.0300255, 7.7575421, 10.0323257, -2.2037883, 2.2062416
9: -6.2923007, -2.8498235, -6.2995672, -2.8427825, -2.9117546, 2.9111676

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1312054, upper bound: 1.1345663
time: 5.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1319924, upper bound: 1.1345666
time: 10.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1355772, -2.0422018, -6.1711307, -2.0047169, -3.2552657, 3.2834058
1: -12.1849995, -9.0147028, -12.2146511, -8.9942474, -2.8725128, 2.9057941
2: -5.5816593, -2.2815893, -5.6007562, -2.2474232, -2.8359232, 2.8400526
3: -5.3241215, -1.5422845, -5.3621831, -1.5097919, -3.5933714, 3.5800862
4: -11.4704733, -7.6031117, -11.5137386, -7.5803013, -3.2058477, 3.3966942
5: -6.2122455, -3.1499157, -6.2863731, -3.0938439, -2.6363711, 2.6536241
6: -12.3852978, -8.7422523, -12.4240456, -8.7028275, -2.9747963, 2.9530902
7: -8.1234407, -4.6925106, -8.1585979, -4.6804171, -3.4430237, 3.4660873
8: 7.7633648, 10.0300255, 7.7457957, 10.0529118, -2.2280469, 2.2359707
9: -6.2923007, -2.8498235, -6.3420029, -2.8094070, -2.9468117, 2.9527617

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1312056, upper bound: 1.1361962
time: 5.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1319924, upper bound: 1.1361965
time: 7.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1461067, -2.0184765, -6.1326585, -2.0187635, -3.2821198, 3.2528572
1: -12.1985989, -9.0083647, -12.1895514, -9.0087070, -2.8706846, 2.8641329
2: -5.5877194, -2.2606521, -5.5764732, -2.2627535, -2.8468494, 2.8327665
3: -5.3424401, -1.5333650, -5.3413157, -1.5354049, -3.5582705, 3.5654302
4: -11.4860182, -7.5943046, -11.4839697, -7.6006584, -3.1970520, 3.2026873
5: -6.2209196, -3.1286130, -6.2195339, -3.1305332, -2.6091919, 2.5950890
6: -12.3931141, -8.7294016, -12.3920021, -8.7305222, -2.9332533, 2.9334273
7: -8.1421127, -4.6891775, -8.1394825, -4.6933384, -3.4487743, 3.4503050
8: 7.7517405, 10.0373182, 7.7532110, 10.0342617, -2.2179375, 2.2218909
9: -6.3039064, -2.8372102, -6.3017874, -2.8385580, -2.9275608, 2.9228525

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1329225, upper bound: 1.1349746
time: 5.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336789, upper bound: 1.1349740
time: 6.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1461067, -2.0184765, -6.1725230, -1.9952760, -3.2777281, 3.2926140
1: -12.1985989, -9.0083647, -12.2192259, -8.9921665, -2.8884478, 2.9210229
2: -5.5877194, -2.2606521, -5.6018105, -2.2394352, -2.8514404, 2.8585725
3: -5.3424401, -1.5333650, -5.3687634, -1.5079589, -3.6069202, 3.5965633
4: -11.4860182, -7.5943046, -11.5188694, -7.5785503, -3.2217703, 3.4123611
5: -6.2209196, -3.1286130, -6.2876148, -3.0854139, -2.6568236, 2.6623106
6: -12.3931141, -8.7294016, -12.4258776, -8.6979103, -2.9886513, 2.9678164
7: -8.1421127, -4.6891775, -8.1649275, -4.6794739, -3.4626389, 3.4757500
8: 7.7517405, 10.0373182, 7.7414713, 10.0548687, -2.2422185, 2.2537630
9: -6.3039064, -2.8372102, -6.3442669, -2.8051825, -2.9626198, 2.9644928

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1329225, upper bound: 1.1366222
time: 6.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1336789, upper bound: 1.1366222
time: 5.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1443205, -2.0420184, -6.1623201, -1.9814258, -3.3116522, 3.2679691
1: -12.1909275, -9.0144768, -12.2123442, -8.9933872, -2.8787870, 2.8781290
2: -5.5889816, -2.2802279, -5.5962439, -2.2301848, -2.8709726, 2.8300638
3: -5.3248453, -1.5409489, -5.3520288, -1.5303955, -3.5582533, 3.5666103
4: -11.4718189, -7.5989757, -11.5052185, -7.5861101, -3.1958170, 3.2170811
5: -6.2131395, -3.1486368, -6.2330117, -3.1334147, -2.6092310, 2.6046066
6: -12.3860159, -8.7415228, -12.4031916, -8.7314262, -2.9433379, 2.9333987
7: -8.1251507, -4.6898055, -8.1547508, -4.6870775, -3.4380732, 3.4649453
8: 7.7624121, 10.0320129, 7.7465248, 10.0453377, -2.2196455, 2.2304921
9: -6.2936754, -2.8489611, -6.3060875, -2.8253868, -2.9380341, 2.9192309

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1324189, upper bound: 1.1345664
time: 5.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1331876, upper bound: 1.1345693
time: 6.40 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1443205, -2.0420184, -6.2021275, -1.9576693, -3.2918186, 3.3077035
1: -12.1909275, -9.0144768, -12.2422447, -8.9768534, -2.8965416, 2.9334631
2: -5.5889816, -2.2802279, -5.6215506, -2.2065661, -2.8660374, 2.8558459
3: -5.3248453, -1.5409489, -5.3795428, -1.5025961, -3.6096992, 3.5978699
4: -11.4718189, -7.5989757, -11.5414562, -7.5640173, -3.2205229, 3.4279923
5: -6.2131395, -3.1486368, -6.3010697, -3.0881948, -2.6471553, 2.6713259
6: -12.3860159, -8.7415228, -12.4370785, -8.6986961, -2.9871097, 2.9678345
7: -8.1251507, -4.6898055, -8.1813183, -4.6732221, -3.4519286, 3.4915128
8: 7.7624121, 10.0320129, 7.7346582, 10.0659332, -2.2438903, 2.2611213
9: -6.2936754, -2.8489611, -6.3484201, -2.7920504, -2.9731655, 2.9607391

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1324189, upper bound: 1.1361957
time: 5.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1331876, upper bound: 1.1361969
time: 6.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.1548553, -2.0182924, -6.1636815, -1.9719949, -3.3339801, 3.2771587
1: -12.2045116, -9.0081434, -12.2169304, -8.9913034, -2.8947153, 2.8920288
2: -5.5950470, -2.2592936, -5.5972958, -2.2221582, -2.8881531, 2.8485923
3: -5.3431654, -1.5320289, -5.3585348, -1.5285628, -3.5731974, 3.5830221
4: -11.4873371, -7.5901661, -11.5102987, -7.5843296, -3.2116289, 3.2328906
5: -6.2218161, -3.1273460, -6.2342668, -3.1250114, -2.6279931, 2.6132889
6: -12.3938351, -8.7286777, -12.4050369, -8.7265224, -2.9559073, 2.9481382
7: -8.1438169, -4.6864729, -8.1610928, -4.6861429, -3.4576740, 3.4746199
8: 7.7507877, 10.0393124, 7.7421989, 10.0472670, -2.2338181, 2.2461307
9: -6.3052878, -2.8363380, -6.3083143, -2.8212564, -2.9537525, 2.9309363

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1341211, upper bound: 1.1349745
time: 5.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1348685, upper bound: 1.1349744
time: 5.42 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1548553, -2.0182924, -6.2035227, -1.9482458, -3.3140941, 3.3169274
1: -12.2045116, -9.0081434, -12.2467947, -8.9747686, -2.9124765, 2.9486470
2: -5.5950470, -2.2592936, -5.6226115, -2.1985610, -2.8829675, 2.8743801
3: -5.3431654, -1.5320289, -5.3860674, -1.5007570, -3.6232262, 3.6142941
4: -11.4873371, -7.5901661, -11.5465622, -7.5622520, -3.2363281, 3.4436002
5: -6.2218161, -3.1273460, -6.3023124, -3.0797868, -2.6675425, 2.6799362
6: -12.3938351, -8.7286777, -12.4389133, -8.6937943, -3.0008764, 2.9825678
7: -8.1438169, -4.6864729, -8.1876078, -4.6722803, -3.4715366, 3.5011349
8: 7.7507877, 10.0393124, 7.7303433, 10.0678816, -2.2580838, 2.2788920
9: -6.3052878, -2.8363380, -6.3506975, -2.7879219, -2.9888883, 2.9724879

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1341211, upper bound: 1.1366223
time: 6.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1348685, upper bound: 1.1366223
time: 5.93 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1755376, -2.0182810, -6.1313019, -2.0282092, -3.3204389, 3.2401743
1: -12.2148027, -8.9978209, -12.1849394, -9.0107851, -2.9105921, 2.8508182
2: -5.6072669, -2.2582211, -5.5754323, -2.2707472, -2.8503799, 2.8141527
3: -5.3517694, -1.5148952, -5.3347573, -1.5372398, -3.5640163, 3.5990624
4: -11.5051823, -7.5807548, -11.4788685, -7.6024227, -3.3898468, 3.2060313
5: -6.2804818, -3.1045227, -6.2182803, -3.1389580, -2.6547942, 2.6468482
6: -12.4193201, -8.7094393, -12.3901615, -8.7354364, -2.9551535, 2.9832635
7: -8.1487770, -4.6783342, -8.1331224, -4.6942701, -3.4545069, 3.4547882
8: 7.7516236, 10.0508127, 7.7575421, 10.0323257, -2.2343950, 2.2334156
9: -6.3349466, -2.8163984, -6.2995672, -2.8427825, -2.9601355, 2.9462652

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1329409, upper bound: 1.1344666
time: 6.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337259, upper bound: 1.1344662
time: 7.03 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1755376, -2.0182810, -6.1713161, -2.0042372, -3.3112574, 3.2953749
1: -12.2148027, -8.9978209, -12.2147942, -8.9938993, -2.8873329, 2.8827043
2: -5.6072669, -2.2582211, -5.6010828, -2.2472787, -2.8577580, 2.8432665
3: -5.3517694, -1.5148952, -5.3624320, -1.5097513, -3.6038570, 3.6093092
4: -11.5051823, -7.5807548, -11.5138264, -7.5800810, -3.3962622, 3.4033074
5: -6.2804818, -3.1045227, -6.2865067, -3.0935402, -2.6948261, 2.6907926
6: -12.4193201, -8.7094393, -12.4241524, -8.7026176, -3.0069518, 3.0052204
7: -8.1487770, -4.6783342, -8.1586962, -4.6800857, -3.4686913, 3.4803619
8: 7.7516236, 10.0508127, 7.7457442, 10.0531387, -2.2599425, 2.2616327
9: -6.3349466, -2.8163984, -6.3422546, -2.8093290, -2.9813995, 2.9808340

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1329411, upper bound: 1.1344665
time: 6.55 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1337262, upper bound: 1.1344659
time: 6.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1861339, -1.9943452, -6.1326585, -2.0187635, -3.3437486, 3.2492099
1: -12.2285213, -8.9914761, -12.1895514, -9.0087070, -2.9265451, 2.8648171
2: -5.6133566, -2.2369673, -5.5764732, -2.2627535, -2.8661938, 2.8346519
3: -5.3701482, -1.5056872, -5.3413157, -1.5354049, -3.5790005, 3.6156611
4: -11.5218954, -7.5719781, -11.4839697, -7.6006584, -3.4067297, 3.2218766
5: -6.2891364, -3.0831113, -6.2195339, -3.1305332, -2.6753120, 2.6563110
6: -12.4271202, -8.6964970, -12.3920021, -8.7305222, -2.9677382, 2.9984999
7: -8.1682463, -4.6749830, -8.1394825, -4.6933384, -3.4749079, 3.4644995
8: 7.7398596, 10.0581455, 7.7532110, 10.0342617, -2.2487187, 2.2491162
9: -6.3466291, -2.8037558, -6.3017874, -2.8385580, -2.9760299, 2.9581218

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1346381, upper bound: 1.1348681
time: 6.03 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354330, upper bound: 1.1348673
time: 7.74 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1861339, -1.9943452, -6.1727061, -1.9947920, -3.3338122, 3.3044600
1: -12.2285213, -8.9914761, -12.2193680, -8.9918194, -2.9034848, 2.8980513
2: -5.6133566, -2.2369673, -5.6021347, -2.2392917, -2.8747110, 2.8637729
3: -5.3701482, -1.5056872, -5.3690124, -1.5079162, -3.6173773, 3.6259322
4: -11.5218954, -7.5719781, -11.5189590, -7.5783281, -3.4130402, 3.4188938
5: -6.2891364, -3.0831113, -6.2877440, -3.0851092, -2.7135463, 2.7003298
6: -12.4271202, -8.6964970, -12.4259853, -8.6976995, -3.0201511, 3.0204506
7: -8.1682463, -4.6749830, -8.1650248, -4.6791439, -3.4891024, 3.4900417
8: 7.7398596, 10.0581455, 7.7414222, 10.0550957, -2.2744126, 2.2795184
9: -6.3466291, -2.8037558, -6.3445177, -2.8051043, -2.9972954, 2.9927320

Time for backsubstitution: 15.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1346383, upper bound: 1.1348682
time: 4.96 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.1354332, upper bound: 1.1348673
time: 7.54 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1842699, -2.0179913, -6.1623201, -1.9814258, -3.3568177, 3.2651920
1: -12.2207975, -8.9975929, -12.2123442, -8.9933872, -2.9347992, 2.8787999
2: -5.6145763, -2.2567101, -5.5962439, -2.2301848, -2.8749352, 2.8302302
3: -5.3524995, -1.5134373, -5.3520288, -1.5303955, -3.5789614, 3.6172824
4: -11.5071135, -7.5766182, -11.5052185, -7.5861101, -3.4070835, 3.2363472
5: -6.2813768, -3.1031904, -6.2330117, -3.1334147, -2.6652966, 2.6608050
6: -12.4200506, -8.7086544, -12.4031916, -8.7314262, -2.9718971, 2.9961271
7: -8.1508808, -4.6756315, -8.1547508, -4.6870775, -3.4638033, 3.4791193
8: 7.7506104, 10.0527983, 7.7465248, 10.0453377, -2.2508044, 2.2576888
9: -6.3363209, -2.8155303, -6.3060875, -2.8253868, -2.9864163, 2.9543476

Time for backsubstitution: 14.94 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.2884626388549805
rel_dist={8: [-1.136630455303056, 1.136628973869513]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392416, upper bound: 1.0405807
time: 8.59 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405782, upper bound: 1.0405772
time: 5.06 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.89
Output dim: 8, lower bound: -1.0392416, upper bound: 1.0405807
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.89
Output dim: 8, lower bound: -1.0405782, upper bound: 1.0405772

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -6.1548796, -2.0182879, -6.1859760, -2.0057130, -3.1569142, 3.2016010
1: -12.2045288, -9.0081425, -12.2218904, -8.9927702, -2.7952943, 2.8191175
2: -5.5950623, -2.2592869, -5.6188431, -2.2436471, -2.7779942, 2.8107119
3: -5.3431683, -1.5320253, -5.3683949, -1.5167835, -3.5291557, 3.5204477
4: -11.4873419, -7.5901585, -11.5171738, -7.5781326, -3.1162291, 3.3105278
5: -6.2218180, -3.1273403, -6.2568631, -3.0843782, -2.5984759, 2.5789480
6: -12.3938370, -8.7286739, -12.4124298, -8.6987152, -2.8914957, 2.8549905
7: -8.1438236, -4.6864657, -8.1612892, -4.6738939, -3.4278345, 3.4748235
8: 7.7507811, 10.0393200, 7.7418709, 10.0556526, -2.1846356, 2.1970520
9: -6.3052926, -2.8363342, -6.3405228, -2.8203783, -2.8691540, 2.8869252

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383416, upper bound: 1.0405794
time: 7.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392402, upper bound: 1.0405770
time: 7.10 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -6.1948943, -1.9940510, -6.1948986, -1.9940438, -3.2329779, 3.2234550
1: -12.2345200, -8.9912529, -12.2345257, -8.9912539, -2.8083181, 2.8305678
2: -5.6206799, -2.2354584, -5.6206827, -2.2354546, -2.8121109, 2.8028660
3: -5.3708854, -1.5042269, -5.3708873, -1.5042176, -3.5584297, 3.5457087
4: -11.5237951, -7.5678372, -11.5237980, -7.5678282, -3.3288908, 3.3186946
5: -6.2900400, -3.0817838, -6.2900624, -3.0817833, -2.6503744, 2.6744056
6: -12.4278564, -8.6957150, -12.4278669, -8.6957169, -2.9230046, 2.9345183
7: -8.1703444, -4.6722717, -8.1703510, -4.6722732, -3.4980712, 3.4980793
8: 7.7388434, 10.0601482, 7.7388420, 10.0601492, -2.2225471, 2.2251821
9: -6.3480148, -2.8028774, -6.3480177, -2.8028653, -2.9353275, 2.9207163

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396746, upper bound: 1.0405762
time: 11.21 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405768, upper bound: 1.0405759
time: 5.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 31.76 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.76
Output dim: 8, lower bound: -1.0383416, upper bound: 1.0405794
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.76
Output dim: 8, lower bound: -1.0392402, upper bound: 1.0405770
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.76
Output dim: 8, lower bound: -1.0396746, upper bound: 1.0405762
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.76
Output dim: 8, lower bound: -1.0405768, upper bound: 1.0405759

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -6.1432714, -2.0185347, -6.1637802, -2.0064535, -3.1444116, 3.1791320
1: -12.1966906, -9.0084362, -12.2067108, -8.9933338, -2.7868538, 2.8035994
2: -5.5853481, -2.2610915, -5.6002488, -2.2474926, -2.7647190, 2.7901583
3: -5.3422046, -1.5337963, -5.3665223, -1.5204854, -3.5188093, 3.5119472
4: -11.4855928, -7.5956411, -11.5123730, -7.5886259, -3.1040096, 3.3005848
5: -6.2206268, -3.1290178, -6.2545671, -3.0877042, -2.5869684, 2.5687680
6: -12.3928843, -8.7296352, -12.4105673, -8.7006884, -2.8789682, 2.8433032
7: -8.1415634, -4.6900539, -8.1559687, -4.6807652, -3.4168186, 3.4659147
8: 7.7520475, 10.0366726, 7.7444558, 10.0505981, -2.1729589, 2.1862922
9: -6.3034582, -2.8374932, -6.3370152, -2.8226123, -2.8610253, 2.8782730

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0369538, upper bound: 1.0400851
time: 6.55 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383401, upper bound: 1.0405748
time: 6.08 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -6.1548524, -2.0182915, -6.1947808, -1.9594231, -3.1959143, 3.2019329
1: -12.2045088, -9.0081434, -12.2343206, -8.9759350, -2.8127899, 2.8311410
2: -5.5950451, -2.2592912, -5.6210647, -2.2065840, -2.8097639, 2.8045049
3: -5.3431664, -1.5320292, -5.3838105, -1.5132725, -3.5351973, 3.5304666
4: -11.4873400, -7.5901661, -11.5400267, -7.5723209, -3.1184883, 3.3331003
5: -6.2218142, -3.1273422, -6.2692919, -3.0820618, -2.6064024, 2.5879793
6: -12.3938370, -8.7286768, -12.4236279, -8.6965933, -2.9023495, 2.8591557
7: -8.1438208, -4.6864729, -8.1785870, -4.6735697, -3.4270668, 3.4921141
8: 7.7507858, 10.0393124, 7.7333207, 10.0636082, -2.1895466, 2.2121010
9: -6.3052902, -2.8363371, -6.3434877, -2.8053339, -2.8877573, 2.8869300

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0378024, upper bound: 1.0400821
time: 7.51 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392387, upper bound: 1.0405759
time: 11.59 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -6.1833038, -1.9944360, -6.1727118, -1.9947848, -3.2204862, 3.2008939
1: -12.2265902, -8.9915485, -12.2193756, -8.9918194, -2.7996569, 2.8150225
2: -5.6109905, -2.2374544, -5.6021357, -2.2392883, -2.7988625, 2.7822676
3: -5.3699112, -1.5061586, -5.3690143, -1.5079076, -3.5481186, 3.5362263
4: -11.5212812, -7.5733180, -11.5189638, -7.5783191, -3.3160152, 3.3087363
5: -6.2888451, -3.0835304, -6.2877674, -3.0851073, -2.6388588, 2.6633220
6: -12.4268837, -8.6967487, -12.4259958, -8.6976986, -2.9104266, 2.9218903
7: -8.1675720, -4.6758599, -8.1650314, -4.6791420, -3.4884300, 3.4891715
8: 7.7401853, 10.0575047, 7.7414179, 10.0550995, -2.2103567, 2.2144029
9: -6.3461838, -2.8040364, -6.3445230, -2.8050928, -2.9272051, 2.9120154

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382989, upper bound: 1.0400816
time: 6.38 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396731, upper bound: 1.0405780
time: 6.59 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -6.1948686, -1.9940515, -6.2037096, -1.9477530, -3.2584600, 3.2243066
1: -12.2345028, -8.9912558, -12.2469416, -8.9744186, -2.8258734, 2.8427110
2: -5.6206646, -2.2354617, -5.6229386, -2.1984119, -2.8210263, 2.7969823
3: -5.3708849, -1.5042315, -5.3863230, -1.5007052, -3.5645676, 3.5553432
4: -11.5237885, -7.5678439, -11.5466547, -7.5620208, -3.3332872, 3.3413758
5: -6.2900381, -3.0817866, -6.3024673, -3.0794802, -2.6583128, 2.6782780
6: -12.4278526, -8.6957169, -12.4390335, -8.6935825, -2.9314728, 2.9356523
7: -8.1703396, -4.6722784, -8.1877117, -4.6719465, -3.4983931, 3.5154333
8: 7.7388468, 10.0601416, 7.7302904, 10.0681124, -2.2275715, 2.2402253
9: -6.3480105, -2.8028808, -6.3509536, -2.7878315, -2.9539194, 2.9206524

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391536, upper bound: 1.0400824
time: 6.46 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405752, upper bound: 1.0405743
time: 6.08 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.77 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.77
Output dim: 8, lower bound: -1.0369538, upper bound: 1.0400851
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.77
Output dim: 8, lower bound: -1.0383401, upper bound: 1.0405748
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.77
Output dim: 8, lower bound: -1.0378024, upper bound: 1.0400821
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.77
Output dim: 8, lower bound: -1.0392387, upper bound: 1.0405759
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.77
Output dim: 8, lower bound: -1.0382989, upper bound: 1.0400816
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.77
Output dim: 8, lower bound: -1.0396731, upper bound: 1.0405780
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.77
Output dim: 8, lower bound: -1.0391536, upper bound: 1.0400824
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.77
Output dim: 8, lower bound: -1.0405752, upper bound: 1.0405743

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -6.1327438, -2.0422614, -6.1621170, -2.0178668, -3.1197314, 3.1533742
1: -12.1830835, -9.0147772, -12.2011642, -8.9958467, -2.7705693, 2.7908778
2: -5.5792904, -2.2820323, -5.5989847, -2.2571487, -2.7474966, 2.7669511
3: -5.3238859, -1.5427186, -5.3585815, -1.5227044, -3.4973001, 3.4941487
4: -11.4700375, -7.6044517, -11.5061731, -7.5907454, -3.0858288, 3.2838454
5: -6.2119560, -3.1503277, -6.2530718, -3.0978904, -2.5664301, 2.5444651
6: -12.3850632, -8.7424889, -12.4083424, -8.7066288, -2.8651252, 2.8278461
7: -8.1228848, -4.6933880, -8.1483126, -4.6818976, -3.3953543, 3.4549246
8: 7.7636719, 10.0293818, 7.7496824, 10.0482492, -2.1584010, 2.1706073
9: -6.2918558, -2.8501034, -6.3343034, -2.8277178, -2.8443160, 2.8621058

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0369523, upper bound: 1.0392694
time: 8.92 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0369523, upper bound: 1.0400809
time: 14.32 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -6.1432700, -2.0185380, -6.1637802, -2.0064564, -3.1444087, 3.1617880
1: -12.1966877, -9.0084362, -12.2067108, -8.9933329, -2.7867465, 2.8068070
2: -5.5853481, -2.2610931, -5.6002493, -2.2474940, -2.7647181, 2.7855182
3: -5.3422027, -1.5337958, -5.3665214, -1.5204856, -3.5109873, 3.5119452
4: -11.4855890, -7.5956450, -11.5123749, -7.5886250, -3.1017303, 3.3005834
5: -6.2206278, -3.1290236, -6.2545662, -3.0877051, -2.5869670, 2.5527034
6: -12.3928804, -8.7296381, -12.4105644, -8.7006865, -2.8793225, 2.8428693
7: -8.1415596, -4.6900539, -8.1559687, -4.6807661, -3.4218998, 3.4659147
8: 7.7520504, 10.0366707, 7.7444553, 10.0505981, -2.1728573, 2.1894677
9: -6.3034577, -2.8374946, -6.3370147, -2.8226132, -2.8610253, 2.8741131

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383386, upper bound: 1.0398448
time: 8.58 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383386, upper bound: 1.0405733
time: 6.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -6.1443157, -2.0420189, -6.1931171, -1.9708107, -3.1673760, 3.1761761
1: -12.1909266, -9.0144768, -12.2288074, -8.9784536, -2.7965078, 2.8184352
2: -5.5889778, -2.2802272, -5.6197863, -2.2162659, -2.7888083, 2.7812786
3: -5.3248434, -1.5409513, -5.3759365, -1.5154917, -3.5137157, 3.5127316
4: -11.4718189, -7.5989780, -11.5338573, -7.5744591, -3.1003523, 3.3163896
5: -6.2131405, -3.1486382, -6.2677898, -3.0922213, -2.5858922, 2.5636940
6: -12.3860140, -8.7415228, -12.4214048, -8.7025185, -2.8885307, 2.8436871
7: -8.1251488, -4.6898055, -8.1709776, -4.6747017, -3.4055519, 3.4811721
8: 7.7624121, 10.0320110, 7.7385378, 10.0612659, -2.1749587, 2.1964421
9: -6.2936745, -2.8489611, -6.3407717, -2.8103237, -2.8711486, 2.8707428

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0378008, upper bound: 1.0392687
time: 6.06 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0378008, upper bound: 1.0400805
time: 8.08 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -6.1548529, -2.0182939, -6.1947803, -1.9594254, -3.1909151, 3.1846061
1: -12.2045078, -9.0081434, -12.2343178, -8.9759350, -2.8126812, 2.8343496
2: -5.5950460, -2.2592955, -5.6210651, -2.2065849, -2.8070292, 2.7998633
3: -5.3431630, -1.5320280, -5.3838077, -1.5132742, -3.5273743, 3.5304646
4: -11.4873371, -7.5901642, -11.5400257, -7.5723219, -3.1161470, 3.3330998
5: -6.2218151, -3.1273484, -6.2692909, -3.0820627, -2.6064005, 2.5719137
6: -12.3938370, -8.7286806, -12.4236269, -8.6965942, -2.9027023, 2.8587236
7: -8.1438169, -4.6864719, -8.1785870, -4.6735697, -3.4320774, 3.4921150
8: 7.7507887, 10.0393114, 7.7333217, 10.0636072, -2.1894436, 2.2152765
9: -6.3052869, -2.8363383, -6.3434887, -2.8053343, -2.8877544, 2.8827734

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392371, upper bound: 1.0398425
time: 9.45 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392372, upper bound: 1.0405735
time: 8.63 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -6.1727061, -2.0183768, -6.1710291, -2.0061893, -3.1957183, 3.1748633
1: -12.2128649, -8.9978962, -12.2138548, -8.9943295, -2.7831430, 2.8022590
2: -5.6049051, -2.2587109, -5.6008635, -2.2489362, -2.7785058, 2.7590175
3: -5.3515325, -1.5153656, -5.3610697, -1.5101340, -3.5266094, 3.5182848
4: -11.5045538, -7.5820928, -11.5127716, -7.5804377, -3.2964497, 3.2920656
5: -6.2801886, -3.1049509, -6.2862668, -3.0952911, -2.6183434, 2.6389098
6: -12.4190798, -8.7096939, -12.4237738, -8.7036352, -2.8965106, 2.9063125
7: -8.1480932, -4.6792130, -8.1573982, -4.6802797, -3.4678135, 3.4781852
8: 7.7519522, 10.0501709, 7.7466373, 10.0527344, -2.1954589, 2.1986883
9: -6.3345041, -2.8166778, -6.3417816, -2.8101935, -2.9104061, 2.8957801

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382973, upper bound: 1.0392683
time: 5.42 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382996, upper bound: 1.0400804
time: 5.91 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -6.1833034, -1.9944420, -6.1727133, -1.9947858, -3.2204838, 3.1832514
1: -12.2265873, -8.9915504, -12.2193737, -8.9918184, -2.7994838, 2.8182325
2: -5.6109891, -2.2374573, -5.6021357, -2.2392869, -2.7967434, 2.7796607
3: -5.3699083, -1.5061598, -5.3690119, -1.5079076, -3.5402975, 3.5362258
4: -11.5212774, -7.5733171, -11.5189629, -7.5783195, -3.3136759, 3.3087335
5: -6.2888441, -3.0835361, -6.2877684, -3.0851092, -2.6388569, 2.6477308
6: -12.4268827, -8.6967516, -12.4259968, -8.6977005, -2.9107804, 2.9211636
7: -8.1675682, -4.6758609, -8.1650295, -4.6791425, -3.4884257, 3.4891686
8: 7.7401876, 10.0575027, 7.7414179, 10.0550995, -2.2101851, 2.2175791
9: -6.3461823, -2.8040414, -6.3445210, -2.8050933, -2.9272013, 2.9079924

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396716, upper bound: 1.0398426
time: 8.16 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396716, upper bound: 1.0405763
time: 6.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -6.1842647, -2.0179896, -6.2020226, -1.9591384, -3.2298164, 3.1982732
1: -12.2207966, -8.9975948, -12.2414532, -8.9769363, -2.8093338, 2.8300266
2: -5.6145725, -2.2567110, -5.6216540, -2.2080839, -2.8000879, 2.7737093
3: -5.3525000, -1.5134382, -5.3784461, -1.5029325, -3.5430841, 3.5374589
4: -11.5071106, -7.5766191, -11.5404968, -7.5641565, -3.3137999, 3.3247747
5: -6.2813778, -3.1031909, -6.3009663, -3.0896363, -2.6378260, 2.6538987
6: -12.4200497, -8.7086563, -12.4368114, -8.6995029, -2.9149833, 2.9200716
7: -8.1508808, -4.6756306, -8.1801233, -4.6730862, -3.4777946, 3.5044928
8: 7.7506104, 10.0527973, 7.7355008, 10.0657578, -2.2126584, 2.2245340
9: -6.3363180, -2.8155305, -6.3481951, -2.7928205, -2.9372244, 2.9043932

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391541, upper bound: 1.0392691
time: 6.45 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391519, upper bound: 1.0400805
time: 5.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -6.1948652, -1.9940536, -6.2037096, -1.9477563, -3.2534418, 3.2066875
1: -12.2344990, -8.9912558, -12.2469406, -8.9744177, -2.8256989, 2.8459206
2: -5.6206632, -2.2354655, -5.6229372, -2.1984134, -2.8182893, 2.7943754
3: -5.3708811, -1.5042312, -5.3863215, -1.5007043, -3.5567446, 3.5553417
4: -11.5237885, -7.5678453, -11.5466537, -7.5620203, -3.3309479, 3.3413763
5: -6.2900372, -3.0817933, -6.3024693, -3.0794806, -2.6583099, 2.6626892
6: -12.4278545, -8.6957188, -12.4390345, -8.6935825, -2.9295115, 2.9349222
7: -8.1703348, -4.6722779, -8.1877098, -4.6719484, -3.4983864, 3.5154319
8: 7.7388496, 10.0601387, 7.7302923, 10.0681124, -2.2273998, 2.2433994
9: -6.3480091, -2.8028822, -6.3509531, -2.7878306, -2.9539180, 2.9166298

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405737, upper bound: 1.0398411
time: 6.60 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405762, upper bound: 1.0405728
time: 6.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.62 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0369523, upper bound: 1.0392694
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0369523, upper bound: 1.0400809
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0383386, upper bound: 1.0398448
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0383386, upper bound: 1.0405733
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0378008, upper bound: 1.0392687
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0378008, upper bound: 1.0400805
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0392371, upper bound: 1.0398425
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0392372, upper bound: 1.0405735
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0382973, upper bound: 1.0392683
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0382996, upper bound: 1.0400804
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0396716, upper bound: 1.0398426
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0396716, upper bound: 1.0405763
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0391541, upper bound: 1.0392691
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0391519, upper bound: 1.0400805
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0405737, upper bound: 1.0398411
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.62
Output dim: 8, lower bound: -1.0405762, upper bound: 1.0405728

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1302958, -2.0425770, -6.1550536, -2.0189388, -3.1143117, 3.1449680
1: -12.1813908, -9.0150661, -12.1962337, -8.9966812, -2.7660718, 2.7836161
2: -5.5787449, -2.2849593, -5.5974360, -2.2658300, -2.7375097, 2.7616210
3: -5.3231120, -1.5443988, -5.3563137, -1.5277677, -3.4871874, 3.4866729
4: -11.4680490, -7.6098132, -11.4996147, -7.6062956, -3.0671053, 3.2719293
5: -6.2099805, -3.1508813, -6.2473273, -3.0995283, -2.5615063, 2.5377884
6: -12.3845367, -8.7429276, -12.4068203, -8.7079258, -2.8615947, 2.8243427
7: -8.1216869, -4.6961136, -8.1442280, -4.6897793, -3.3844748, 3.4481144
8: 7.7648106, 10.0285273, 7.7530632, 10.0457554, -2.1536984, 2.1656306
9: -6.2910047, -2.8524196, -6.3318305, -2.8344314, -2.8364859, 2.8571844

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361882, upper bound: 1.0392687
time: 6.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361882, upper bound: 1.0392690
time: 5.92 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1327238, -2.0422645, -6.1713843, -1.9981706, -3.1375551, 3.1637864
1: -12.1830750, -9.0147791, -12.2153597, -8.9916687, -2.7732487, 2.8117204
2: -5.5792880, -2.2820544, -5.6161718, -2.2512164, -2.7530727, 2.7825623
3: -5.3238821, -1.5427327, -5.3753357, -1.5165071, -3.5053015, 3.5142627
4: -11.4700212, -7.6044655, -11.5695219, -7.5869145, -3.0910187, 3.3379588
5: -6.2119455, -3.1503305, -6.2622271, -3.0859327, -2.5775213, 2.5533633
6: -12.3850574, -8.7424927, -12.4133224, -8.7033825, -2.8693533, 2.8343492
7: -8.1228771, -4.6933975, -8.1877451, -4.6780634, -3.3985434, 3.4943476
8: 7.7636833, 10.0293751, 7.7415695, 10.0626945, -2.1754823, 2.1798413
9: -6.2918477, -2.8501146, -6.3633533, -2.8249276, -2.8454428, 2.8907590

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0366855, upper bound: 1.0366350
time: 6.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0369478, upper bound: 1.0400741
time: 9.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1408129, -2.0188546, -6.1567063, -2.0075293, -3.1389723, 3.1533585
1: -12.1950102, -9.0087233, -12.2018089, -8.9941692, -2.7822733, 2.7996240
2: -5.5847979, -2.2640119, -5.5986919, -2.2561631, -2.7547350, 2.7801857
3: -5.3414326, -1.5354795, -5.3642550, -1.5255516, -3.5008707, 3.5044656
4: -11.4836311, -7.6010103, -11.5058479, -7.6041751, -3.0829830, 3.2887034
5: -6.2186508, -3.1295614, -6.2488198, -3.0893302, -2.5820642, 2.5460386
6: -12.3923597, -8.7300758, -12.4090519, -8.7019844, -2.8757973, 2.8393679
7: -8.1403627, -4.6927814, -8.1518965, -4.6886477, -3.4110670, 3.4591150
8: 7.7531815, 10.0358076, 7.7478285, 10.0480909, -2.1681390, 2.1844919
9: -6.3025947, -2.8398089, -6.3345265, -2.8293211, -2.8531885, 2.8691826

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375867, upper bound: 1.0398422
time: 8.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375842, upper bound: 1.0398448
time: 7.27 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1432524, -2.0185404, -6.1730375, -1.9867377, -3.1622286, 3.1721992
1: -12.1966743, -9.0084372, -12.2209148, -8.9891634, -2.7894182, 2.8277278
2: -5.5853453, -2.2611172, -5.6174293, -2.2415652, -2.7702875, 2.8011255
3: -5.3421998, -1.5338118, -5.3833270, -1.5142918, -3.5189695, 3.5320973
4: -11.4855766, -7.5956593, -11.5757256, -7.5848088, -3.1069326, 3.3542027
5: -6.2206173, -3.1290255, -6.2637210, -3.0757289, -2.5980854, 2.5615954
6: -12.3928785, -8.7296429, -12.4155350, -8.6974525, -2.8835430, 2.8493838
7: -8.1415529, -4.6900663, -8.1953697, -4.6769152, -3.4251308, 3.5053034
8: 7.7520595, 10.0366669, 7.7362833, 10.0650377, -2.1899409, 2.1987050
9: -6.3034539, -2.8375037, -6.3660698, -2.8198209, -2.8621578, 2.9027801

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0380482, upper bound: 1.0371224
time: 8.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383340, upper bound: 1.0405659
time: 5.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -6.1418648, -2.0423315, -6.1860027, -1.9718692, -3.1616230, 3.1677098
1: -12.1892118, -9.0147648, -12.2236862, -8.9792862, -2.7920170, 2.8110576
2: -5.5884333, -2.2831559, -5.6181936, -2.2248888, -2.7787442, 2.7758975
3: -5.3240724, -1.5426333, -5.3736811, -1.5205860, -3.5036316, 3.5052776
4: -11.4698467, -7.6043396, -11.5274563, -7.5899997, -3.0816517, 3.3046508
5: -6.2111654, -3.1491923, -6.2620745, -3.0939517, -2.5810051, 2.5570483
6: -12.3854885, -8.7419605, -12.4199409, -8.7038183, -2.8850341, 2.8402491
7: -8.1239529, -4.6925311, -8.1669064, -4.6825857, -3.3945875, 3.4743752
8: 7.7635517, 10.0311584, 7.7419071, 10.0587282, -2.1701994, 2.1914744
9: -6.2928224, -2.8512757, -6.3382783, -2.8170228, -2.8633342, 2.8657980

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0369990, upper bound: 1.0392684
time: 5.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0370015, upper bound: 1.0392691
time: 8.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -6.1442986, -2.0420198, -6.2023678, -1.9510889, -3.1733809, 3.1865606
1: -12.1909103, -9.0144777, -12.2429743, -8.9742737, -2.7991829, 2.8391371
2: -5.5889745, -2.2802515, -5.6369147, -2.2100966, -2.7942543, 2.7968302
3: -5.3248425, -1.5409651, -5.3928432, -1.5093191, -3.5217400, 3.5330801
4: -11.4718037, -7.5989928, -11.5970783, -7.5706625, -3.1055183, 3.3535647
5: -6.2131276, -3.1486425, -6.2770362, -3.0804105, -2.5968943, 2.5726748
6: -12.3860102, -8.7415266, -12.4265099, -8.6993189, -2.8926229, 2.8504152
7: -8.1251421, -4.6898155, -8.2103481, -4.6708651, -3.4087896, 3.5205326
8: 7.7624216, 10.0320072, 7.7304001, 10.0755987, -2.1919665, 2.2056947
9: -6.2936678, -2.8489685, -6.3697381, -2.8075283, -2.8723049, 2.8993344

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375276, upper bound: 1.0366362
time: 8.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0377938, upper bound: 1.0400733
time: 5.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -6.1523886, -2.0186093, -6.1876516, -1.9604831, -3.1851668, 3.1761203
1: -12.2028131, -9.0084305, -12.2292261, -8.9767647, -2.8082156, 2.8270459
2: -5.5944939, -2.2622147, -5.6194639, -2.2151852, -2.7969894, 2.7944837
3: -5.3423953, -1.5337179, -5.3815508, -1.5183704, -3.5172863, 3.5230041
4: -11.4853916, -7.5955343, -11.5336552, -7.5878615, -3.0974216, 3.3213887
5: -6.2198381, -3.1278915, -6.2635703, -3.0837798, -2.6015339, 2.5652800
6: -12.3933134, -8.7291145, -12.4221802, -8.6978931, -2.8992052, 2.8552856
7: -8.1426210, -4.6891999, -8.1745262, -4.6814547, -3.4211550, 3.4853263
8: 7.7519207, 10.0384483, 7.7366815, 10.0610552, -2.1846733, 2.2103109
9: -6.3044252, -2.8386526, -6.3409796, -2.8120282, -2.8799286, 2.8778152

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0385012, upper bound: 1.0398434
time: 6.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0385012, upper bound: 1.0398448
time: 7.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -6.1548319, -2.0182943, -6.2040181, -1.9396791, -3.1969495, 3.1949897
1: -12.2044945, -9.0081453, -12.2484951, -8.9717607, -2.8153505, 2.8551221
2: -5.5950427, -2.2593176, -5.6381922, -2.2003980, -2.8124971, 2.8154087
3: -5.3431606, -1.5320442, -5.4007649, -1.5071032, -3.5353823, 3.5508480
4: -11.4873238, -7.5901809, -11.6032553, -7.5685406, -3.1213255, 3.3697033
5: -6.2218046, -3.1273532, -6.2785292, -3.0702319, -2.6174307, 2.5808907
6: -12.3938313, -8.7286816, -12.4287357, -8.6934052, -2.9067869, 2.8654623
7: -8.1438084, -4.6864834, -8.2179279, -4.6697154, -3.4353590, 3.5314445
8: 7.7507973, 10.0393066, 7.7251253, 10.0779324, -2.2064524, 2.2245414
9: -6.3052821, -2.8363476, -6.3724575, -2.8025355, -2.8889122, 2.9113731

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0389500, upper bound: 1.0371227
time: 6.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392302, upper bound: 1.0405666
time: 7.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -6.1702676, -2.0187469, -6.1639757, -2.0072644, -3.1902847, 3.1664433
1: -12.2111092, -8.9981852, -12.2089033, -8.9951639, -2.7785501, 2.7950201
2: -5.6043787, -2.2617226, -5.5993209, -2.2576246, -2.7684894, 2.7536888
3: -5.3507509, -1.5171227, -5.3588023, -1.5152121, -3.5164595, 3.5106325
4: -11.5023088, -7.5874538, -11.5062408, -7.5959826, -3.2775888, 3.2801285
5: -6.2782068, -3.1055260, -6.2805262, -3.0969262, -2.6134043, 2.6322875
6: -12.4185543, -8.7101536, -12.4222555, -8.7049351, -2.8929868, 2.9027343
7: -8.1466799, -4.6819353, -8.1533089, -4.6881590, -3.4585209, 3.4713736
8: 7.7531261, 10.0493240, 7.7500224, 10.0502510, -2.1907873, 2.1937075
9: -6.3336673, -2.8190017, -6.3393326, -2.8169096, -2.9025846, 2.8908596

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375373, upper bound: 1.0392721
time: 8.45 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375373, upper bound: 1.0392688
time: 5.78 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -6.1726890, -2.0183792, -6.1803236, -1.9865127, -3.2135544, 3.1848578
1: -12.2128534, -8.9978981, -12.2280416, -8.9901543, -2.7857490, 2.8232231
2: -5.6049013, -2.2587371, -5.6180315, -2.2430043, -2.7839875, 2.7750807
3: -5.3515267, -1.5153801, -5.3778477, -1.5039239, -3.5346107, 3.5376716
4: -11.5045395, -7.5821075, -11.5760679, -7.5766134, -3.3012600, 3.3431444
5: -6.2801785, -3.1049557, -6.2954302, -3.0833516, -2.6293569, 2.6476555
6: -12.4190769, -8.7096977, -12.4287596, -8.7004175, -2.9007063, 2.9122944
7: -8.1480856, -4.6792221, -8.1967602, -4.6764321, -3.4716535, 3.5175381
8: 7.7519608, 10.0501652, 7.7385550, 10.0671444, -2.2120609, 2.2078559
9: -6.3344970, -2.8166859, -6.3707819, -2.8074069, -2.9115314, 2.9243722

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0380002, upper bound: 1.0366386
time: 6.12 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382902, upper bound: 1.0400726
time: 5.14 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -6.1808538, -1.9948082, -6.1656446, -1.9958577, -3.2150388, 3.1748080
1: -12.2248507, -8.9918375, -12.2144527, -8.9926510, -2.7949095, 2.8110704
2: -5.6104555, -2.2404590, -5.6005812, -2.2479634, -2.7867527, 2.7743311
3: -5.3691306, -1.5079205, -5.3667479, -1.5129895, -3.5301399, 3.5285683
4: -11.5190668, -7.5786824, -11.5124626, -7.5938659, -3.2948523, 3.2968345
5: -6.2868586, -3.0840955, -6.2820215, -3.0867281, -2.6339383, 2.6411190
6: -12.4263582, -8.6972113, -12.4244766, -8.6989994, -2.9072628, 2.9175611
7: -8.1661606, -4.6785855, -8.1609545, -4.6870222, -3.4791384, 3.4823689
8: 7.7413578, 10.0566483, 7.7447934, 10.0526028, -2.2055016, 2.2125976
9: -6.3453374, -2.8063626, -6.3420563, -2.8118050, -2.9193764, 2.9030547

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 135

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0389279, upper bound: 1.0398411
time: 5.17 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0389279, upper bound: 1.0398411
time: 5.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -6.1832838, -1.9944429, -6.1819916, -1.9750807, -3.2383180, 3.1932621
1: -12.2265759, -8.9915485, -12.2335720, -8.9876499, -2.8020811, 2.8392725
2: -5.6109867, -2.2374816, -5.6193018, -2.2333579, -2.8022466, 2.7957201
3: -5.3699036, -1.5061760, -5.3858438, -1.5017028, -3.5482788, 3.5556569
4: -11.5212650, -7.5733356, -11.5822563, -7.5745096, -3.3184814, 3.3592811
5: -6.2888308, -3.0835400, -6.2969217, -3.0731511, -2.6498961, 2.6564727
6: -12.4268761, -8.6967564, -12.4309654, -8.6944904, -2.9149709, 2.9271393
7: -8.1675606, -4.6758709, -8.2043610, -4.6752744, -3.4922862, 3.5284901
8: 7.7401972, 10.0574970, 7.7332773, 10.0695047, -2.2267885, 2.2267475
9: -6.3461757, -2.8040481, -6.3735251, -2.8023040, -2.9283328, 2.9365911

Time for backsubstitution: 14.28 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.225489616394043
rel_dist={8: [-1.040580051898944, 1.0405791313826693]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2423.68 seconds
