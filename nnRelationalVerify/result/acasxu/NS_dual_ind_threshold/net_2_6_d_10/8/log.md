## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 42.289434762380004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160)
1: (-17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844)
2: (-13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863)
3: (-14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786)
4: (-11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.46 + 1.71 = 4.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -42.3021254, upper bound: 42.3021254

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2929218, upper bound: 42.2912901
time: 0.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.3007659, upper bound: 42.3007659
time: 0.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.35 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 3, lower bound: -42.2929218, upper bound: 42.2912901
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.35
Output dim: 3, lower bound: -42.3007659, upper bound: 42.3007659

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -64.5731506, 107.2421494, -97.5574417, 165.7690125, -230.3421478, 204.7995911
1: -11.5029459, 15.4048920, -17.7561874, 23.8874989, -35.3904457, 33.1610794
2: -8.6509132, 14.4953432, -13.4649935, 21.8683014, -30.5192146, 27.9603329
3: -9.1930447, 24.5248833, -14.3390293, 36.8165550, -46.0095978, 38.8639107
4: -7.2651896, 18.4225006, -11.3048334, 27.4657993, -34.7309837, 29.7273331

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2834459, upper bound: 42.2834459
time: 0.58 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2834459, upper bound: 42.2834459
time: 0.60 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -96.6716614, 164.1812592, -97.5574417, 165.7690125, -262.4406738, 261.7386780
1: -17.5651093, 23.6462498, -17.7561874, 23.8874989, -41.4526062, 41.4024353
2: -13.3320065, 21.6681614, -13.4649935, 21.8683014, -35.2002983, 35.1331558
3: -14.2025223, 36.4909859, -14.3390293, 36.8165550, -51.0190735, 50.8300133
4: -11.1946545, 27.2369251, -11.3048334, 27.4657993, -38.6604538, 38.5417595

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912901, upper bound: 42.2929218
time: 0.56 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912901, upper bound: 42.3007659
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.60 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.60
Output dim: 3, lower bound: -42.2834459, upper bound: 42.2834459
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 3.60
Output dim: 3, lower bound: -42.2834459, upper bound: 42.2834459
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 3, lower bound: -42.2912901, upper bound: 42.2929218
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 3, lower bound: -42.2912901, upper bound: 42.3007659

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -96.6716614, 164.1812592, -64.5731506, 107.2421494, -203.9137726, 228.7543945
1: -17.5651093, 23.6462498, -11.5029459, 15.4048920, -32.9700012, 35.1491966
2: -13.3320065, 21.6681614, -8.6509132, 14.4953432, -27.8273430, 30.3190746
3: -14.2025223, 36.4909859, -9.1930447, 24.5248833, -38.7274055, 45.6840286
4: -11.1946545, 27.2369251, -7.2651896, 18.4225006, -29.6171551, 34.5021133

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2886357, upper bound: 42.2845504
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2887191, upper bound: 42.2908856
time: 0.55 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -96.6716614, 164.1812592, -96.6716614, 164.1812592, -260.8529053, 260.8529053
1: -17.5651093, 23.6462498, -17.5651093, 23.6462498, -41.2113571, 41.2113571
2: -13.3320065, 21.6681614, -13.3320065, 21.6681614, -35.0001640, 35.0001678
3: -14.2025223, 36.4909859, -14.2025223, 36.4909859, -50.6935081, 50.6935081
4: -11.1946545, 27.2369251, -11.1946545, 27.2369251, -38.4315796, 38.4315796

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2886357, upper bound: 42.2845505
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2887191, upper bound: 42.2984590
time: 0.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.65 seconds
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.65
Output dim: 3, lower bound: -42.2886357, upper bound: 42.2845504
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 3, lower bound: -42.2887191, upper bound: 42.2908856
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.65
Output dim: 3, lower bound: -42.2886357, upper bound: 42.2845505
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.65
Output dim: 3, lower bound: -42.2887191, upper bound: 42.2984590

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -93.9068680, 158.9183044, -64.5731506, 107.2421494, -201.1490021, 223.4914246
1: -16.9745560, 22.9028091, -11.5029459, 15.4048920, -32.3794479, 34.4057541
2: -12.9039955, 21.0168800, -8.6509132, 14.4953432, -27.3993320, 29.6677933
3: -13.7340364, 35.3916016, -9.1930447, 24.5248833, -38.2589188, 44.5846481
4: -10.8326273, 26.4757767, -7.2651896, 18.4225006, -29.2551270, 33.7409668

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2807513, upper bound: 42.2831640
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2887191, upper bound: 42.2908856
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -93.9068680, 158.9183044, -96.6716614, 164.1812592, -258.0881042, 255.5899658
1: -16.9745560, 22.9028091, -17.5651093, 23.6462498, -40.6208038, 40.4679184
2: -12.9039955, 21.0168800, -13.3320065, 21.6681614, -34.5721588, 34.3488770
3: -13.7340364, 35.3916016, -14.2025223, 36.4909859, -50.2250214, 49.5941238
4: -10.8326273, 26.4757767, -11.1946545, 27.2369251, -38.0695534, 37.6704330

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2983403
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2984590
time: 0.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.68 seconds
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.68
Output dim: 3, lower bound: -42.2807513, upper bound: 42.2831640
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 3, lower bound: -42.2887191, upper bound: 42.2908856
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2983403
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.68
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2984590

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -93.9068680, 158.9183044, -64.1949463, 106.5913620, -200.4982147, 223.1132507
1: -16.9745560, 22.9028091, -11.4337902, 15.3050909, -32.2796478, 34.3365936
2: -12.9039955, 21.0168800, -8.5953493, 14.4179525, -27.3219490, 29.6122265
3: -13.7340364, 35.3916016, -9.1349344, 24.3906498, -38.1246872, 44.5265236
4: -10.8326273, 26.4757767, -7.2177763, 18.3264828, -29.1591091, 33.6935539

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2840100, upper bound: 42.2857580
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2840100, upper bound: 42.2908856
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -93.9068680, 158.9183044, -87.6639481, 146.3591461, -240.2659912, 246.5822449
1: -16.9745560, 22.9028091, -15.4490967, 21.1456165, -38.1201706, 38.3519058
2: -12.9039955, 21.0168800, -11.8700638, 19.5199356, -32.4239311, 32.8869400
3: -13.7340364, 35.3916016, -12.6006975, 32.8834038, -46.6174393, 47.9922981
4: -10.8326273, 26.4757767, -9.9631863, 24.7217846, -35.5544128, 36.4389610

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2883119, upper bound: 42.2939642
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2983403
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -93.9068680, 158.9183044, -93.9068680, 158.9183044, -252.8251495, 252.8251648
1: -16.9745560, 22.9028091, -16.9745560, 22.9028091, -39.8773651, 39.8773651
2: -12.9039955, 21.0168800, -12.9039955, 21.0168800, -33.9208755, 33.9208755
3: -13.7340364, 35.3916016, -13.7340364, 35.3916016, -49.1256371, 49.1256371
4: -10.8326273, 26.4757767, -10.8326273, 26.4757767, -37.3083992, 37.3084030

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2883119, upper bound: 42.2940380
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2984590
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.65 seconds
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.65
Output dim: 3, lower bound: -42.2840100, upper bound: 42.2857580
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 3, lower bound: -42.2840100, upper bound: 42.2908856
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 3, lower bound: -42.2883119, upper bound: 42.2939642
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2983403
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 3, lower bound: -42.2883119, upper bound: 42.2940380
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.65
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2984590

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -93.4003983, 158.0383148, -64.1949463, 106.5913620, -199.9917603, 222.2332458
1: -16.8792496, 22.7737808, -11.4337902, 15.3050909, -32.1843338, 34.2075653
2: -12.8302355, 20.9064522, -8.5953493, 14.4179525, -27.2481880, 29.5017967
3: -13.6584692, 35.2013512, -9.1349344, 24.3906498, -38.0491104, 44.3362694
4: -10.7709274, 26.3413048, -7.2177763, 18.3264828, -29.0974064, 33.5590782

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2749902, upper bound: 42.2894468
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2749902, upper bound: 42.2896232
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -91.2296677, 155.4463196, -87.6639481, 146.3591461, -237.5888062, 243.1102600
1: -16.6061668, 22.4123840, -15.4490967, 21.1456165, -37.7517853, 37.8614769
2: -12.5907145, 20.5355873, -11.8700638, 19.5199356, -32.1106491, 32.4056511
3: -13.4153957, 34.5722237, -12.6006975, 32.8834038, -46.2987976, 47.1729202
4: -10.5654392, 25.8755703, -9.9631863, 24.7217846, -35.2872238, 35.8387566

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2883119, upper bound: 42.2903501
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2883119, upper bound: 42.2939642
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -93.4003983, 158.0383148, -87.6639481, 146.3591461, -239.7595520, 245.7022400
1: -16.8792496, 22.7737808, -15.4490967, 21.1456165, -38.0248604, 38.2228775
2: -12.8302355, 20.9064522, -11.8700638, 19.5199356, -32.3501663, 32.7765121
3: -13.6584692, 35.2013512, -12.6006975, 32.8834038, -46.5418701, 47.8020477
4: -10.7709274, 26.3413048, -9.9631863, 24.7217846, -35.4927139, 36.3044853

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2945836
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2983403
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -91.2296677, 155.4463196, -93.9068680, 158.9183044, -250.1479645, 249.3531799
1: -16.6061668, 22.4123840, -16.9745560, 22.9028091, -39.5089760, 39.3869400
2: -12.5907145, 20.5355873, -12.9039955, 21.0168800, -33.6075897, 33.4395828
3: -13.4153957, 34.5722237, -13.7340364, 35.3916016, -48.8069916, 48.3062592
4: -10.5654392, 25.8755703, -10.8326273, 26.4757767, -37.0412102, 36.7081947

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2891660, upper bound: 42.2891269
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2891660, upper bound: 42.2940380
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -93.4003983, 158.0383148, -93.9068680, 158.9183044, -252.3186951, 251.9451599
1: -16.8792496, 22.7737808, -16.9745560, 22.9028091, -39.7820511, 39.7483368
2: -12.8302355, 20.9064522, -12.9039955, 21.0168800, -33.8471107, 33.8104477
3: -13.6584692, 35.2013512, -13.7340364, 35.3916016, -49.0500679, 48.9353867
4: -10.7709274, 26.3413048, -10.8326273, 26.4757767, -37.2467041, 37.1739273

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2942286, upper bound: 42.2933844
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2942286, upper bound: 42.2984589
time: 0.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.71 seconds
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 3, lower bound: -42.2749902, upper bound: 42.2894468
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 3, lower bound: -42.2749902, upper bound: 42.2896232
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 3, lower bound: -42.2883119, upper bound: 42.2903501
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 3, lower bound: -42.2883119, upper bound: 42.2939642
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2945836
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 3, lower bound: -42.2930210, upper bound: 42.2983403
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.71
Output dim: 3, lower bound: -42.2891660, upper bound: 42.2891269
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 3, lower bound: -42.2891660, upper bound: 42.2940380
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 3, lower bound: -42.2942286, upper bound: 42.2933844
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 3, lower bound: -42.2942286, upper bound: 42.2984589

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -93.4003983, 158.0383148, -58.6807938, 96.2245941, -189.6250000, 216.7190704
1: -16.8792496, 22.7737808, -10.1782589, 13.7426033, -30.6218529, 32.9520378
2: -12.8302355, 20.9064522, -7.6865330, 13.1743593, -26.0045929, 28.5929813
3: -13.6584692, 35.2013512, -8.1302853, 22.3398209, -35.9982796, 43.3316345
4: -10.7709274, 26.3413048, -6.4551730, 16.7901936, -27.5611191, 32.7964783

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2813344, upper bound: 42.2839539
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2807057, upper bound: 42.2888532
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -93.4003983, 158.0383148, -62.3252487, 103.2832413, -196.6836395, 220.3635559
1: -16.8792496, 22.7737808, -11.0508442, 14.8055944, -31.6848412, 33.8246155
2: -12.8302355, 20.9064522, -8.3132668, 14.0101976, -26.8404331, 29.2197170
3: -13.6584692, 35.2013512, -8.8166170, 23.7160416, -37.3745003, 44.0179672
4: -10.7709274, 26.3413048, -6.9744468, 17.8107491, -28.5816746, 33.3157501

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2813344, upper bound: 42.2849628
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2807057, upper bound: 42.2892291
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -91.2296677, 155.4463196, -85.1457443, 143.3562012, -234.5858612, 240.5920563
1: -16.6061668, 22.4123840, -15.1331406, 20.7084198, -37.3145828, 37.5455246
2: -12.5907145, 20.5355873, -11.5880480, 19.0748348, -31.6655502, 32.1236305
3: -13.4153957, 34.5722237, -12.3083191, 32.1063652, -45.5217514, 46.8805428
4: -10.5654392, 25.8755703, -9.7200127, 24.1985092, -34.7639427, 35.5955811

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2880236, upper bound: 42.2885551
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2880177, upper bound: 42.2900280
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -91.2296677, 155.4463196, -87.1590881, 145.4854431, -236.7151031, 242.6053619
1: -16.6061668, 22.4123840, -15.3542900, 21.0158844, -37.6220512, 37.7666702
2: -12.5907145, 20.5355873, -11.7963285, 19.4107037, -32.0014153, 32.3319168
3: -13.4153957, 34.5722237, -12.5249882, 32.6966591, -46.1120453, 47.0972099
4: -10.5654392, 25.8755703, -9.9017277, 24.5874481, -35.1528854, 35.7772865

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2880236, upper bound: 42.2920357
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2880177, upper bound: 42.2935989
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -93.4003983, 158.0383148, -85.1457443, 143.3562012, -236.7565918, 243.1840363
1: -16.8792496, 22.7737808, -15.1331406, 20.7084198, -37.5876617, 37.9069214
2: -12.8302355, 20.9064522, -11.5880480, 19.0748348, -31.9050674, 32.4944916
3: -13.6584692, 35.2013512, -12.3083191, 32.1063652, -45.7648277, 47.5096664
4: -10.7709274, 26.3413048, -9.7200127, 24.1985092, -34.9694366, 36.0613174

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2928654, upper bound: 42.2891149
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2922366, upper bound: 42.2944185
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -93.4003983, 158.0383148, -87.1590881, 145.4854431, -238.8858337, 245.1973267
1: -16.8792496, 22.7737808, -15.3542900, 21.0158844, -37.8951263, 38.1280708
2: -12.8302355, 20.9064522, -11.7963285, 19.4107037, -32.2409325, 32.7027779
3: -13.6584692, 35.2013512, -12.5249882, 32.6966591, -46.3551178, 47.7263298
4: -10.7709274, 26.3413048, -9.9017277, 24.5874481, -35.3583755, 36.2430229

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2928654, upper bound: 42.2921000
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2922366, upper bound: 42.2974028
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -91.2296677, 155.4463196, -93.4003983, 158.0383148, -249.2679749, 248.8467102
1: -16.6061668, 22.4123840, -16.8792496, 22.7737808, -39.3799477, 39.2916222
2: -12.5907145, 20.5355873, -12.8302355, 20.9064522, -33.4971619, 33.3658218
3: -13.4153957, 34.5722237, -13.6584692, 35.2013512, -48.6167412, 48.2306900
4: -10.5654392, 25.8755703, -10.7709274, 26.3413048, -36.9067383, 36.6464996

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2889003, upper bound: 42.2920498
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2888953, upper bound: 42.2936589
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -93.4003983, 158.0383148, -91.2296677, 155.4463196, -248.8467102, 249.2679749
1: -16.8792496, 22.7737808, -16.6061668, 22.4123840, -39.2916183, 39.3799477
2: -12.8302355, 20.9064522, -12.5907145, 20.5355873, -33.3658218, 33.4971581
3: -13.6584692, 35.2013512, -13.4153957, 34.5722237, -48.2306900, 48.6167412
4: -10.7709274, 26.3413048, -10.5654392, 25.8755703, -36.6464996, 36.9067383

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2940650, upper bound: 42.2879132
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2934362, upper bound: 42.2932176
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -93.4003983, 158.0383148, -93.4003983, 158.0383148, -251.4387207, 251.4387207
1: -16.8792496, 22.7737808, -16.8792496, 22.7737808, -39.6530266, 39.6530266
2: -12.8302355, 20.9064522, -12.8302355, 20.9064522, -33.7366829, 33.7366829
3: -13.6584692, 35.2013512, -13.6584692, 35.2013512, -48.8598137, 48.8598137
4: -10.7709274, 26.3413048, -10.7709274, 26.3413048, -37.1122322, 37.1122322

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2940650, upper bound: 42.2923310
time: 1.94 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2934362, upper bound: 42.2976032
time: 0.59 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.04 seconds
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2813344, upper bound: 42.2839539
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2807057, upper bound: 42.2888532
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2813344, upper bound: 42.2849628
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2807057, upper bound: 42.2892291
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2880236, upper bound: 42.2885551
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2880177, upper bound: 42.2900280
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2880236, upper bound: 42.2920357
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2880177, upper bound: 42.2935989
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2928654, upper bound: 42.2891149
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2922366, upper bound: 42.2944185
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2928654, upper bound: 42.2921000
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2922366, upper bound: 42.2974028
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2889003, upper bound: 42.2920498
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2888953, upper bound: 42.2936589
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2940650, upper bound: 42.2879132
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2934362, upper bound: 42.2932176
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2940650, upper bound: 42.2923310
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.04
Output dim: 3, lower bound: -42.2934362, upper bound: 42.2976032

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -85.1457443, 143.3562012, -234.3001862, 240.0538330
1: -16.5433254, 22.3302612, -15.1331406, 20.7084198, -37.2517433, 37.4634018
2: -12.5446377, 20.4681129, -11.5880480, 19.0748348, -31.6194725, 32.0561600
3: -13.3658085, 34.4641609, -12.3083191, 32.1063652, -45.4721756, 46.7724800
4: -10.5271111, 25.7950706, -9.7200127, 24.1985092, -34.7256203, 35.5150833

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2852927, upper bound: 42.2872987
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2852927, upper bound: 42.2900280
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -82.8065567, 140.2019348, -87.1590881, 145.4854431, -228.2919769, 227.3609772
1: -14.7189159, 20.2101326, -15.3542900, 21.0158844, -35.7348022, 35.5644188
2: -11.3123865, 18.7014847, -11.7963285, 19.4107037, -30.7230873, 30.4978142
3: -11.9629860, 31.3418674, -12.5249882, 32.6966591, -44.6596451, 43.8668442
4: -9.4549694, 23.7045956, -9.9017277, 24.5874481, -34.0424156, 33.6063156

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2852924, upper bound: 42.2918799
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2852924, upper bound: 42.2920357
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -87.1590881, 145.4854431, -236.4294281, 242.0671539
1: -16.5433254, 22.3302612, -15.3542900, 21.0158844, -37.5592117, 37.6845474
2: -12.5446377, 20.4681129, -11.7963285, 19.4107037, -31.9553394, 32.2644424
3: -13.3658085, 34.4641609, -12.5249882, 32.6966591, -46.0624695, 46.9891396
4: -10.5271111, 25.7950706, -9.9017277, 24.5874481, -35.1145592, 35.6967926

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2852924, upper bound: 42.2920350
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2852924, upper bound: 42.2935989
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -91.5973816, 154.5887604, -85.1457443, 143.3562012, -234.9535828, 239.7344971
1: -16.4933624, 22.3109341, -15.1331406, 20.7084198, -37.2017784, 37.4440765
2: -12.5618811, 20.4642582, -11.5880480, 19.0748348, -31.6367149, 32.0523071
3: -13.3621187, 34.4498138, -12.3083191, 32.1063652, -45.4684792, 46.7581329
4: -10.5422421, 25.8184700, -9.7200127, 24.1985092, -34.7407494, 35.5384789

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2777516, upper bound: 42.2890999
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2777516, upper bound: 42.2891149
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -92.7636185, 156.7567902, -85.1457443, 143.3562012, -236.1197968, 241.9025116
1: -16.7301445, 22.5964622, -15.1331406, 20.7084198, -37.4385643, 37.7296028
2: -12.7266340, 20.7519531, -11.5880480, 19.0748348, -31.8014679, 32.3399963
3: -13.5472450, 34.9396820, -12.3083191, 32.1063652, -45.6536064, 47.2479935
4: -10.6834011, 26.1624432, -9.7200127, 24.1985092, -34.8819084, 35.8824539

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2777516, upper bound: 42.2944025
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2777516, upper bound: 42.2944185
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -91.5973816, 154.5887604, -87.1590881, 145.4854431, -237.0828247, 241.7478333
1: -16.4933624, 22.3109341, -15.3542900, 21.0158844, -37.5092468, 37.6652222
2: -12.5618811, 20.4642582, -11.7963285, 19.4107037, -31.9725780, 32.2605858
3: -13.3621187, 34.4498138, -12.5249882, 32.6966591, -46.0587692, 46.9747963
4: -10.5422421, 25.8184700, -9.9017277, 24.5874481, -35.1296883, 35.7201881

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2890389, upper bound: 42.2921000
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2890389, upper bound: 42.2921000
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -92.7636185, 156.7567902, -87.1590881, 145.4854431, -238.2490387, 243.9158325
1: -16.7301445, 22.5964622, -15.3542900, 21.0158844, -37.7460289, 37.9507523
2: -12.7266340, 20.7519531, -11.7963285, 19.4107037, -32.1373291, 32.5482826
3: -13.5472450, 34.9396820, -12.5249882, 32.6966591, -46.2438965, 47.4646568
4: -10.6834011, 26.1624432, -9.9017277, 24.5874481, -35.2708511, 36.0641632

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2890389, upper bound: 42.2974028
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2890389, upper bound: 42.2974028
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -82.8065567, 140.2019348, -93.4003983, 158.0383148, -240.8448486, 233.6023254
1: -14.7189159, 20.2101326, -16.8792496, 22.7737808, -37.4926987, 37.0893745
2: -11.3123865, 18.7014847, -12.8302355, 20.9064522, -32.2188339, 31.5317192
3: -11.9629860, 31.3418674, -13.6584692, 35.2013512, -47.1643333, 45.0003281
4: -9.4549694, 23.7045956, -10.7709274, 26.3413048, -35.7962723, 34.4755249

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2870833, upper bound: 42.2918880
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2870833, upper bound: 42.2920498
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -93.4003983, 158.0383148, -248.9822845, 248.3085022
1: -16.5433254, 22.3302612, -16.8792496, 22.7737808, -39.3171082, 39.2094994
2: -12.5446377, 20.4681129, -12.8302355, 20.9064522, -33.4510880, 33.2983437
3: -13.3658085, 34.4641609, -13.6584692, 35.2013512, -48.5671616, 48.1226273
4: -10.5271111, 25.7950706, -10.7709274, 26.3413048, -36.8684120, 36.5659981

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2870834, upper bound: 42.2920472
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2870834, upper bound: 42.2936589
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -91.5973816, 154.5887604, -91.2296677, 155.4463196, -247.0437012, 245.8184204
1: -16.4933624, 22.3109341, -16.6061668, 22.4123840, -38.9057388, 38.9170990
2: -12.5618811, 20.4642582, -12.5907145, 20.5355873, -33.0974693, 33.0549660
3: -13.3621187, 34.4498138, -13.4153957, 34.5722237, -47.9343414, 47.8652039
4: -10.5422421, 25.8184700, -10.5654392, 25.8755703, -36.4178123, 36.3839035

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -92.7636185, 156.7567902, -91.2296677, 155.4463196, -248.2099152, 247.9864502
1: -16.7301445, 22.5964622, -16.6061668, 22.4123840, -39.1425247, 39.2026291
2: -12.7266340, 20.7519531, -12.5907145, 20.5355873, -33.2622185, 33.3426628
3: -13.5472450, 34.9396820, -13.4153957, 34.5722237, -48.1194687, 48.3550682
4: -10.6834011, 26.1624432, -10.5654392, 25.8755703, -36.5589638, 36.7278824

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2869257, upper bound: 42.2870829
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2930971, upper bound: 42.2926965
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -91.5973816, 154.5887604, -93.4003983, 158.0383148, -249.6356964, 247.9891663
1: -16.4933624, 22.3109341, -16.8792496, 22.7737808, -39.2671432, 39.1901817
2: -12.5618811, 20.4642582, -12.8302355, 20.9064522, -33.4683304, 33.2944908
3: -13.3621187, 34.4498138, -13.6584692, 35.2013512, -48.5634651, 48.1082764
4: -10.5422421, 25.8184700, -10.7709274, 26.3413048, -36.8835411, 36.5893974

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2925337, upper bound: 42.2923310
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2925337, upper bound: 42.2923310
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -92.7636185, 156.7567902, -93.4003983, 158.0383148, -250.8018951, 250.1571960
1: -16.7301445, 22.5964622, -16.8792496, 22.7737808, -39.5039253, 39.4757080
2: -12.7266340, 20.7519531, -12.8302355, 20.9064522, -33.6330795, 33.5821838
3: -13.5472450, 34.9396820, -13.6584692, 35.2013512, -48.7485924, 48.5981407
4: -10.6834011, 26.1624432, -10.7709274, 26.3413048, -37.0247002, 36.9333725

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2925337, upper bound: 42.2976033
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2925337, upper bound: 42.2976033
time: 0.64 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.85 seconds
NS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2852927, upper bound: 42.2872987
NS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2852927, upper bound: 42.2900280
NS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2852924, upper bound: 42.2918799
NS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2852924, upper bound: 42.2920357
NS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2852924, upper bound: 42.2920350
NS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2852924, upper bound: 42.2935989
NS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2777516, upper bound: 42.2890999
NS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2777516, upper bound: 42.2891149
NS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2777516, upper bound: 42.2944025
NS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2777516, upper bound: 42.2944185
NS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2890389, upper bound: 42.2921000
NS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2890389, upper bound: 42.2921000
NS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2890389, upper bound: 42.2974028
NS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2890389, upper bound: 42.2974028
NS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2870833, upper bound: 42.2918880
NS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2870833, upper bound: 42.2920498
NS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2870834, upper bound: 42.2920472
NS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2870834, upper bound: 42.2936589
NS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2869257, upper bound: 42.2870829
NS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2930971, upper bound: 42.2926965
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2925337, upper bound: 42.2923310
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2925337, upper bound: 42.2923310
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2925337, upper bound: 42.2976033
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 3, lower bound: -42.2925337, upper bound: 42.2976033

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -84.8850632, 142.8439484, -233.7879486, 239.7931671
1: -16.5433254, 22.3302612, -15.0730915, 20.6303272, -37.1736488, 37.4033508
2: -12.5446377, 20.4681129, -11.5441523, 19.0132561, -31.5578938, 32.0122643
3: -13.3658085, 34.4641609, -12.2610378, 32.0082817, -45.3740921, 46.7251968
4: -10.5271111, 25.7950706, -9.6835575, 24.1226692, -34.6497803, 35.4786301

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -82.8065567, 140.2019348, -80.5592499, 134.4231720, -217.2297363, 220.7611389
1: -14.7189159, 20.2101326, -14.0120878, 19.3485775, -34.0674896, 34.2222214
2: -11.3123865, 18.7014847, -10.8378916, 18.0931339, -29.4055195, 29.5393753
3: -11.9629860, 31.3418674, -11.4429255, 30.3752327, -42.3382111, 42.7847862
4: -9.4549694, 23.7045956, -9.0627041, 22.9575558, -32.4125252, 32.7672997

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -82.8065567, 140.2019348, -86.8999100, 144.9782104, -227.7847595, 227.1018372
1: -14.7189159, 20.2101326, -15.2939510, 20.9383564, -35.6572647, 35.5040779
2: -11.3123865, 18.7014847, -11.7524242, 19.3501606, -30.6625423, 30.4539089
3: -11.9629860, 31.3418674, -12.4774256, 32.5995598, -44.5625458, 43.8192940
4: -9.4549694, 23.7045956, -9.8652716, 24.5129929, -33.9679604, 33.5698662

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -80.5592499, 134.4231720, -225.3671875, 235.4673004
1: -16.5433254, 22.3302612, -14.0120878, 19.3485775, -35.8918991, 36.3423500
2: -12.5446377, 20.4681129, -10.8378916, 18.0931339, -30.6377716, 31.3059998
3: -13.3658085, 34.4641609, -11.4429255, 30.3752327, -43.7410431, 45.9070854
4: -10.5271111, 25.7950706, -9.0627041, 22.9575558, -33.4846649, 34.8577728

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -86.8999100, 144.9782104, -235.9221954, 241.8080139
1: -16.5433254, 22.3302612, -15.2939510, 20.9383564, -37.4816704, 37.6242142
2: -12.5446377, 20.4681129, -11.7524242, 19.3501606, -31.8947945, 32.2205353
3: -13.3658085, 34.4641609, -12.4774256, 32.5995598, -45.9653702, 46.9415855
4: -10.5271111, 25.7950706, -9.8652716, 24.5129929, -35.0401039, 35.6603432

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -92.7636185, 156.7567902, -84.2522964, 141.5968475, -234.3604279, 241.0090485
1: -16.7301445, 22.5964622, -14.9411621, 20.4636497, -37.1937943, 37.5376244
2: -12.7266340, 20.7519531, -11.4510040, 18.8597946, -31.5864258, 32.2029572
3: -13.5472450, 34.9396820, -12.1371870, 31.7596703, -45.3069115, 47.0768700
4: -10.6834011, 26.1624432, -9.6021099, 23.9315891, -34.6149864, 35.7645531

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2763017, upper bound: 42.2901329
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2771017, upper bound: 42.2930032
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -92.7636185, 156.7567902, -84.5565338, 142.2301331, -234.9937286, 241.3132935
1: -16.7301445, 22.5964622, -15.0061140, 20.5480175, -37.2781601, 37.6025772
2: -12.7266340, 20.7519531, -11.4953394, 18.9423313, -31.6689644, 32.2472916
3: -13.5472450, 34.9396820, -12.2066050, 31.8855362, -45.4327774, 47.1462860
4: -10.6834011, 26.1624432, -9.6422052, 24.0349655, -34.7183647, 35.8046455

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2763017, upper bound: 42.2901645
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2771017, upper bound: 42.2930156
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -91.5973816, 154.5887604, -85.5928268, 142.3337250, -233.9311066, 240.1815796
1: -16.4933624, 22.3109341, -14.9906960, 20.5860748, -37.0794373, 37.3016281
2: -12.5618811, 20.4642582, -11.5474100, 19.0303516, -31.5922318, 32.0116692
3: -13.3621187, 34.4498138, -12.2445526, 32.0809517, -45.4430695, 46.6943665
4: -10.5422421, 25.8184700, -9.6921005, 24.1245098, -34.6667442, 35.5105667

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2876179, upper bound: 42.2902789
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2883651, upper bound: 42.2898971
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -91.5973816, 154.5887604, -86.5624771, 144.3015137, -235.8988953, 241.1512451
1: -16.4933624, 22.3109341, -15.2155390, 20.8484764, -37.3418388, 37.5264740
2: -12.5618811, 20.4642582, -11.6986370, 19.2733593, -31.8352356, 32.1628876
3: -13.3621187, 34.4498138, -12.4184294, 32.4693947, -45.8315125, 46.8682404
4: -10.5422421, 25.8184700, -9.8197441, 24.4203224, -34.9625626, 35.6382141

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2876179, upper bound: 42.2902789
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2883651, upper bound: 42.2898971
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -92.7636185, 156.7567902, -85.5928268, 142.3337250, -235.0973511, 242.3495941
1: -16.7301445, 22.5964622, -14.9906960, 20.5860748, -37.3162193, 37.5871582
2: -12.7266340, 20.7519531, -11.5474100, 19.0303516, -31.7569847, 32.2993622
3: -13.5472450, 34.9396820, -12.2445526, 32.0809517, -45.6281967, 47.1842270
4: -10.6834011, 26.1624432, -9.6921005, 24.1245098, -34.8079071, 35.8545456

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2869462, upper bound: 42.2928380
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2877461, upper bound: 42.2958219
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -92.7636185, 156.7567902, -86.5624771, 144.3015137, -237.0651093, 243.3192749
1: -16.7301445, 22.5964622, -15.2155390, 20.8484764, -37.5786209, 37.8120003
2: -12.7266340, 20.7519531, -11.6986370, 19.2733593, -31.9999886, 32.4505882
3: -13.5472450, 34.9396820, -12.4184294, 32.4693947, -46.0166397, 47.3581009
4: -10.6834011, 26.1624432, -9.8197441, 24.4203224, -35.1037216, 35.9821854

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2869461, upper bound: 42.2928381
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2877461, upper bound: 42.2958219
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -82.8065567, 140.2019348, -84.7485046, 142.5131531, -225.3196869, 224.9504242
1: -14.7189159, 20.2101326, -14.9432907, 20.5217361, -35.2406540, 35.1534233
2: -11.3123865, 18.7014847, -11.5180264, 19.0611954, -30.3735809, 30.2195091
3: -11.9629860, 31.3418674, -12.1737843, 31.9677715, -43.9307518, 43.5156479
4: -9.4549694, 23.7045956, -9.6289072, 24.1498451, -33.6048126, 33.3335037

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -82.8065567, 140.2019348, -93.1277313, 157.5046387, -240.3111877, 233.3296509
1: -14.7189159, 20.2101326, -16.8167305, 22.6931000, -37.4120102, 37.0268631
2: -11.3123865, 18.7014847, -12.7847471, 20.8404922, -32.1528778, 31.4862270
3: -11.9629860, 31.3418674, -13.6093645, 35.0940666, -47.0570526, 44.9512215
4: -9.4549694, 23.7045956, -10.7330494, 26.2626457, -35.7176132, 34.4376450

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -84.7485046, 142.5131531, -233.4571381, 239.6566010
1: -16.5433254, 22.3302612, -14.9432907, 20.5217361, -37.0650635, 37.2735481
2: -12.5446377, 20.4681129, -11.5180264, 19.0611954, -31.6058331, 31.9861393
3: -13.3658085, 34.4641609, -12.1737843, 31.9677715, -45.3335800, 46.6379471
4: -10.5271111, 25.7950706, -9.6289072, 24.1498451, -34.6769562, 35.4239769

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -90.9440155, 154.9081116, -93.1277313, 157.5046387, -248.4486237, 248.0358276
1: -16.5433254, 22.3302612, -16.8167305, 22.6931000, -39.2364159, 39.1469917
2: -12.5446377, 20.4681129, -12.7847471, 20.8404922, -33.3851204, 33.2528496
3: -13.3658085, 34.4641609, -13.6093645, 35.0940666, -48.4598770, 48.0735207
4: -10.5271111, 25.7950706, -10.7330494, 26.2626457, -36.7897568, 36.5281219

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -92.7636185, 156.7567902, -90.9440155, 154.9081116, -247.6717072, 247.7007904
1: -16.7301445, 22.5964622, -16.5433254, 22.3302612, -39.0604057, 39.1397858
2: -12.7266340, 20.7519531, -12.5446377, 20.4681129, -33.1947365, 33.2965889
3: -13.5472450, 34.9396820, -13.3658085, 34.4641609, -48.0114059, 48.3054886
4: -10.6834011, 26.1624432, -10.5271111, 25.7950706, -36.4784698, 36.6895523

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2913285, upper bound: 42.2882214
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2922509, upper bound: 42.2913651
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -91.5973816, 154.5887604, -91.5973816, 154.5887604, -246.1861420, 246.1861420
1: -16.4933624, 22.3109341, -16.4933624, 22.3109341, -38.8042946, 38.8042946
2: -12.5618811, 20.4642582, -12.5618811, 20.4642582, -33.0261345, 33.0261345
3: -13.3621187, 34.4498138, -13.3621187, 34.4498138, -47.8119278, 47.8119278
4: -10.5422421, 25.8184700, -10.5422421, 25.8184700, -36.3607063, 36.3607063

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2908605, upper bound: 42.2904341
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2908905, upper bound: 42.2900326
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -91.5973816, 154.5887604, -92.7636185, 156.7567902, -248.3541565, 247.3523865
1: -16.4933624, 22.3109341, -16.7301445, 22.5964622, -39.0898247, 39.0410767
2: -12.5618811, 20.4642582, -12.7266340, 20.7519531, -33.3138313, 33.1908836
3: -13.3621187, 34.4498138, -13.5472450, 34.9396820, -48.3017921, 47.9970551
4: -10.5422421, 25.8184700, -10.6834011, 26.1624432, -36.7046852, 36.5018654

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2908605, upper bound: 42.2904341
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2908905, upper bound: 42.2900326
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -92.7636185, 156.7567902, -91.5973816, 154.5887604, -247.3523865, 248.3541718
1: -16.7301445, 22.5964622, -16.4933624, 22.3109341, -39.0410767, 39.0898247
2: -12.7266340, 20.7519531, -12.5618811, 20.4642582, -33.1908836, 33.3138313
3: -13.5472450, 34.9396820, -13.3621187, 34.4498138, -47.9970551, 48.3017921
4: -10.6834011, 26.1624432, -10.5422421, 25.8184700, -36.5018654, 36.7046852

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2904226, upper bound: 42.2930337
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2906377, upper bound: 42.2959029
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -92.7636185, 156.7567902, -92.7636185, 156.7567902, -249.5203552, 249.5203857
1: -16.7301445, 22.5964622, -16.7301445, 22.5964622, -39.3266068, 39.3266068
2: -12.7266340, 20.7519531, -12.7266340, 20.7519531, -33.4785805, 33.4785805
3: -13.5472450, 34.9396820, -13.5472450, 34.9396820, -48.4869194, 48.4869194
4: -10.6834011, 26.1624432, -10.6834011, 26.1624432, -36.8458405, 36.8458443

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2904226, upper bound: 42.2930338
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2906377, upper bound: 42.2959029
time: 0.64 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 3.88 seconds
NS_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2763017, upper bound: 42.2901329
NS_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2771017, upper bound: 42.2930032
NS_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2763017, upper bound: 42.2901645
NS_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2771017, upper bound: 42.2930156
NS_A2_B2_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2876179, upper bound: 42.2902789
NS_A2_B2_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2883651, upper bound: 42.2898971
NS_A2_B2_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2876179, upper bound: 42.2902789
NS_A2_B2_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2883651, upper bound: 42.2898971
NS_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2869462, upper bound: 42.2928380
NS_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2877461, upper bound: 42.2958219
NS_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2869461, upper bound: 42.2928381
NS_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2877461, upper bound: 42.2958219
NS_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2913285, upper bound: 42.2882214
NS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2922509, upper bound: 42.2913651
NS_A2_B2_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2908605, upper bound: 42.2904341
NS_A2_B2_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2908905, upper bound: 42.2900326
NS_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2908605, upper bound: 42.2904341
NS_A2_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2908905, upper bound: 42.2900326
NS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2904226, upper bound: 42.2930337
NS_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2906377, upper bound: 42.2959029
NS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2904226, upper bound: 42.2930338
NS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.88
Output dim: 3, lower bound: -42.2906377, upper bound: 42.2959029

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -84.2522964, 141.5968475, -230.4356995, 234.3409271
1: -15.9774551, 21.5835915, -14.9411621, 20.4636497, -36.4410973, 36.5247498
2: -12.1512346, 19.9041157, -11.4510040, 18.8597946, -31.0110283, 31.3551178
3: -12.9723568, 33.5429039, -12.1371870, 31.7596703, -44.7320251, 45.6800919
4: -10.2096796, 25.1019897, -9.6021099, 23.9315891, -34.1412697, 34.7041016

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2726245, upper bound: 42.2899048
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2654744, upper bound: 42.2864545
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2716317, upper bound: 42.2886242
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -84.2522964, 141.5968475, -233.8305817, 240.0310669
1: -16.6187267, 22.4468498, -14.9411621, 20.4636497, -37.0823708, 37.3880119
2: -12.6434240, 20.6275883, -11.4510040, 18.8597946, -31.5032120, 32.0785904
3: -13.4623756, 34.7367935, -12.1371870, 31.7596703, -45.2220383, 46.8739815
4: -10.6144371, 26.0106106, -9.6021099, 23.9315891, -34.5460281, 35.6127205

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2696748, upper bound: 42.2859049
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2767755, upper bound: 42.2924932
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -84.5565338, 142.2301331, -231.0690002, 234.6451721
1: -15.9774551, 21.5835915, -15.0061140, 20.5480175, -36.5254707, 36.5897064
2: -12.1512346, 19.9041157, -11.4953394, 18.9423313, -31.0935669, 31.3994522
3: -12.9723568, 33.5429039, -12.2066050, 31.8855362, -44.8578911, 45.7495079
4: -10.2096796, 25.1019897, -9.6422052, 24.0349655, -34.2446442, 34.7441940

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2797844, upper bound: 42.2899396
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2871066, upper bound: 42.2897839
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -84.5565338, 142.2301331, -234.4638672, 240.3353271
1: -16.6187267, 22.4468498, -15.0061140, 20.5480175, -37.1667442, 37.4529648
2: -12.6434240, 20.6275883, -11.4953394, 18.9423313, -31.5857544, 32.1229286
3: -13.4623756, 34.7367935, -12.2066050, 31.8855362, -45.3479080, 46.9433975
4: -10.6144371, 26.0106106, -9.6422052, 24.0349655, -34.6494026, 35.6528130

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2845797, upper bound: 42.2867290
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2883573, upper bound: 42.2924996
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -87.8565369, 148.2602386, -85.5928268, 142.3337250, -230.1902618, 233.8530579
1: -15.7788296, 21.3406467, -14.9906960, 20.5860748, -36.3649025, 36.3313408
2: -12.0125332, 19.6584721, -11.5474100, 19.0303516, -31.0428848, 31.2058811
3: -12.8138676, 33.1243134, -12.2445526, 32.0809517, -44.8948212, 45.3688622
4: -10.0904112, 24.8035183, -9.6921005, 24.1245098, -34.2149162, 34.4956207

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2661201, upper bound: 42.2835073
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2856663, upper bound: 42.2899754
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -91.0659866, 153.6119080, -85.5928268, 142.3337250, -233.3997192, 239.2047272
1: -16.3823013, 22.1608734, -14.9906960, 20.5860748, -36.9683762, 37.1515694
2: -12.4787483, 20.3399181, -11.5474100, 19.0303516, -31.5091000, 31.8873291
3: -13.2771797, 34.2468834, -12.2445526, 32.0809517, -45.3581314, 46.4914360
4: -10.4732609, 25.6666126, -9.6921005, 24.1245098, -34.5977707, 35.3587036

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2843158, upper bound: 42.2859810
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2880350, upper bound: 42.2900910
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -87.8565369, 148.2602386, -86.5624771, 144.3015137, -232.1580505, 234.8227234
1: -15.7788296, 21.3406467, -15.2155390, 20.8484764, -36.6273041, 36.5561752
2: -12.0125332, 19.6584721, -11.6986370, 19.2733593, -31.2858868, 31.3571033
3: -12.8138676, 33.1243134, -12.4184294, 32.4693947, -45.2832642, 45.5427361
4: -10.0904112, 24.8035183, -9.8197441, 24.4203224, -34.5107346, 34.6232605

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2776788, upper bound: 42.2834850
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2875161, upper bound: 42.2891890
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -91.0659866, 153.6119080, -86.5624771, 144.3015137, -235.3674927, 240.1743774
1: -16.3823013, 22.1608734, -15.2155390, 20.8484764, -37.2307777, 37.3764038
2: -12.4787483, 20.3399181, -11.6986370, 19.2733593, -31.7521057, 32.0385551
3: -13.2771797, 34.2468834, -12.4184294, 32.4693947, -45.7465744, 46.6653099
4: -10.4732609, 25.6666126, -9.8197441, 24.4203224, -34.8935852, 35.4863586

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2843578, upper bound: 42.2845817
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2916447, upper bound: 42.2893980
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -85.5928268, 142.3337250, -231.1726074, 235.6815033
1: -15.9774551, 21.5835915, -14.9906960, 20.5860748, -36.5635300, 36.5742874
2: -12.1512346, 19.9041157, -11.5474100, 19.0303516, -31.1815872, 31.4515228
3: -12.9723568, 33.5429039, -12.2445526, 32.0809517, -45.0533066, 45.7874565
4: -10.2096796, 25.1019897, -9.6921005, 24.1245098, -34.3341904, 34.7940865

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2833785, upper bound: 42.2935368
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2831786, upper bound: 42.2905480
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2654744, upper bound: 42.2864545
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2850103, upper bound: 42.2927156
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -85.5928268, 142.3337250, -234.5674591, 241.3716278
1: -16.6187267, 22.4468498, -14.9906960, 20.5860748, -37.2048035, 37.4375458
2: -12.6434240, 20.6275883, -11.5474100, 19.0303516, -31.6737747, 32.1749992
3: -13.4623756, 34.7367935, -12.2445526, 32.0809517, -45.5433273, 46.9813461
4: -10.6144371, 26.0106106, -9.6921005, 24.1245098, -34.7389450, 35.7027130

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2843158, upper bound: 42.2926947
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2874595, upper bound: 42.2960022
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -86.5624771, 144.3015137, -233.1403809, 236.6511536
1: -15.9774551, 21.5835915, -15.2155390, 20.8484764, -36.8259315, 36.7991257
2: -12.1512346, 19.9041157, -11.6986370, 19.2733593, -31.4245911, 31.6027470
3: -12.9723568, 33.5429039, -12.4184294, 32.4693947, -45.4417496, 45.9613342
4: -10.2096796, 25.1019897, -9.8197441, 24.4203224, -34.6300011, 34.9217339

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2857995, upper bound: 42.2928121
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2871066, upper bound: 42.2908706
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -86.5624771, 144.3015137, -236.5352478, 242.3412781
1: -16.6187267, 22.4468498, -15.2155390, 20.8484764, -37.4672012, 37.6623878
2: -12.6434240, 20.6275883, -11.6986370, 19.2733593, -31.9167786, 32.3262253
3: -13.4623756, 34.7367935, -12.4184294, 32.4693947, -45.9317703, 47.1552162
4: -10.6144371, 26.0106106, -9.8197441, 24.4203224, -35.0347595, 35.8303528

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2847223, upper bound: 42.2919018
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2885189, upper bound: 42.2953227
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -90.9440155, 154.9081116, -243.7469635, 241.0326691
1: -15.9774551, 21.5835915, -16.5433254, 22.3302612, -38.3077087, 38.1269150
2: -12.1512346, 19.9041157, -12.5446377, 20.4681129, -32.6193466, 32.4487534
3: -12.9723568, 33.5429039, -13.3658085, 34.4641609, -47.4365158, 46.9087143
4: -10.2096796, 25.1019897, -10.5271111, 25.7950706, -36.0047493, 35.6290970

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -90.9440155, 154.9081116, -247.1418457, 246.7227631
1: -16.6187267, 22.4468498, -16.5433254, 22.3302612, -38.9489861, 38.9901733
2: -12.6434240, 20.6275883, -12.5446377, 20.4681129, -33.1115303, 33.1722221
3: -13.4623756, 34.7367935, -13.3658085, 34.4641609, -47.9265366, 48.1026001
4: -10.6144371, 26.0106106, -10.5271111, 25.7950706, -36.4095078, 36.5377197

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -87.8565369, 148.2602386, -91.5973816, 154.5887604, -242.4452972, 239.8576202
1: -15.7788296, 21.3406467, -16.4933624, 22.3109341, -38.0897636, 37.8340073
2: -12.0125332, 19.6584721, -12.5618811, 20.4642582, -32.4767876, 32.2203522
3: -12.8138676, 33.1243134, -13.3621187, 34.4498138, -47.2636795, 46.4864273
4: -10.0904112, 24.8035183, -10.5422421, 25.8184700, -35.9088783, 35.3457603

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2904582, upper bound: 42.2902151
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2904582, upper bound: 42.2906355
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -91.0659866, 153.6119080, -91.5973816, 154.5887604, -245.6547546, 245.2092896
1: -16.3823013, 22.1608734, -16.4933624, 22.3109341, -38.6932373, 38.6542358
2: -12.4787483, 20.3399181, -12.5618811, 20.4642582, -32.9430084, 32.9017982
3: -13.2771797, 34.2468834, -13.3621187, 34.4498138, -47.7269897, 47.6089973
4: -10.4732609, 25.6666126, -10.5422421, 25.8184700, -36.2917328, 36.2088509

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2904582, upper bound: 42.2902151
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2904582, upper bound: 42.2906355
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -87.8565369, 148.2602386, -92.7636185, 156.7567902, -244.6133270, 241.0238342
1: -15.7788296, 21.3406467, -16.7301445, 22.5964622, -38.3752899, 38.0707855
2: -12.0125332, 19.6584721, -12.7266340, 20.7519531, -32.7644882, 32.3851013
3: -12.8138676, 33.1243134, -13.5472450, 34.9396820, -47.7535477, 46.6715546
4: -10.0904112, 24.8035183, -10.6834011, 26.1624432, -36.2528534, 35.4869194

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2905848, upper bound: 42.2895908
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2905848, upper bound: 42.2900326
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -91.0659866, 153.6119080, -92.7636185, 156.7567902, -247.8227539, 246.3755035
1: -16.3823013, 22.1608734, -16.7301445, 22.5964622, -38.9787636, 38.8910141
2: -12.4787483, 20.3399181, -12.7266340, 20.7519531, -33.2307014, 33.0665512
3: -13.2771797, 34.2468834, -13.5472450, 34.9396820, -48.2168541, 47.7941246
4: -10.4732609, 25.6666126, -10.6834011, 26.1624432, -36.6357040, 36.3500061

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2905848, upper bound: 42.2895908
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2905848, upper bound: 42.2900326
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -91.5973816, 154.5887604, -243.4276428, 241.6860504
1: -15.9774551, 21.5835915, -16.4933624, 22.3109341, -38.2883911, 38.0769501
2: -12.1512346, 19.9041157, -12.5618811, 20.4642582, -32.6154900, 32.4659920
3: -12.9723568, 33.5429039, -13.3621187, 34.4498138, -47.4221687, 46.9050217
4: -10.2096796, 25.1019897, -10.5422421, 25.8184700, -36.0281448, 35.6442261

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2904165, upper bound: 42.2930619
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2904165, upper bound: 42.2935907
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -91.5973816, 154.5887604, -246.8224945, 247.3761749
1: -16.6187267, 22.4468498, -16.4933624, 22.3109341, -38.9296608, 38.9402122
2: -12.6434240, 20.6275883, -12.5618811, 20.4642582, -33.1076775, 33.1894646
3: -13.4623756, 34.7367935, -13.3621187, 34.4498138, -47.9121857, 48.0989037
4: -10.6144371, 26.0106106, -10.5422421, 25.8184700, -36.4329071, 36.5528526

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2904241, upper bound: 42.2951542
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2904241, upper bound: 42.2964182
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -92.7636185, 156.7567902, -245.5956421, 242.8522644
1: -15.9774551, 21.5835915, -16.7301445, 22.5964622, -38.5739174, 38.3137321
2: -12.1512346, 19.9041157, -12.7266340, 20.7519531, -32.9031868, 32.6307449
3: -12.9723568, 33.5429039, -13.5472450, 34.9396820, -47.9120331, 47.0901489
4: -10.2096796, 25.1019897, -10.6834011, 26.1624432, -36.3721237, 35.7853851

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2908445, upper bound: 42.2924055
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2908445, upper bound: 42.2930337
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -92.7636185, 156.7567902, -248.9905090, 248.5423584
1: -16.6187267, 22.4468498, -16.7301445, 22.5964622, -39.2151871, 39.1769943
2: -12.6434240, 20.6275883, -12.7266340, 20.7519531, -33.3953705, 33.3542137
3: -13.4623756, 34.7367935, -13.5472450, 34.9396820, -48.4020500, 48.2840347
4: -10.6144371, 26.0106106, -10.6834011, 26.1624432, -36.7768784, 36.6940117

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2908593, upper bound: 42.2945148
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2908593, upper bound: 42.2959029
time: 0.64 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 3.93 seconds
NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2654744, upper bound: 42.2864545
NS_A2_B2_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2716317, upper bound: 42.2886242
NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2696748, upper bound: 42.2859049
NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2767755, upper bound: 42.2924932
NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2797844, upper bound: 42.2899396
NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2871066, upper bound: 42.2897839
NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2845797, upper bound: 42.2867290
NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2883573, upper bound: 42.2924996
NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2661201, upper bound: 42.2835073
NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2856663, upper bound: 42.2899754
NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2843158, upper bound: 42.2859810
NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2880350, upper bound: 42.2900910
NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2776788, upper bound: 42.2834850
NS_A2_B2_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2875161, upper bound: 42.2891890
NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2843578, upper bound: 42.2845817
NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2916447, upper bound: 42.2893980
NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2654744, upper bound: 42.2864545
NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2850103, upper bound: 42.2927156
NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2843158, upper bound: 42.2926947
NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2874595, upper bound: 42.2960022
NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2857995, upper bound: 42.2928121
NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2871066, upper bound: 42.2908706
NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2847223, upper bound: 42.2919018
NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2885189, upper bound: 42.2953227
NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2904582, upper bound: 42.2902151
NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2904582, upper bound: 42.2906355
NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2904582, upper bound: 42.2902151
NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2904582, upper bound: 42.2906355
NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2905848, upper bound: 42.2895908
NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2905848, upper bound: 42.2900326
NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2905848, upper bound: 42.2895908
NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2905848, upper bound: 42.2900326
NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2904165, upper bound: 42.2930619
NS_A2_B2_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2904165, upper bound: 42.2935907
NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2904241, upper bound: 42.2951542
NS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2904241, upper bound: 42.2964182
NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2908445, upper bound: 42.2924055
NS_A2_B2_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2908445, upper bound: 42.2930337
NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2908593, upper bound: 42.2945148
NS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.93
Output dim: 3, lower bound: -42.2908593, upper bound: 42.2959029

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -84.0043869, 141.1097412, -233.3434753, 239.7831726
1: -16.6187267, 22.4468498, -14.8851252, 20.3888416, -37.0075684, 37.3319702
2: -12.6434240, 20.6275883, -11.4091396, 18.8010769, -31.4444981, 32.0367279
3: -13.4623756, 34.7367935, -12.0918798, 31.6670876, -45.1294632, 46.8286667
4: -10.6144371, 26.0106106, -9.5673609, 23.8584099, -34.4728470, 35.5779724

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2767252, upper bound: 42.2899964
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2767252, upper bound: 42.2924932
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -82.1584320, 138.1771240, -227.0159760, 232.2470856
1: -15.9774551, 21.5835915, -14.5676260, 19.9160347, -35.8934860, 36.1512146
2: -12.1512346, 19.9041157, -11.1428814, 18.4110985, -30.5623322, 31.0469971
3: -12.9723568, 33.5429039, -11.8499060, 31.0118618, -43.9842186, 45.3928108
4: -10.2096796, 25.1019897, -9.3512535, 23.3689785, -33.5786514, 34.4532433

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2797844, upper bound: 42.2896994
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2797844, upper bound: 42.2897839
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -88.4452896, 149.4085236, -238.2473907, 238.5339661
1: -15.9774551, 21.5835915, -15.7988329, 21.5715942, -37.5490456, 37.3824196
2: -12.1512346, 19.9041157, -12.0960321, 19.8623848, -32.0136185, 32.0001488
3: -12.9723568, 33.5429039, -12.8286610, 33.3805885, -46.3529434, 46.3715668
4: -10.2096796, 25.1019897, -10.1384697, 25.1978149, -35.4074936, 35.2404594

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2866360, upper bound: 42.2896994
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2866360, upper bound: 42.2897839
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -84.3076172, 141.7448273, -233.9785614, 240.0864258
1: -16.6187267, 22.4468498, -14.9501991, 20.4733162, -37.0920410, 37.3970490
2: -12.6434240, 20.6275883, -11.4534969, 18.8836308, -31.5270538, 32.0810852
3: -13.4623756, 34.7367935, -12.1613541, 31.7922535, -45.2546234, 46.8981361
4: -10.6144371, 26.0106106, -9.6074724, 23.9618416, -34.5762787, 35.6180840

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2881557, upper bound: 42.2899948
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2881557, upper bound: 42.2924995
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -87.8565369, 148.2602386, -85.4100189, 142.0038910, -229.8604279, 233.6702576
1: -15.7788296, 21.3406467, -14.9541483, 20.5368729, -36.3157005, 36.2947922
2: -12.0125332, 19.6584721, -11.5190477, 18.9898376, -31.0023689, 31.1775208
3: -12.8138676, 33.1243134, -12.2158871, 32.0147133, -44.8285828, 45.3401985
4: -10.0904112, 24.8035183, -9.6688738, 24.0742645, -34.1646767, 34.4723930

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -91.0659866, 153.6119080, -85.3486633, 141.8551788, -232.9211731, 238.9605713
1: -16.3823013, 22.1608734, -14.9350080, 20.5122375, -36.8945389, 37.0958786
2: -12.4787483, 20.3399181, -11.5057383, 18.9729424, -31.4516907, 31.8456554
3: -13.2771797, 34.2468834, -12.1990309, 31.9901772, -45.2673569, 46.4459152
4: -10.4732609, 25.6666126, -9.6574984, 24.0528316, -34.5260925, 35.3241119

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2877811, upper bound: 42.2875601
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2877811, upper bound: 42.2900910
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -91.0659866, 153.6119080, -86.3044510, 143.7938538, -234.8598328, 239.9163513
1: -16.3823013, 22.1608734, -15.1550951, 20.7709827, -37.1532822, 37.3159676
2: -12.4787483, 20.3399181, -11.6547279, 19.2128544, -31.6916027, 31.9946461
3: -13.2771797, 34.2468834, -12.3708429, 32.3723221, -45.6495018, 46.6177254
4: -10.4732609, 25.6666126, -9.7832890, 24.3459301, -34.8191872, 35.4498978

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2913909, upper bound: 42.2866932
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2913909, upper bound: 42.2893980
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -85.4100189, 142.0038910, -230.8427582, 235.4987030
1: -15.9774551, 21.5835915, -14.9541483, 20.5368729, -36.5143280, 36.5377388
2: -12.1512346, 19.9041157, -11.5190477, 18.9898376, -31.1410713, 31.4231644
3: -12.9723568, 33.5429039, -12.2158871, 32.0147133, -44.9870682, 45.7587891
4: -10.2096796, 25.1019897, -9.6688738, 24.0742645, -34.2839432, 34.7708626

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -79.3272247, 132.1277924, -224.3615265, 235.1060181
1: -16.6187267, 22.4468498, -13.7443905, 19.0133953, -35.6321144, 36.1912384
2: -12.6434240, 20.6275883, -10.6429863, 17.7980385, -30.4414616, 31.2705669
3: -13.4623756, 34.7367935, -11.2251892, 29.8933907, -43.3557663, 45.9619827
4: -10.6144371, 26.0106106, -8.8965445, 22.5877476, -33.2021828, 34.9071541

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2843158, upper bound: 42.2926947
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2843158, upper bound: 42.2926947
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -85.3486633, 141.8551788, -234.0889130, 241.1274719
1: -16.6187267, 22.4468498, -14.9350080, 20.5122375, -37.1309662, 37.3818550
2: -12.6434240, 20.6275883, -11.5057383, 18.9729424, -31.6163616, 32.1333237
3: -13.4623756, 34.7367935, -12.1990309, 31.9901772, -45.4525452, 46.9358215
4: -10.6144371, 26.0106106, -9.6574984, 24.0528316, -34.6672668, 35.6681099

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2874092, upper bound: 42.2932192
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2874092, upper bound: 42.2960022
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -84.1436920, 140.1779022, -229.0167694, 234.2323303
1: -15.9774551, 21.5835915, -14.7563162, 20.2128735, -36.1903267, 36.3399086
2: -12.1512346, 19.9041157, -11.3433113, 18.7373219, -30.8885574, 31.2474213
3: -12.9723568, 33.5429039, -12.0597258, 31.5847816, -44.5571365, 45.6026268
4: -10.2096796, 25.1019897, -9.5258999, 23.7586212, -33.9682999, 34.6278877

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2857995, upper bound: 42.2906482
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2857995, upper bound: 42.2908705
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -88.8388824, 150.0886841, -90.6792297, 152.2106171, -241.0494690, 240.7679138
1: -15.9774551, 21.5835915, -16.1383705, 21.9492912, -37.9267426, 37.7219582
2: -12.1512346, 19.9041157, -12.3531570, 20.2570972, -32.4083328, 32.2572708
3: -12.9723568, 33.5429039, -13.0955219, 34.0648232, -47.0371780, 46.6384277
4: -10.2096796, 25.1019897, -10.3613548, 25.6371899, -35.8468704, 35.4633446

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2870385, upper bound: 42.2906482
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2870385, upper bound: 42.2908706
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -80.0819550, 133.5184326, -225.7521667, 235.8607635
1: -16.6187267, 22.4468498, -13.9032040, 19.2114315, -35.8301582, 36.3500519
2: -12.6434240, 20.6275883, -10.7579861, 17.9829807, -30.6264019, 31.3855724
3: -13.4623756, 34.7367935, -11.3570356, 30.1943436, -43.6567192, 46.0938301
4: -10.6144371, 26.0106106, -8.9949493, 22.8206730, -33.4351120, 35.0055618

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2847223, upper bound: 42.2919018
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2847223, upper bound: 42.2919018
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -92.2337341, 155.7788086, -86.3044510, 143.7938538, -236.0275879, 242.0832520
1: -16.6187267, 22.4468498, -15.1550951, 20.7709827, -37.3897057, 37.6019440
2: -12.6434240, 20.6275883, -11.6547279, 19.2128544, -31.8562756, 32.2823181
3: -13.4623756, 34.7367935, -12.3708429, 32.3723221, -45.8346977, 47.1076355
4: -10.6144371, 26.0106106, -9.7832890, 24.3459301, -34.9603653, 35.7938995

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2883743, upper bound: 42.2923723
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2883743, upper bound: 42.2953227
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -87.8565369, 148.2602386, -87.8565369, 148.2602386, -236.1167755, 236.1167755
1: -15.7788296, 21.3406467, -15.7788296, 21.3406467, -37.1194687, 37.1194687
2: -12.0125332, 19.6584721, -12.0125332, 19.6584721, -31.6710052, 31.6710033
3: -12.8138676, 33.1243134, -12.8138676, 33.1243134, -45.9381790, 45.9381790
4: -10.0904112, 24.8035183, -10.0904112, 24.8035183, -34.8939285, 34.8939285

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2896688, upper bound: 42.2867385
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2900152, upper bound: 42.2898294
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -87.8565369, 148.2602386, -91.0659866, 153.6119080, -241.4684448, 239.3262329
1: -15.7788296, 21.3406467, -16.3823013, 22.1608734, -37.9397049, 37.7229462
2: -12.0125332, 19.6584721, -12.4787483, 20.3399181, -32.3524513, 32.1372223
3: -12.8138676, 33.1243134, -13.2771797, 34.2468834, -47.0607529, 46.4014893
4: -10.0904112, 24.8035183, -10.4732609, 25.6666126, -35.7570229, 35.2767792

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2896688, upper bound: 42.2875545
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2900152, upper bound: 42.2898294
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -91.0659866, 153.6119080, -87.8565369, 148.2602386, -239.3262329, 241.4684448
1: -16.3823013, 22.1608734, -15.7788296, 21.3406467, -37.7229462, 37.9397011
2: -12.4787483, 20.3399181, -12.0125332, 19.6584721, -32.1372223, 32.3524513
3: -13.2771797, 34.2468834, -12.8138676, 33.1243134, -46.4014893, 47.0607529
4: -10.4732609, 25.6666126, -10.0904112, 24.8035183, -35.2767792, 35.7570190

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2896664, upper bound: 42.2866347
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897687, upper bound: 42.2896352
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -91.0659866, 153.6119080, -91.0659866, 153.6119080, -244.6778870, 244.6778870
1: -16.3823013, 22.1608734, -16.3823013, 22.1608734, -38.5431747, 38.5431747
2: -12.4787483, 20.3399181, -12.4787483, 20.3399181, -32.8186646, 32.8186646
3: -13.2771797, 34.2468834, -13.2771797, 34.2468834, -47.5240631, 47.5240631
4: -10.4732609, 25.6666126, -10.4732609, 25.6666126, -36.1398735, 36.1398735

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2896664, upper bound: 42.2869368
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897687, upper bound: 42.2897041
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -87.8565369, 148.2602386, -88.8388824, 150.0886841, -237.9452209, 237.0991058
1: -15.7788296, 21.3406467, -15.9774551, 21.5835915, -37.3624153, 37.3180923
2: -12.0125332, 19.6584721, -12.1512346, 19.9041157, -31.9166451, 31.8097057
3: -12.8138676, 33.1243134, -12.9723568, 33.5429039, -46.3567734, 46.0966682
4: -10.0904112, 24.8035183, -10.2096796, 25.1019897, -35.1924019, 35.0131989

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903812, upper bound: 42.2860566
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2906057, upper bound: 42.2891727
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -87.8565369, 148.2602386, -92.2337341, 155.7788086, -243.6353455, 240.4939728
1: -15.7788296, 21.3406467, -16.6187267, 22.4468498, -38.2256775, 37.9593697
2: -12.0125332, 19.6584721, -12.6434240, 20.6275883, -32.6401215, 32.3018913
3: -12.8138676, 33.1243134, -13.4623756, 34.7367935, -47.5506592, 46.5866852
4: -10.0904112, 24.8035183, -10.6144371, 26.0106106, -36.1010208, 35.4179535

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2903812, upper bound: 42.2869790
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2906057, upper bound: 42.2897779
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -91.0659866, 153.6119080, -88.8388824, 150.0886841, -241.1546631, 242.4507599
1: -16.3823013, 22.1608734, -15.9774551, 21.5835915, -37.9658928, 38.1383286
2: -12.4787483, 20.3399181, -12.1512346, 19.9041157, -32.3828659, 32.4911537
3: -13.2771797, 34.2468834, -12.9723568, 33.5429039, -46.8200836, 47.2192383
4: -10.4732609, 25.6666126, -10.2096796, 25.1019897, -35.5752487, 35.8762894

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898384, upper bound: 42.2859713
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898285, upper bound: 42.2890112
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -91.0659866, 153.6119080, -92.2337341, 155.7788086, -246.8447876, 245.8456421
1: -16.3823013, 22.1608734, -16.6187267, 22.4468498, -38.8291512, 38.7795982
2: -12.4787483, 20.3399181, -12.6434240, 20.6275883, -33.1063385, 32.9833412
3: -13.2771797, 34.2468834, -13.4623756, 34.7367935, -48.0139694, 47.7092590
4: -10.4732609, 25.6666126, -10.6144371, 26.0106106, -36.4838715, 36.2810478

Time for backsubstitution: 2.49 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.17 + 416.78 = 420.95 seconds
