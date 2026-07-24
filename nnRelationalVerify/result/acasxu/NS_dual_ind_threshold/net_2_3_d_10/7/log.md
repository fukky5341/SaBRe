## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 465.361891711094


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480)
1: (-100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447)
2: (-110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621)
3: (-99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465)
4: (-158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.74 + 2.13 = 3.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -465.3898151, upper bound: 465.3898151

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3892586, upper bound: 465.3897715
time: 0.99 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3892635, upper bound: 465.3892635
time: 0.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.80 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 0, lower bound: -465.3892586, upper bound: 465.3897715
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 0, lower bound: -465.3892635, upper bound: 465.3892635

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -124.9379730, 360.6490173, -135.3351440, 390.9148865, -515.8527832, 495.9841309
1: -88.8310776, 226.5958557, -96.2211838, 245.2118530, -334.0429077, 322.8170471
2: -97.4581146, 209.4259796, -105.4396896, 226.6281738, -324.0863037, 314.8656006
3: -87.8858337, 270.6470032, -95.0952530, 292.9920349, -380.8778687, 365.7422485
4: -140.4488525, 221.7328949, -151.9560242, 239.9570923, -380.4059143, 373.6889038

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3897683
time: 0.71 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3897592
time: 0.68 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -135.5962219, 392.8481140, -138.8964996, 401.8407288, -537.4369507, 531.7445679
1: -96.5429153, 245.9777069, -98.8086090, 251.7576447, -348.3005371, 344.7863159
2: -105.5482788, 227.4414368, -108.1267242, 232.7208099, -338.2690125, 335.5681763
3: -95.2409973, 293.9291077, -97.5721207, 300.8709412, -396.1119385, 391.5012207
4: -152.4237976, 240.6746979, -155.9930725, 246.3488922, -398.7727051, 396.6677246

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889456, upper bound: 465.3892562
time: 0.91 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889542, upper bound: 465.3892286
time: 0.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.34 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3897683
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3897592
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -465.3889456, upper bound: 465.3892562
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.34
Output dim: 0, lower bound: -465.3889542, upper bound: 465.3892286

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -114.9707947, 332.4857788, -115.7249451, 335.8876343, -450.8583984, 448.2107239
1: -81.8756256, 208.6849976, -82.5744858, 210.1385040, -292.0141296, 291.2594604
2: -90.1034622, 192.8183441, -90.9898987, 194.1059875, -284.2093811, 283.8082275
3: -81.1417618, 249.5965729, -81.8434753, 251.7733459, -332.9151001, 331.4400635
4: -129.7347565, 204.4361725, -130.9807739, 206.0576782, -335.7924194, 335.4169312

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3858042, upper bound: 465.3892723
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3895932
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3897592
time: 0.79 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -121.4597626, 350.4768372, -173.5142059, 504.7874146, -624.8697510, 523.9909058
1: -86.3611298, 220.3018646, -123.6216507, 315.1566467, -401.5177307, 343.4882812
2: -94.7813950, 203.5743866, -135.0269470, 291.0980835, -385.8794861, 338.4497986
3: -85.4767761, 263.1320496, -122.1035385, 377.1527710, -462.2721863, 385.2355957
4: -136.5728302, 215.5892334, -194.3131714, 309.2504883, -445.8233032, 409.5017700

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3858539, upper bound: 465.3891742
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3809313, upper bound: 465.3876591
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3807707, upper bound: 465.3845662
time: 0.73 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -126.0175400, 365.8258667, -119.0943375, 346.1478577, -472.1654053, 484.9201965
1: -89.8756027, 228.7751465, -85.0260620, 216.3020782, -306.1776733, 313.8012085
2: -98.4746170, 211.4781494, -93.5035095, 199.8215485, -298.2961426, 304.9816589
3: -88.7607498, 273.7176208, -84.1698914, 259.1827393, -347.9434814, 357.8874817
4: -142.1277313, 224.0521851, -134.7596130, 212.0496216, -354.1773682, 358.8117981

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3858021, upper bound: 465.3891624
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3857851, upper bound: 465.3858509
time: 0.83 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -131.8397064, 381.7165833, -177.4099426, 516.1660156, -646.9955444, 559.1265259
1: -93.8738708, 239.1234894, -126.4682693, 322.0534668, -415.9273376, 365.4156494
2: -102.6521988, 221.0408325, -137.9620209, 297.4664612, -400.1186218, 358.9809875
3: -92.6292419, 285.7597046, -124.7713547, 385.4618835, -477.9682922, 410.5310669
4: -148.2069092, 233.9802094, -198.5434723, 316.0032043, -464.2101135, 432.2339783

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3809352, upper bound: 465.3870828
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3804886, upper bound: 465.3804886
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.23 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3895932
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3897592
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -465.3809313, upper bound: 465.3876591
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -465.3807707, upper bound: 465.3845662
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -465.3858021, upper bound: 465.3891624
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -465.3857851, upper bound: 465.3858509
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -465.3809352, upper bound: 465.3870828
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 0, lower bound: -465.3804886, upper bound: 465.3804886

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -104.8477325, 304.1072693, -115.7249451, 335.8876343, -440.7353516, 419.8322144
1: -74.8315887, 190.6109619, -82.5744858, 210.1385040, -284.9700928, 273.1854553
2: -82.6269760, 176.0292969, -90.9898987, 194.1059875, -276.7329102, 267.0191956
3: -74.2861862, 228.3231964, -81.8434753, 251.7733459, -326.0595093, 310.1666870
4: -118.8996887, 186.9278564, -130.9807739, 206.0576782, -324.9573669, 317.9086304

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3896104
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3896107
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -160.9913940, 468.3616333, -115.7249451, 335.8876343, -496.8789368, 582.7871094
1: -114.7227402, 292.6838074, -82.5744858, 210.1385040, -324.4157715, 375.2583008
2: -125.5135803, 270.2168274, -90.9898987, 194.1059875, -319.4066162, 361.2067261
3: -113.4593964, 350.2796631, -81.8434753, 251.7733459, -365.2327271, 431.7456970
4: -180.5515442, 287.2357483, -130.9807739, 206.0576782, -386.1586304, 418.2164917

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3897682
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3897683
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -113.7575150, 328.7003784, -173.2542877, 504.0447083, -616.4133301, 501.9546509
1: -80.9861374, 206.6053009, -123.4392548, 314.6941528, -395.6802673, 329.5967712
2: -89.0871124, 190.9403381, -134.8439789, 290.6593018, -379.7463989, 325.6250916
3: -80.2338257, 246.8856201, -121.9289169, 376.6089478, -456.4794006, 368.8145447
4: -128.3227997, 202.2826538, -194.0486298, 308.7947083, -437.1174622, 395.9214172

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3807707, upper bound: 465.3845662
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3807707, upper bound: 465.3845662
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -144.8214569, 422.5209351, -168.6903534, 490.6096802, -633.4849854, 590.0923462
1: -103.4686661, 263.7456055, -120.1782608, 306.4163208, -409.4952087, 383.2546082
2: -113.9267807, 244.0141602, -131.3672180, 282.9039307, -396.5194092, 375.0414124
3: -102.4696884, 316.0253601, -118.7566833, 366.7396851, -468.5375977, 434.5482483
4: -163.9672089, 258.7939758, -189.0064697, 300.6584167, -464.1654053, 447.3608398

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3784931, upper bound: 465.3838715
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3807707, upper bound: 465.3845662
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -120.3880005, 349.7455444, -117.0759659, 340.3104248, -460.6984253, 466.8215027
1: -85.8299942, 218.7897644, -83.5670013, 212.6948090, -298.5247498, 302.3567505
2: -94.0457916, 202.3363800, -91.9010010, 196.5158844, -290.5616150, 294.2373657
3: -84.7843475, 261.6694946, -82.7354126, 254.8205261, -339.6048584, 344.4049072
4: -135.8403168, 214.2314453, -132.4740143, 208.4970703, -344.3373718, 346.7054443

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3835739, upper bound: 465.3885465
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3853840, upper bound: 465.3887896
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -129.9508667, 380.1845398, -114.9673843, 334.8294983, -464.7803650, 495.1518860
1: -92.6972809, 237.5349426, -82.1479568, 209.1412048, -301.8384705, 319.6828308
2: -101.7429352, 220.4954529, -90.4017639, 193.2857513, -295.0286865, 310.8972168
3: -91.6481552, 283.8942871, -81.3451996, 250.6168518, -342.2649536, 365.2394409
4: -147.3830566, 232.4895172, -130.3932800, 205.0456696, -352.4287109, 362.8828125

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3835427, upper bound: 465.3854119
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3853641, upper bound: 465.3854437
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -124.8085785, 361.9147034, -177.1608276, 515.4533081, -639.2384033, 539.0755615
1: -88.9823761, 226.6152496, -126.2929459, 321.6090393, -410.5914307, 352.7250366
2: -97.4444351, 209.4962616, -137.7862549, 297.0456543, -394.4900818, 347.2563782
3: -87.8468933, 270.9575195, -124.6035461, 384.9398499, -472.6610718, 395.5610657
4: -140.6762238, 221.8431854, -198.2893372, 315.5655212, -456.2416687, 419.8332214

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3808853, upper bound: 465.3842584
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3771904, upper bound: 465.3838926
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -157.9335327, 460.4708862, -172.3631897, 501.3425903, -657.6106567, 631.8847046
1: -112.8128586, 286.8265381, -122.8648148, 312.9160767, -425.5855713, 409.3231506
2: -123.7316208, 265.3747253, -134.1410675, 288.8995667, -412.5149231, 399.3048706
3: -111.3689575, 343.7764282, -121.2701874, 374.5761719, -485.4862366, 465.0465393
4: -178.1462097, 281.3798828, -193.0043640, 307.0215149, -484.8643494, 474.0551758

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3803070, upper bound: 465.3767837
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3766432, upper bound: 465.3766432
time: 0.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.20 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3896104
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3896107
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3897682
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3889498, upper bound: 465.3897683
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3807707, upper bound: 465.3845662
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3807707, upper bound: 465.3845662
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3784931, upper bound: 465.3838715
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3807707, upper bound: 465.3845662
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3835739, upper bound: 465.3885465
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3853840, upper bound: 465.3887896
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3835427, upper bound: 465.3854119
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3853641, upper bound: 465.3854437
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3808853, upper bound: 465.3842584
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3771904, upper bound: 465.3838926
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3803070, upper bound: 465.3767837
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.20
Output dim: 0, lower bound: -465.3766432, upper bound: 465.3766432

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -104.8477325, 304.1072693, -104.8477325, 304.1072693, -408.9550171, 408.9550171
1: -74.8315887, 190.6109619, -74.8315887, 190.6109619, -265.4425659, 265.4425659
2: -82.6269760, 176.0292969, -82.6269760, 176.0292969, -258.6562805, 258.6562805
3: -74.2861862, 228.3231964, -74.2861862, 228.3231964, -302.6093445, 302.6093445
4: -118.8996887, 186.9278564, -118.8996887, 186.9278564, -305.8275452, 305.8275452

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889696, upper bound: 465.3889562
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889535, upper bound: 465.3895744
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -104.8477325, 304.1072693, -115.7158203, 336.8948059, -441.7425537, 419.8230896
1: -74.8315887, 190.6109619, -82.7079086, 210.3442993, -285.1759033, 273.3188782
2: -82.6269760, 176.0292969, -90.8642197, 194.3667755, -276.9937134, 266.8935242
3: -74.2861862, 228.3231964, -81.7819443, 252.0463867, -326.3325195, 310.1051025
4: -118.8996887, 186.9278564, -131.0894470, 206.2101440, -325.1098328, 318.0173035

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889696, upper bound: 465.3889563
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889535, upper bound: 465.3895747
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -160.6238403, 467.3055420, -104.8477325, 304.1072693, -464.7310791, 570.7335205
1: -114.4639740, 292.0288696, -74.8315887, 190.6109619, -304.5963440, 366.8604431
2: -125.2534332, 269.5950012, -82.6269760, 176.0292969, -301.0680542, 352.2219238
3: -113.2132034, 349.5109253, -74.2861862, 228.3231964, -341.5364075, 423.3729858
4: -180.1744537, 286.5898438, -118.8996887, 186.9278564, -366.6432190, 405.4895325

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3883187, upper bound: 465.3860236
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889342, upper bound: 465.3897387
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -160.6094818, 467.2616882, -115.7158203, 336.8948059, -497.5042725, 581.6782837
1: -114.4537735, 292.0023499, -82.7079086, 210.3442993, -324.4216614, 374.7102661
2: -125.2429428, 269.5708313, -90.8642197, 194.3667755, -319.4096680, 360.4350586
3: -113.2036057, 349.4791260, -81.7819443, 252.0463867, -365.2500000, 430.8899231
4: -180.1582642, 286.5648499, -131.0894470, 206.2101440, -385.9266968, 417.6542969

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3883187, upper bound: 465.3860236
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3889342, upper bound: 465.3897387
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -113.7575150, 328.7003784, -168.4045258, 490.6112366, -602.9436646, 497.1049194
1: -80.9861374, 206.6053009, -120.0795898, 306.1496582, -387.1358032, 326.2263794
2: -89.0871124, 190.9403381, -131.2113800, 282.8360901, -371.9232178, 321.9782104
3: -80.2338257, 246.8856201, -118.6369400, 366.4380493, -446.2868042, 365.5224915
4: -128.3227997, 202.2826538, -188.8110046, 300.5189514, -428.8417053, 390.6622620

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3805462, upper bound: 465.3861820
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3809313, upper bound: 465.3876591
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -113.7575150, 328.7003784, -198.1445312, 579.1904297, -689.7386475, 526.6384277
1: -80.9861374, 206.6053009, -141.3370819, 360.6016235, -441.3380127, 346.8789978
2: -89.0871124, 190.9403381, -154.9710999, 333.3670044, -422.3694153, 345.2855530
3: -80.2338257, 246.8856201, -139.9387207, 431.9634094, -511.3124390, 386.6851196
4: -128.3227997, 202.2826538, -222.7794189, 354.1849365, -482.5076904, 423.9779968

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3805462, upper bound: 465.3861820
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3809313, upper bound: 465.3876591
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -143.7654877, 419.5052490, -162.3312683, 472.4241943, -614.2062988, 580.6910400
1: -102.7205200, 261.8682861, -115.6636658, 295.0989990, -397.4070740, 376.8473206
2: -113.1239395, 242.2788696, -126.5254135, 272.4363098, -385.2453918, 368.4535217
3: -101.7363129, 313.7779846, -114.3465652, 353.1777344, -454.2220764, 427.8801575
4: -162.8161011, 256.9476013, -182.0711060, 289.5286865, -451.8806152, 438.5613708

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3777321, upper bound: 465.3759427
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3777321, upper bound: 465.3838715
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -143.8913574, 419.8497009, -165.9076996, 482.5168152, -624.4443970, 584.6306763
1: -102.8087540, 262.0800781, -118.2640076, 301.0656738, -403.5286865, 379.6759033
2: -113.2184296, 242.4689178, -129.0776215, 277.8726196, -390.7820129, 371.2275696
3: -101.8234787, 314.0302124, -116.7365723, 360.5200806, -461.6850281, 430.5308838
4: -162.9506226, 257.1592102, -185.7076721, 295.4857483, -457.9715271, 442.4667053

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3797980, upper bound: 465.3760569
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3797980, upper bound: 465.3845662
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -116.2288818, 338.2392883, -109.5310822, 319.3970337, -435.6258850, 447.7703857
1: -82.9504013, 211.4934845, -78.3369827, 199.4492340, -282.3995972, 289.8303833
2: -91.0312881, 195.6333313, -86.4208984, 184.3394470, -275.3706970, 282.0541687
3: -81.9896698, 253.0622864, -77.6577911, 239.1646271, -321.1542969, 330.7200317
4: -131.4785309, 207.1595612, -124.5348282, 195.6466827, -327.1252136, 331.6943970

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3812858, upper bound: 465.3821776
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3813959, upper bound: 465.3870143
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -118.7160110, 344.5063782, -117.5057678, 342.8114929, -461.5274658, 462.0121460
1: -84.6201630, 215.6566620, -84.1257324, 213.7738647, -298.3940430, 299.7824097
2: -92.6952972, 199.3886108, -92.5211868, 197.3695221, -290.0647583, 291.9097900
3: -83.5835114, 257.8869019, -83.1970367, 256.4366455, -340.0201416, 341.0839233
4: -133.8462524, 211.1509705, -133.3737030, 209.7115173, -343.5577393, 344.5246582

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3849579, upper bound: 465.3864653
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3849787, upper bound: 465.3887343
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -125.6034546, 368.0972595, -107.4954987, 314.0312195, -439.6346436, 475.5927429
1: -89.6893311, 229.9038239, -76.9543686, 195.9820404, -285.6713257, 306.8581848
2: -98.5811386, 213.4853058, -84.9533463, 181.1819763, -279.7631226, 298.4385986
3: -88.7218704, 274.8890991, -76.3028107, 235.0559540, -323.7778015, 351.1918945
4: -142.8032074, 225.0913849, -122.4922256, 192.2758331, -335.0790100, 347.5836182

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3835424, upper bound: 465.3854011
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3835427, upper bound: 465.3854119
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -128.8109589, 376.6734619, -115.7013397, 338.1573181, -466.9682617, 492.3747864
1: -91.8750763, 235.4201050, -82.9129028, 210.7470551, -302.6221313, 318.3329773
2: -100.8290253, 218.5209045, -91.2589417, 194.6057129, -295.4347534, 309.7798462
3: -90.8324814, 281.3391724, -82.0178604, 252.8691254, -343.7015991, 363.3569946
4: -146.0384369, 230.4092407, -131.6242828, 206.7683105, -352.8067627, 362.0335083

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3853641, upper bound: 465.3854034
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3853641, upper bound: 465.3854437
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -122.3155441, 354.7505188, -171.3325500, 498.8831177, -620.1337280, 526.0830078
1: -87.1952438, 222.1851654, -122.1148911, 311.2919617, -398.4871521, 344.0940247
2: -95.4745560, 205.4307861, -133.1983643, 287.5982361, -383.0727844, 338.5966187
3: -86.0819397, 265.6045837, -120.4880371, 372.5160217, -458.4551697, 386.0926208
4: -137.8734741, 217.4793091, -191.8069458, 305.4091797, -443.2826538, 408.9715881

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3808851, upper bound: 465.3842584
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3808712, upper bound: 465.3836479
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -120.9867020, 351.3432617, -179.6169891, 525.1378784, -644.7979126, 530.9602051
1: -86.2972641, 219.9383240, -128.0572815, 327.4964905, -413.7937622, 347.7931519
2: -94.5418930, 203.4011993, -139.9877930, 303.2390442, -397.7809448, 343.3606262
3: -85.2100754, 262.9645996, -126.4893799, 391.6736145, -476.6748657, 389.4539795
4: -136.5908356, 215.3151550, -201.9300385, 321.1727600, -457.7635803, 416.8968506

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3771898, upper bound: 465.3838558
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3771760, upper bound: 465.3834748
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -155.5840607, 453.8144531, -167.0918274, 486.5509949, -640.4237061, 619.9519043
1: -111.1246490, 282.6829224, -119.0881653, 303.6496887, -414.6128235, 401.3824463
2: -121.8893051, 261.5825500, -130.0041656, 280.4479370, -402.2091064, 391.3650818
3: -109.7148361, 338.7882385, -117.5588379, 363.4248352, -472.6652222, 456.3470459
4: -175.5529327, 277.2999268, -187.1935883, 297.9009094, -473.1358337, 464.1502380

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3802645, upper bound: 465.3767026
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3803070, upper bound: 465.3767837
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -154.0382385, 449.4748840, -175.6552124, 513.3949585, -665.4552612, 624.2836304
1: -110.0521927, 279.9425354, -125.2165070, 320.2814331, -430.1863098, 404.7520142
2: -120.7516098, 259.0637512, -136.9269867, 296.4862061, -417.0057373, 395.7752686
3: -108.6616058, 335.5243530, -123.7163925, 383.0472717, -491.1576538, 459.2407532
4: -173.9039001, 274.6371765, -197.4875336, 314.0735779, -487.6747437, 471.7467346

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3765881, upper bound: 465.3765628
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3766432, upper bound: 465.3766432
time: 0.67 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.36 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3889696, upper bound: 465.3889562
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3889535, upper bound: 465.3895744
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3889696, upper bound: 465.3889563
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3889535, upper bound: 465.3895747
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3883187, upper bound: 465.3860236
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3889342, upper bound: 465.3897387
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3883187, upper bound: 465.3860236
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3889342, upper bound: 465.3897387
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3805462, upper bound: 465.3861820
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3809313, upper bound: 465.3876591
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3805462, upper bound: 465.3861820
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3809313, upper bound: 465.3876591
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3777321, upper bound: 465.3759427
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3777321, upper bound: 465.3838715
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3797980, upper bound: 465.3760569
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3797980, upper bound: 465.3845662
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3812858, upper bound: 465.3821776
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3813959, upper bound: 465.3870143
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3849579, upper bound: 465.3864653
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3849787, upper bound: 465.3887343
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3835424, upper bound: 465.3854011
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3835427, upper bound: 465.3854119
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3853641, upper bound: 465.3854034
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3853641, upper bound: 465.3854437
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3808851, upper bound: 465.3842584
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3808712, upper bound: 465.3836479
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3771898, upper bound: 465.3838558
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3771760, upper bound: 465.3834748
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3802645, upper bound: 465.3767026
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3803070, upper bound: 465.3767837
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3765881, upper bound: 465.3765628
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 0, lower bound: -465.3766432, upper bound: 465.3766432

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -100.0361404, 290.3537903, -103.1272049, 299.1770630, -399.2131958, 393.4809875
1: -71.4795227, 181.9232788, -73.6329346, 187.4949188, -258.9744263, 255.5562134
2: -78.9742889, 167.8774872, -81.3232803, 173.0993805, -252.0736694, 249.2007751
3: -70.9767380, 218.0625610, -73.1035080, 224.6492920, -295.6260376, 291.1660767
4: -113.6319504, 178.4671936, -117.0159073, 183.8951569, -297.5270996, 295.4830933

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3884482, upper bound: 465.3886221
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3896068, upper bound: 465.3889801
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -104.0908051, 302.0028381, -102.9691162, 298.2714844, -402.3622742, 404.9719543
1: -74.3352127, 189.1371460, -73.4818420, 187.1039734, -261.4391785, 262.6189880
2: -82.2218933, 174.6660919, -81.1459427, 172.7424622, -254.9643402, 255.8120422
3: -73.8360977, 226.7537537, -72.9570312, 224.0908661, -297.9269104, 299.7107849
4: -118.1998062, 185.5865326, -116.7125702, 183.4762115, -301.6759949, 302.2991028

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3884314, upper bound: 465.3893350
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3884314, upper bound: 465.3895994
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -100.0361404, 290.3537903, -114.0106277, 331.9869995, -432.0231018, 404.3643799
1: -71.4795227, 181.9232788, -81.5171280, 207.2425995, -278.7220764, 263.4403992
2: -78.9742889, 167.8774872, -89.5723495, 191.4510040, -270.4252930, 257.4498291
3: -70.9767380, 218.0625610, -80.6097717, 248.3947144, -319.3714600, 298.6723328
4: -113.6319504, 178.4671936, -129.2161255, 203.1953888, -316.8273315, 307.6833191

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3869266, upper bound: 465.3876405
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3882941, upper bound: 465.3884474
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -104.0908051, 302.0028381, -113.6506424, 330.5096741, -434.6004333, 415.6534729
1: -74.3352127, 189.1371460, -81.2247772, 206.4840546, -280.8192444, 270.3619385
2: -82.2218933, 174.6660919, -89.2414627, 190.7369537, -272.9588623, 263.9075012
3: -73.8360977, 226.7537537, -80.3250580, 247.4081879, -321.2442627, 307.0787964
4: -118.1998062, 185.5865326, -128.6965637, 202.4125061, -320.6122437, 314.2830200

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3869240, upper bound: 465.3887302
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3869240, upper bound: 465.3891984
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -155.6344299, 453.0532837, -103.1272049, 299.1770630, -454.8114929, 554.7042847
1: -110.9758606, 283.0027466, -73.6329346, 187.4949188, -297.9930115, 356.6356812
2: -121.4006348, 261.1921387, -81.3232803, 173.0993805, -294.2680054, 342.5153809
3: -109.7723694, 338.8316040, -73.1035080, 224.6492920, -334.4216309, 411.4977417
4: -174.6009674, 277.8516541, -117.0159073, 183.8951569, -358.0012207, 394.8675537

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3812242, upper bound: 465.3792569
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3891668, upper bound: 465.3860448
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -156.2541962, 453.8079224, -102.9691162, 298.2714844, -454.5256958, 555.4738770
1: -111.3962784, 283.7268982, -73.4818420, 187.1039734, -298.0276794, 357.2087402
2: -121.9939423, 261.5901794, -81.1459427, 172.7424622, -294.5263367, 342.7361145
3: -110.1854782, 339.7637939, -72.9570312, 224.0908661, -334.2763367, 412.3088074
4: -175.2595062, 278.4399414, -116.7125702, 183.4762115, -358.3111877, 395.1525269

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3884127, upper bound: 465.3895701
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3895830, upper bound: 465.3897632
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -155.6199036, 453.0093079, -114.0106277, 331.9869995, -487.6069031, 565.6627197
1: -110.9656754, 282.9757996, -81.5171280, 207.2425995, -317.8328247, 364.4929199
2: -121.3900528, 261.1674805, -89.5723495, 191.4510040, -312.6231689, 350.7398376
3: -109.7627335, 338.7994385, -80.6097717, 248.3947144, -358.1574402, 419.0246277
4: -174.5846405, 277.8262939, -129.2161255, 203.1953888, -377.3020020, 407.0424194

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3863538, upper bound: 465.3859980
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3883187, upper bound: 465.3860236
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -156.3222961, 453.9724731, -113.6506424, 330.5096741, -486.8319702, 566.4394531
1: -111.4418945, 283.8342285, -81.2247772, 206.4840546, -317.5566101, 365.0588989
2: -122.0406723, 261.6873779, -89.2414627, 190.7369537, -312.5820618, 350.9287720
3: -110.2305222, 339.8923645, -80.3250580, 247.4081879, -357.6386719, 419.8564758
4: -175.3215332, 278.5470581, -128.6965637, 202.4125061, -377.3265686, 407.2435303

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3869046, upper bound: 465.3895088
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3882618, upper bound: 465.3896968
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -108.6367111, 314.0116272, -167.1928711, 487.1614990, -594.3501587, 481.2044983
1: -77.3477631, 197.4969330, -119.2199631, 303.9987793, -381.3465576, 316.2514038
2: -85.1665573, 182.5008850, -130.2921143, 280.8477783, -366.0143433, 312.6135864
3: -76.6756287, 235.9512482, -117.7979126, 363.8609314, -440.1426086, 353.7490845
4: -122.7255249, 193.2916870, -187.4959717, 298.4033508, -421.1288757, 380.3515625

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3828685, upper bound: 465.3720548
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3846993, upper bound: 465.3721019
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -108.7068634, 314.8726196, -167.7442322, 488.6790161, -596.0393066, 482.6168213
1: -77.5290070, 197.5921478, -119.6109009, 304.9420166, -382.4710083, 316.7927551
2: -85.1347656, 182.6287994, -130.7097626, 281.6992188, -366.8338928, 313.1745911
3: -76.6961517, 236.3076782, -118.1802826, 365.0031433, -441.3546448, 354.4879761
4: -122.7812195, 193.5673981, -188.0816650, 299.3370972, -422.1183167, 381.2189026

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3854588, upper bound: 465.3882840
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3866188, upper bound: 465.3882625
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -108.6367111, 314.0116272, -196.8901062, 575.5847168, -680.9767456, 510.6788330
1: -77.3477631, 197.4969330, -140.4484100, 358.3578186, -435.4427490, 336.8725281
2: -85.1665573, 182.5008850, -154.0192566, 331.2828674, -416.3602600, 335.8861084
3: -76.6756287, 235.9512482, -139.0694427, 429.2792358, -505.0577087, 374.8742371
4: -122.7255249, 193.2916870, -221.4092865, 351.9793396, -474.7048645, 413.6094055

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3730324, upper bound: 465.3711730
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3777079, upper bound: 465.3714914
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -108.7068634, 314.8726196, -197.3660431, 576.9288940, -682.5074463, 512.0218506
1: -77.5290070, 197.5921478, -140.7868042, 359.1946411, -436.4910583, 337.3625793
2: -85.1347656, 182.6287994, -154.3843231, 332.0475159, -417.1289368, 336.3955383
3: -76.6961517, 236.3076782, -139.4024811, 430.2855225, -506.1353760, 375.5824280
4: -122.7812195, 193.5673981, -221.9290771, 352.8028870, -475.5841064, 414.4122620

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3760268, upper bound: 465.3873209
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3807492, upper bound: 465.3874730
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -139.6014709, 407.5807495, -162.3312683, 472.4241943, -610.0128784, 568.7515259
1: -99.7634277, 254.4498901, -115.6636658, 295.0989990, -394.4417114, 369.4102783
2: -109.9431610, 235.4137573, -126.5254135, 272.4363098, -382.0515137, 361.5802002
3: -98.8385391, 304.8928833, -114.3465652, 353.1777344, -451.3125610, 418.9787903
4: -158.2583160, 249.6465759, -182.0711060, 289.5286865, -447.3117371, 431.2567444

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3703072, upper bound: 465.3627873
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3687573, upper bound: 465.3627607
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -139.5540009, 407.9426270, -162.3312683, 472.4241943, -610.0633545, 569.1078491
1: -99.8550186, 254.2766418, -115.6636658, 295.0989990, -394.5642700, 369.3041077
2: -109.7563858, 235.3020630, -126.5254135, 272.4363098, -381.9077454, 361.4825745
3: -98.7758560, 304.8869934, -114.3465652, 353.1777344, -451.2779846, 418.9998169
4: -158.0924072, 249.6439056, -182.0711060, 289.5286865, -447.1988525, 431.2545471

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3703072, upper bound: 465.3627873
time: 1.16 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3687573, upper bound: 465.3831133
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -139.6014709, 407.5807495, -165.9076996, 482.5168152, -620.1105347, 572.3421021
1: -99.7634277, 254.4498901, -118.2640076, 301.0656738, -400.4737854, 372.0250854
2: -109.9431610, 235.4137573, -129.0776215, 277.8726196, -387.4921570, 364.1633301
3: -98.8385391, 304.8928833, -116.7365723, 360.5200806, -458.6873779, 421.3751526
4: -158.2583160, 249.6465759, -185.7076721, 295.4857483, -453.2625122, 434.9472656

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3703072, upper bound: 465.3629575
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3687573, upper bound: 465.3632575
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -139.5540009, 407.9426270, -165.9076996, 482.5168152, -620.1721191, 572.7061157
1: -99.8550186, 254.2766418, -118.2640076, 301.0656738, -400.5963135, 371.9189148
2: -109.7563858, 235.3020630, -129.0776215, 277.8726196, -387.3483887, 364.0656738
3: -98.7758560, 304.8869934, -116.7365723, 360.5200806, -458.6528015, 421.3961487
4: -158.0924072, 249.6439056, -185.7076721, 295.4857483, -453.1536560, 434.9493408

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3703072, upper bound: 465.3842077
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3687573, upper bound: 465.3842315
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -110.0814896, 320.6661682, -107.8976822, 314.6862793, -424.7677612, 428.5638428
1: -78.6690979, 200.3631592, -77.1948242, 196.4745331, -275.1436157, 277.5579834
2: -86.4002380, 185.1860199, -85.1796875, 181.5439301, -267.9440613, 270.3656921
3: -77.7816772, 239.9759674, -76.5336227, 235.6584473, -313.4400635, 316.5095520
4: -124.7724152, 196.3521881, -122.7357483, 192.7556152, -317.5280151, 319.0878601

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3812858, upper bound: 465.3821776
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3812858, upper bound: 465.3821776
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -114.5573959, 333.5451355, -107.6483459, 313.5895081, -428.1469116, 441.1934814
1: -81.8061676, 208.3855286, -76.9864578, 195.9348907, -277.7409973, 285.3719788
2: -89.9281082, 192.7340393, -84.9548721, 181.0259552, -270.9540100, 277.6888428
3: -80.9151840, 249.5615692, -76.3363190, 234.9469452, -315.8621216, 325.8978271
4: -129.8063660, 204.2036438, -122.3711243, 192.1836700, -321.9900513, 326.5747681

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3813959, upper bound: 465.3870143
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3813959, upper bound: 465.3870142
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -112.5738754, 327.0722351, -115.8718109, 338.2049255, -450.7788086, 442.9440308
1: -80.3542175, 204.5864410, -82.9930038, 210.8720551, -291.2262573, 287.5794067
2: -88.0893326, 189.0109406, -91.2993088, 194.6647949, -282.7540894, 280.3101807
3: -79.3962021, 244.8850098, -82.0811310, 253.0032654, -332.3994751, 326.9660645
4: -127.1909943, 200.4104004, -131.6153107, 206.8747711, -334.0657349, 332.0256653

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3849560, upper bound: 465.3864653
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3828252, upper bound: 465.3863831
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -116.7092209, 338.8756714, -115.1083679, 335.6657410, -452.3749695, 453.9840088
1: -83.2440491, 211.9827118, -82.4218674, 209.3723755, -292.6163940, 294.4045715
2: -91.3362045, 195.9703674, -90.6590195, 193.2552185, -284.5914307, 286.6293945
3: -82.2788696, 253.6849823, -81.5204697, 251.1696472, -333.4485168, 335.2054443
4: -131.8187714, 207.6286011, -130.6588287, 205.4033508, -337.2220459, 338.2874146

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3849786, upper bound: 465.3887343
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3828384, upper bound: 465.3875011
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -115.1088486, 338.7149963, -107.4954987, 314.0312195, -429.1400757, 446.2104797
1: -82.3785629, 211.1726837, -76.9543686, 195.9820404, -278.3605652, 288.1270142
2: -90.8334427, 196.1108398, -84.9533463, 181.1819763, -272.0153809, 281.0641174
3: -81.6075897, 252.8109283, -76.3028107, 235.0559540, -316.6634827, 329.1137390
4: -131.5858765, 206.9515533, -122.4922256, 192.2758331, -323.8616943, 329.4437866

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3835423, upper bound: 465.3854011
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3835423, upper bound: 465.3854011
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -170.3359985, 499.8345947, -107.4954987, 314.0312195, -484.3671570, 606.0072021
1: -121.6268082, 311.2960510, -76.9543686, 195.9820404, -317.4249573, 388.2504272
2: -132.9615173, 288.3020935, -84.9533463, 181.1819763, -314.1434937, 373.2554321
3: -120.1368790, 372.4183655, -76.3028107, 235.0559540, -355.1928406, 448.5327148
4: -192.1242523, 305.2953491, -122.4922256, 192.2758331, -384.0415649, 427.7875671

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3835423, upper bound: 465.3854119
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3835423, upper bound: 465.3854119
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -117.8901749, 345.9811707, -115.7013397, 338.1573181, -456.0474548, 461.6824646
1: -84.2718430, 215.9084167, -82.9129028, 210.7470551, -295.0188599, 298.8212891
2: -92.7665787, 200.4199066, -91.2589417, 194.6057129, -287.3722839, 291.6788330
3: -83.4355011, 258.3346558, -82.0178604, 252.8691254, -336.3046265, 340.3525085
4: -134.3442535, 211.5190582, -131.6242828, 206.7683105, -341.1125488, 343.1433105

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3758681, upper bound: 465.3613914
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3606963, upper bound: 465.3606963
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -171.2318573, 502.3306580, -115.7013397, 338.1573181, -509.3891602, 616.6256104
1: -122.2152786, 312.9855042, -82.9129028, 210.7470551, -332.7581482, 395.8983459
2: -133.6574249, 289.9031982, -91.2589417, 194.6057129, -328.2631226, 381.1621399
3: -120.7250748, 374.3629761, -82.0178604, 252.8691254, -373.5942078, 456.1602173
4: -193.1705933, 306.8623352, -131.6242828, 206.7683105, -399.5623474, 438.4866028

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3758681, upper bound: 465.3641477
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3606963, upper bound: 465.3633701
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -116.0375595, 336.7356567, -170.1153107, 495.3865356, -610.3270874, 506.8509521
1: -82.7420731, 210.9910278, -121.2521744, 309.1150818, -391.8571472, 332.0388794
2: -90.6554947, 195.0819397, -132.2730255, 285.5810242, -376.2364807, 327.3249512
3: -81.7132416, 252.1979065, -119.6451874, 369.9133301, -451.4805603, 371.8430786
4: -130.9823456, 206.4844666, -190.4754333, 303.2756042, -434.2578735, 396.6470947

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3808851, upper bound: 465.3842584
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3808851, upper bound: 465.3842584
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -121.7872391, 351.4432068, -170.6469879, 496.8966064, -617.4939575, 522.0902100
1: -86.6562576, 220.2011108, -121.6254196, 310.0502014, -396.7064514, 341.6369629
2: -94.8467789, 203.2822876, -132.6780243, 286.4336243, -381.2803955, 335.9381104
3: -85.5451050, 263.3420410, -120.0112991, 371.0342712, -456.3804321, 383.3533325
4: -136.6193237, 215.5151367, -191.0586853, 304.1824951, -440.8017883, 406.2621765

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3808712, upper bound: 465.3836479
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3808712, upper bound: 465.3836479
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -115.0802460, 334.2513733, -178.4552002, 521.7940674, -635.5192871, 512.7065430
1: -82.0918198, 209.3503113, -127.2317200, 325.4159546, -407.5077209, 336.3813171
2: -89.9987717, 193.5926208, -139.1091461, 301.3016357, -391.3003540, 332.6764526
3: -81.0935211, 250.2762604, -125.6861191, 389.1882019, -470.0667725, 375.9623718
4: -130.0672607, 204.9152222, -200.6668854, 319.1246948, -449.1919250, 405.2344971

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3771898, upper bound: 465.3838558
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3771898, upper bound: 465.3838558
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -120.6688766, 348.3511353, -178.8880463, 523.0400391, -642.2529907, 527.2391357
1: -85.8945389, 218.2328796, -127.5393295, 326.1956787, -412.0902100, 345.5751343
2: -94.0763550, 201.4404449, -139.4289398, 302.0364685, -396.1128235, 340.8511047
3: -84.8224258, 261.0282593, -125.9821854, 390.1115417, -474.6595459, 387.0104370
4: -135.4841919, 213.5996246, -201.1262360, 319.8973694, -455.3815613, 414.3812561

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3771760, upper bound: 465.3834748
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3771760, upper bound: 465.3834748
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -149.4750061, 436.1450500, -165.8729706, 483.0365906, -630.7382202, 601.0443115
1: -106.7874451, 271.7247009, -118.2240295, 301.4657593, -408.0674438, 389.5392761
2: -117.2126923, 251.4536285, -129.0738678, 278.4223328, -395.4871826, 380.2987976
3: -105.4562683, 325.6643066, -116.7132339, 360.8129883, -465.7730713, 442.3775330
4: -168.7911377, 266.5663452, -185.8533325, 295.7600708, -464.2250366, 452.0704651

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3781093, upper bound: 465.3764986
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3781093, upper bound: 465.3767026
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -155.1564026, 450.3472595, -166.4208832, 484.5969543, -637.8965454, 615.8059692
1: -110.6557465, 280.7829285, -118.6084824, 302.4302673, -412.8757324, 398.9859619
2: -121.3893585, 259.3779297, -129.4925385, 279.3036194, -400.5079346, 388.6588440
3: -109.2498245, 336.6260376, -117.0911407, 361.9692688, -470.6665955, 453.7171326
4: -174.3116760, 275.3439026, -186.4567108, 296.6965332, -470.7417603, 461.4565125

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3803070, upper bound: 465.3767837
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3803070, upper bound: 465.3767837
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -148.3392792, 432.9606934, -174.5361176, 510.1843262, -656.4839478, 606.6355591
1: -106.0035324, 269.7103271, -124.4219971, 318.2805481, -424.1150513, 393.7050781
2: -116.3989258, 249.5952454, -136.0825195, 294.6246643, -410.7722168, 385.4565735
3: -104.6933060, 323.2795105, -122.9443054, 380.6580505, -484.7766113, 446.2238159
4: -167.6053009, 264.6125183, -196.2744904, 312.1053772, -479.4017334, 460.5022278

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3765073, upper bound: 465.3765073
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3765073, upper bound: 465.3765628
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -154.0126801, 447.0500183, -174.9734192, 511.4239807, -663.3045654, 621.1522827
1: -109.8676224, 278.7258606, -124.7296677, 319.0613098, -428.7268677, 403.0165710
2: -120.5597916, 257.4514160, -136.4009399, 295.3599243, -415.6255798, 393.6452332
3: -108.4752884, 334.1849060, -123.2395706, 381.5796814, -489.4221802, 457.4244690
4: -173.0802765, 273.3322754, -196.7314301, 312.8763733, -485.7031555, 469.6822510

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3701196, upper bound: 465.3753836
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3756882, upper bound: 465.3756882
time: 0.88 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.93 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3884482, upper bound: 465.3886221
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3896068, upper bound: 465.3889801
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3884314, upper bound: 465.3893350
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3884314, upper bound: 465.3895994
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3869266, upper bound: 465.3876405
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3882941, upper bound: 465.3884474
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3869240, upper bound: 465.3887302
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3869240, upper bound: 465.3891984
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3812242, upper bound: 465.3792569
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3891668, upper bound: 465.3860448
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3884127, upper bound: 465.3895701
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3895830, upper bound: 465.3897632
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3863538, upper bound: 465.3859980
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3883187, upper bound: 465.3860236
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3869046, upper bound: 465.3895088
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3882618, upper bound: 465.3896968
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3828685, upper bound: 465.3720548
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3846993, upper bound: 465.3721019
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3854588, upper bound: 465.3882840
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3866188, upper bound: 465.3882625
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3730324, upper bound: 465.3711730
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3777079, upper bound: 465.3714914
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3760268, upper bound: 465.3873209
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3807492, upper bound: 465.3874730
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3703072, upper bound: 465.3627873
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3687573, upper bound: 465.3627607
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3703072, upper bound: 465.3627873
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3687573, upper bound: 465.3831133
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3703072, upper bound: 465.3629575
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3687573, upper bound: 465.3632575
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3703072, upper bound: 465.3842077
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3687573, upper bound: 465.3842315
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3812858, upper bound: 465.3821776
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3812858, upper bound: 465.3821776
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3813959, upper bound: 465.3870143
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3813959, upper bound: 465.3870142
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3849560, upper bound: 465.3864653
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3828252, upper bound: 465.3863831
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3849786, upper bound: 465.3887343
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3828384, upper bound: 465.3875011
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3835423, upper bound: 465.3854011
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3835423, upper bound: 465.3854011
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3835423, upper bound: 465.3854119
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3835423, upper bound: 465.3854119
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3758681, upper bound: 465.3613914
NS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3606963, upper bound: 465.3606963
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3758681, upper bound: 465.3641477
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3606963, upper bound: 465.3633701
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3808851, upper bound: 465.3842584
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3808851, upper bound: 465.3842584
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3808712, upper bound: 465.3836479
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3808712, upper bound: 465.3836479
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3771898, upper bound: 465.3838558
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3771898, upper bound: 465.3838558
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3771760, upper bound: 465.3834748
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3771760, upper bound: 465.3834748
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3781093, upper bound: 465.3764986
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3781093, upper bound: 465.3767026
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3803070, upper bound: 465.3767837
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3803070, upper bound: 465.3767837
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3765073, upper bound: 465.3765073
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3765073, upper bound: 465.3765628
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3701196, upper bound: 465.3753836
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.93
Output dim: 0, lower bound: -465.3756882, upper bound: 465.3756882

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -98.9626617, 287.2854309, -97.8213425, 284.0072632, -382.9699097, 385.1067810
1: -70.7175674, 180.0175323, -69.8671799, 178.0800934, -248.7976532, 249.8847046
2: -78.1570892, 166.1090088, -77.2790146, 164.3645782, -242.5216522, 243.3880157
3: -70.2329407, 215.7759094, -69.4243622, 213.3529053, -283.5858459, 285.2002563
4: -112.4645691, 176.5840149, -111.2429886, 174.5916595, -287.0561829, 287.8269958

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3883876, upper bound: 465.3871804
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3883876, upper bound: 465.3886221
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -99.1204376, 287.7214050, -98.6615982, 287.0763245, -386.1967773, 386.3829956
1: -70.8295212, 180.2803955, -70.5921326, 179.5635986, -250.3931122, 250.8725281
2: -78.2836838, 166.3481903, -77.8458328, 165.7890625, -244.0727234, 244.1940308
3: -70.3450317, 216.1015625, -69.9771805, 215.3574066, -285.7024231, 286.0786743
4: -112.6371918, 176.8542786, -112.1569748, 176.2185669, -288.8557739, 289.0112610

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3893151, upper bound: 465.3872091
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3893151, upper bound: 465.3889801
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -103.0321884, 298.9808350, -97.7009430, 283.2174988, -386.2496948, 396.6817627
1: -73.5850754, 187.2626801, -69.7474213, 177.7686615, -251.3537292, 257.0101013
2: -81.4183350, 172.9279480, -77.1369705, 164.0828247, -245.5011597, 250.0649109
3: -73.1037369, 224.5025024, -69.3099899, 212.8850708, -285.9888000, 293.8125000
4: -117.0516586, 183.7342224, -110.9886475, 174.2541962, -291.3058472, 294.7228394

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3883903, upper bound: 465.3883903
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3883903, upper bound: 465.3893350
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -103.1364059, 299.2523499, -98.3678131, 285.7961121, -388.9324951, 397.6201172
1: -73.6577759, 187.4216919, -70.3460083, 178.9328156, -252.5905914, 257.7677002
2: -81.5002975, 173.0689697, -77.5639114, 165.2085419, -246.7088318, 250.6328583
3: -73.1755295, 224.7066040, -69.7376556, 214.5131378, -287.6886292, 294.4442444
4: -117.1587448, 183.9033203, -111.7085037, 175.5670166, -292.7257080, 295.6118164

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3893351, upper bound: 465.3884313
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3893351, upper bound: 465.3895994
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -96.2668381, 279.8480225, -106.7331696, 311.7525635, -408.0194092, 386.5811768
1: -68.8492203, 175.2839050, -76.4614487, 194.4356079, -263.2847595, 251.7453308
2: -76.2257690, 161.7787323, -84.2686615, 179.6858521, -255.9116058, 246.0473633
3: -68.4307632, 210.1986389, -75.7037048, 233.2474213, -301.6781311, 285.9023438
4: -109.6437302, 172.0195160, -121.5265121, 190.7765961, -300.4202881, 293.5460205

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3868328, upper bound: 465.3843629
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3869153, upper bound: 465.3876236
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -98.3455505, 285.1060791, -114.8201141, 335.6200867, -433.9656067, 399.9261169
1: -70.2573776, 178.7717590, -82.3316879, 209.0596771, -279.3170166, 261.1034241
2: -77.6180954, 164.9145203, -90.4973984, 192.9994202, -270.6175232, 255.4119263
3: -69.7653656, 214.2520294, -81.3425980, 250.8743591, -320.6396790, 295.5946045
4: -111.6383133, 175.3678131, -130.5654755, 205.1132355, -316.7515259, 305.9332886

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3881635, upper bound: 465.3843665
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3882881, upper bound: 465.3884474
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -100.3113861, 291.4890137, -106.2498398, 309.9829712, -410.2942810, 397.7388306
1: -71.7086716, 182.4913483, -76.0896912, 193.4771729, -265.1858215, 258.5810547
2: -79.4862747, 168.5380859, -83.8652878, 178.7720642, -258.2583313, 252.4033813
3: -71.2920151, 218.8952332, -75.3460846, 232.0372314, -303.3292542, 294.2413330
4: -114.2203369, 179.1295013, -120.9037857, 189.7959290, -304.0162048, 300.0332947

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3868684, upper bound: 465.3874540
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3868684, upper bound: 465.3887302
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -102.2480392, 296.3070984, -113.8719788, 332.5976562, -434.8457031, 410.1790161
1: -73.0040207, 185.7211304, -81.6339951, 207.2572021, -280.2612305, 267.3551331
2: -80.7430801, 171.4665527, -89.7233353, 191.3121490, -272.0552368, 261.1898499
3: -72.5148163, 222.6089630, -80.6574554, 248.6756134, -321.1904297, 303.2664185
4: -116.0368805, 182.2219086, -129.4237671, 203.3407745, -319.3775940, 311.6456909

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3881649, upper bound: 465.3874573
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3882828, upper bound: 465.3891984
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -154.5136719, 449.8396912, -97.8213425, 284.0072632, -438.5209351, 546.1621704
1: -110.1804504, 281.0078735, -69.8671799, 178.0800934, -287.7781982, 350.8750610
2: -120.5519028, 259.3380432, -77.2790146, 164.3645782, -284.6770630, 336.6170654
3: -108.9969330, 336.4420776, -69.4243622, 213.3529053, -322.3498535, 405.4215393
4: -173.3846283, 275.8852539, -111.2429886, 174.5916595, -347.4760437, 387.1282349

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -154.7544861, 450.5191040, -98.6615982, 287.0763245, -441.8307800, 547.7759399
1: -110.3549957, 281.4153137, -70.5921326, 179.5635986, -289.4850769, 352.0074463
2: -120.7379303, 259.7094727, -77.8458328, 165.7890625, -286.2993469, 337.5552979
3: -109.1671524, 336.9425049, -69.9771805, 215.3574066, -324.5245667, 406.5282898
4: -173.6398010, 276.3007202, -112.1569748, 176.2185669, -349.3628845, 388.4577026

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -155.0852203, 450.4833679, -97.7009430, 283.2174988, -438.3027344, 546.8596802
1: -110.5687485, 281.6607361, -69.7474213, 177.7686615, -287.8605042, 351.4081421
2: -121.1120148, 259.6744080, -77.1369705, 164.0828247, -284.9781189, 336.8113708
3: -109.3779755, 337.2868652, -69.3099899, 212.8850708, -322.2630310, 406.1774902
4: -174.0006409, 276.4014893, -110.9886475, 174.2541962, -347.8255920, 387.3901062

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3883825, upper bound: 465.3887775
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3883826, upper bound: 465.3895701
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -155.2610626, 450.9585266, -98.3678131, 285.7961121, -441.0571899, 548.0952148
1: -110.6950378, 281.9440002, -70.3460083, 178.9328156, -289.1993408, 352.2900085
2: -121.2478867, 259.9263916, -77.5639114, 165.2085419, -286.2519226, 337.4902954
3: -109.5023727, 337.6398010, -69.7376556, 214.5131378, -324.0154724, 407.0113831
4: -174.1818085, 276.6947021, -111.7085037, 175.5670166, -349.3240356, 388.4031677

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3894240, upper bound: 465.3876693
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3895817, upper bound: 465.3897582
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -154.5108337, 449.8269348, -107.9418335, 314.5551453, -469.0659790, 556.3634644
1: -110.1783218, 281.0006714, -77.2077789, 196.4079590, -306.2120667, 358.2084351
2: -120.5496216, 259.3318176, -84.9154587, 181.4307709, -301.7595520, 344.2472839
3: -108.9951630, 336.4335022, -76.3863907, 235.4177704, -344.4129333, 412.4268494
4: -173.3799591, 275.8795776, -122.5460968, 192.5542603, -365.4541321, 398.4256287

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -154.8123779, 450.6875000, -113.0442047, 327.4678040, -482.2801819, 562.2503052
1: -110.3957062, 281.5194092, -80.6862946, 204.4673004, -314.4965820, 362.2056885
2: -120.7792358, 259.8073425, -88.6489487, 188.5514832, -309.1165771, 348.4562988
3: -109.2059479, 337.0655212, -79.7939377, 245.2247314, -354.4306030, 416.4127197
4: -173.7007294, 276.4021912, -127.5067978, 200.4747772, -373.6962585, 403.9089966

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -153.2008514, 445.2471008, -106.2498398, 309.9829712, -463.1837463, 550.2942505
1: -109.2767944, 278.3091431, -76.0896912, 193.4771729, -302.4021301, 354.3988037
2: -119.7397690, 256.5823059, -83.8652878, 178.7720642, -298.3243103, 340.4476013
3: -108.1250153, 333.3455505, -75.3460846, 232.0372314, -340.1622314, 408.3226318
4: -171.9713898, 273.1914673, -120.9037857, 189.7959290, -361.3391418, 394.0952454

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3868574, upper bound: 465.3881882
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3868921, upper bound: 465.3895088
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -153.4907837, 445.7294617, -113.8719788, 332.5976562, -485.8434753, 558.3378906
1: -109.4131546, 278.8185425, -81.6339951, 207.2572021, -316.3147583, 360.4311523
2: -119.9176788, 257.0509033, -89.7233353, 191.3121490, -311.0220032, 346.7742310
3: -108.2643204, 333.8749084, -80.6574554, 248.6756134, -356.9399414, 414.1355896
4: -172.2520447, 273.6018677, -129.4237671, 203.3407745, -375.1485291, 403.0256042

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3871763, upper bound: 465.3867280
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3873171, upper bound: 465.3895403
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -100.2106247, 289.1964111, -142.3795013, 413.6552124, -512.3218994, 431.5758362
1: -71.3948822, 181.9924011, -101.5920868, 258.1519470, -329.5468140, 283.1034546
2: -78.5806503, 167.9049683, -110.8600845, 237.5555267, -316.1361694, 278.5291138
3: -70.7460480, 217.5656433, -100.2973557, 309.4163818, -379.7031555, 317.8630066
4: -113.2042236, 178.1275940, -159.3281708, 253.4917908, -366.6960144, 336.9476013

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3825899, upper bound: 465.3720208
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3825899, upper bound: 465.3720208
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -105.0811996, 303.2578430, -174.5791626, 508.2886353, -611.9564819, 477.8370056
1: -74.7929230, 190.8291626, -124.5796814, 317.0033875, -391.7962646, 314.9632568
2: -82.3746643, 176.1674500, -135.7697449, 292.6322021, -375.0068665, 311.7417908
3: -74.1761322, 228.0349426, -122.8871536, 379.5870667, -453.3692627, 350.9220886
4: -118.6597519, 186.7631836, -195.4279480, 311.0459595, -429.7056885, 381.7695007

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3820815, upper bound: 465.3719470
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3819968, upper bound: 465.3699278
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -100.4633942, 290.5993347, -142.6190338, 414.3415527, -513.3734131, 433.2183838
1: -71.7106628, 182.4192810, -101.7660522, 258.5608521, -330.2715149, 283.7548218
2: -78.6814194, 168.3338013, -111.0438538, 237.9208221, -316.6022339, 279.1588440
3: -70.9081726, 218.3163147, -100.4673767, 309.9195557, -380.4032593, 318.7836609
4: -113.4555969, 178.7257538, -159.5897675, 253.8997192, -367.3552551, 337.8120422

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3836629, upper bound: 465.3867078
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3837751, upper bound: 465.3879430
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -105.4510269, 305.2128296, -175.0046692, 509.5276794, -613.6512451, 480.2174683
1: -75.2040176, 191.6070709, -124.8803177, 317.7760315, -392.9800110, 316.0871277
2: -82.5961685, 177.0068054, -136.1127930, 293.3599854, -375.9561157, 312.9435425
3: -74.4090958, 229.1646576, -123.1897430, 380.5119019, -454.5807190, 352.3543701
4: -119.1052322, 187.7067871, -195.9099426, 311.8114929, -430.9166870, 383.1973267

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3853461, upper bound: 465.3861168
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3854721, upper bound: 465.3879584
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -100.2106247, 289.1964111, -172.9140472, 503.9114685, -600.6564331, 461.6413574
1: -71.3948822, 181.9924011, -123.3825836, 313.8063660, -384.7719421, 304.2640076
2: -78.5806503, 167.9049683, -135.2442932, 289.1201782, -367.6055603, 302.3823242
3: -70.7460480, 217.5656433, -122.1359558, 376.4189148, -446.1505127, 339.4486389
4: -113.2042236, 178.1275940, -193.9514160, 308.3344421, -421.5386658, 370.8745728

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3730324, upper bound: 465.3711730
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3730324, upper bound: 465.3711730
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -105.0811996, 303.2578430, -203.9862061, 595.6998901, -697.6156006, 506.9203186
1: -74.7929230, 190.8291626, -145.5752106, 370.7030029, -445.2550354, 335.3729858
2: -82.3746643, 176.1674500, -159.2775269, 342.4388123, -424.7426453, 334.7871399
3: -74.1761322, 228.0349426, -143.9412079, 444.2957764, -517.5798950, 371.8242493
4: -118.6597519, 186.7631836, -229.0064240, 364.0603027, -482.7200317, 414.6898193

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3777079, upper bound: 465.3714863
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3776093, upper bound: 465.3694639
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -100.4633942, 290.5993347, -173.2361298, 504.8168640, -601.9291382, 463.3669739
1: -71.7106628, 182.4192810, -123.6094284, 314.3665161, -385.6755371, 304.9689636
2: -78.6814194, 168.3338013, -135.4866943, 289.6333313, -368.2555847, 303.0752563
3: -70.9081726, 218.3163147, -122.3571777, 377.0892334, -447.0209351, 340.4404602
4: -113.4555969, 178.7257538, -194.3000946, 308.8808899, -422.3364563, 371.8295288

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3760268, upper bound: 465.3864604
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3760266, upper bound: 465.3865905
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -105.4510269, 305.2128296, -204.5400543, 597.3362427, -699.7118530, 509.4411316
1: -75.2040176, 191.6070709, -145.9645386, 371.7246704, -446.7173157, 336.5886230
2: -82.5961685, 177.0068054, -159.7177887, 343.4033813, -425.9597473, 336.0884094
3: -74.4090958, 229.1646576, -144.3319702, 445.5057678, -519.0781860, 373.3620300
4: -119.1052322, 187.7067871, -229.6352997, 365.0619812, -484.1671753, 416.2661743

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3807492, upper bound: 465.3866598
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3807492, upper bound: 465.3868238
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -131.4670563, 383.4336853, -137.1757812, 398.0640564, -527.3922729, 519.2899780
1: -93.9892426, 239.4385986, -97.8142242, 248.6630707, -342.1100769, 336.5133362
2: -103.5490341, 221.2840881, -106.8384094, 228.6098633, -331.8072510, 327.6965942
3: -93.0851364, 287.0437012, -96.6105118, 298.0443726, -390.3460083, 383.3156738
4: -148.9834137, 234.9425659, -153.5528412, 244.0547333, -392.5230713, 387.9540405

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3686952, upper bound: 465.3627607
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3686952, upper bound: 465.3627607
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -136.2674103, 397.7971191, -170.1419220, 494.8369751, -629.1177979, 566.7142944
1: -97.3845291, 248.3691254, -121.3350983, 308.8383179, -405.8210144, 369.0163574
2: -107.3587646, 229.6970978, -132.3434448, 284.8477783, -391.8926392, 361.6696472
3: -96.5106812, 297.6369934, -119.7493820, 369.8316345, -465.6364746, 417.0993347
4: -154.5352936, 243.6772461, -190.5021362, 302.9221497, -456.9841919, 433.7292480

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3686952, upper bound: 465.3627607
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3686952, upper bound: 465.3627607
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -131.8692169, 384.9013977, -137.1757812, 398.0640564, -527.8535767, 520.7484131
1: -94.3516312, 239.9428558, -97.8142242, 248.6630707, -342.5029602, 337.0871277
2: -103.6913910, 221.7835236, -106.8384094, 228.6098633, -331.9944153, 328.2106323
3: -93.3327255, 287.8720703, -96.6105118, 298.0443726, -390.5899353, 384.1728516
4: -149.2588348, 235.5991211, -153.5528412, 244.0547333, -392.8520203, 388.6098328

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3705689, upper bound: 465.3751354
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3705689, upper bound: 465.3831133
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -136.3430786, 398.5223694, -170.1419220, 494.8369751, -629.2898560, 567.4416504
1: -97.5566711, 248.4735870, -121.3350983, 308.8383179, -406.0241089, 369.1884766
2: -107.2458267, 229.8995972, -132.3434448, 284.8477783, -391.8210449, 361.8867493
3: -96.4979172, 297.9104919, -119.7493820, 369.8316345, -465.6713257, 417.4011230
4: -154.4759521, 243.9422455, -190.5021362, 302.9221497, -456.9788513, 433.9935303

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3707131, upper bound: 465.3824462
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3707244, upper bound: 465.3830715
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -131.4670563, 383.4336853, -142.7585297, 413.2647705, -542.6284790, 524.8507690
1: -93.9892426, 239.4385986, -101.7312775, 258.2026978, -351.7032166, 340.4227295
2: -103.5490341, 221.2840881, -111.0128326, 237.3333893, -340.5343018, 331.8759766
3: -93.0851364, 287.0437012, -100.4638214, 309.4389343, -401.7804871, 387.1387329
4: -148.9834137, 234.9425659, -159.3436890, 253.3702545, -401.8389587, 393.8081055

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3725620, upper bound: 465.3629575
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3725620, upper bound: 465.3629575
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -136.2674103, 397.7971191, -172.3658142, 500.9129333, -635.1948853, 568.9349976
1: -97.3845291, 248.3691254, -122.9552307, 312.3786011, -409.4291382, 370.6548462
2: -107.3587646, 229.6970978, -133.7885590, 288.0889587, -395.1394043, 363.1492615
3: -96.5106812, 297.6369934, -121.1591873, 374.2218323, -470.0581970, 418.5347900
4: -154.5352936, 243.6772461, -192.5800018, 306.4768982, -460.5392456, 435.8600159

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3771294, upper bound: 465.3632575
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3771294, upper bound: 465.3632575
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -131.8692169, 384.9013977, -142.7585297, 413.2647705, -543.0997925, 526.3173828
1: -94.3516312, 239.9428558, -101.7312775, 258.2026978, -352.0960999, 340.9964905
2: -103.6913910, 221.7835236, -111.0128326, 237.3333893, -340.7214355, 332.3899841
3: -93.3327255, 287.8720703, -100.4638214, 309.4389343, -402.0244141, 387.9959106
4: -149.2588348, 235.5991211, -159.3436890, 253.3702545, -402.1716003, 394.4681702

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3753042, upper bound: 465.3760716
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3753042, upper bound: 465.3842057
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -136.3430786, 398.5223694, -172.3658142, 500.9129333, -635.3723145, 569.6714478
1: -97.5566711, 248.4735870, -122.9552307, 312.3786011, -409.6322327, 370.8269348
2: -107.2458267, 229.8995972, -133.7885590, 288.0889587, -395.0677795, 363.3663635
3: -96.4979172, 297.9104919, -121.1591873, 374.2218323, -470.0930481, 418.8365784
4: -154.4759521, 243.9422455, -192.5800018, 306.4768982, -460.5373535, 436.1287537

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3804516, upper bound: 465.3839751
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3804600, upper bound: 465.3841919
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -110.0814896, 320.6661682, -105.1765137, 306.9780579, -417.0595398, 425.8426819
1: -78.6690979, 200.3631592, -75.2394638, 191.6631317, -270.3322144, 275.6026306
2: -86.4002380, 185.1860199, -83.0649261, 177.1489410, -263.5490723, 268.2509460
3: -77.7816772, 239.9759674, -74.6267471, 229.8652802, -307.6469421, 314.6026611
4: -124.7724152, 196.3521881, -119.7333527, 188.0309601, -312.8033752, 316.0854492

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3812858, upper bound: 465.3821776
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3812858, upper bound: 465.3821776
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -110.0814896, 320.6661682, -113.8092041, 333.8419495, -443.9234314, 434.4753723
1: -78.6690979, 200.3631592, -81.3977814, 208.3932648, -287.0623474, 281.7609253
2: -86.4002380, 185.1860199, -89.9816895, 193.3934479, -279.7935486, 275.1676941
3: -77.7816772, 239.9759674, -80.7692337, 249.5844269, -327.3660889, 320.7452087
4: -124.7724152, 196.3521881, -129.9748993, 204.3415222, -329.1139526, 326.3269958

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3812860, upper bound: 465.3821777
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3812858, upper bound: 465.3821776
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -114.5573959, 333.5451355, -105.0165024, 306.1036682, -420.6610718, 438.5616150
1: -81.8061676, 208.3855286, -75.0942307, 191.2737427, -273.0798340, 283.4797668
2: -89.9281082, 192.7340393, -82.9085388, 176.7626190, -266.6907349, 275.6425781
3: -80.9151840, 249.5615692, -74.4904175, 229.3310242, -310.2462158, 324.0519714
4: -129.8063660, 204.2036438, -119.4621811, 187.6046448, -317.4110107, 323.6658325

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3813959, upper bound: 465.3862823
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3813959, upper bound: 465.3870142
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -114.5573959, 333.5451355, -113.6345596, 333.0340576, -447.5914612, 447.1796875
1: -81.8061676, 208.3855286, -81.2479477, 208.0487366, -289.8548889, 289.6334839
2: -89.9281082, 192.7340393, -89.8134308, 193.1169128, -283.0449524, 282.5474854
3: -80.9151840, 249.5615692, -80.6248093, 249.1081696, -330.0233459, 330.1863098
4: -129.8063660, 204.2036438, -129.6971436, 203.9720154, -333.7783508, 333.9006958

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3813959, upper bound: 465.3862822
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3813959, upper bound: 465.3870143
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -111.4026947, 323.7058411, -109.5273743, 320.1062622, -431.5089111, 433.2332153
1: -79.5247650, 202.4992676, -78.4963226, 199.5879669, -279.1126709, 280.9956055
2: -87.1978073, 187.0755615, -86.4496536, 184.2199097, -271.4177246, 273.5251465
3: -78.5850449, 242.3878326, -77.6797943, 239.4947205, -318.0797119, 320.0676270
4: -125.9117889, 198.3573914, -124.6858749, 195.7778473, -321.6895752, 323.0432739

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3849560, upper bound: 465.3864653
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3849560, upper bound: 465.3864653
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -111.9127197, 325.1272278, -114.2370071, 332.3099060, -444.2225952, 439.3642273
1: -79.8807144, 203.3769684, -81.7361221, 207.0390930, -286.9197998, 285.1130676
2: -87.5879364, 187.8740540, -89.8640213, 190.9127197, -278.5006714, 277.7380676
3: -78.9350052, 243.4403076, -80.8051376, 248.6285553, -327.5635681, 324.2454529
4: -126.4639511, 199.2170868, -129.3035583, 203.1979065, -329.6618652, 328.5206299

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3828252, upper bound: 465.3863831
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3827919, upper bound: 465.3863831
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -115.4952469, 335.3753357, -108.7869797, 317.6510010, -433.1462097, 444.1623230
1: -82.3838348, 209.8147430, -77.9438477, 198.1430206, -280.5268555, 287.7586060
2: -90.4095764, 193.9608917, -85.8363724, 182.8651276, -273.2747192, 279.7972107
3: -81.4361801, 251.0886536, -77.1379547, 237.7242432, -319.1604309, 328.2266235
4: -130.4877167, 205.4974976, -123.7680435, 194.3587189, -324.8464355, 329.2655334

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3849786, upper bound: 465.3881873
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3849786, upper bound: 465.3887343
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -116.0693893, 337.0025330, -113.6420822, 330.3133240, -446.3826904, 450.6446228
1: -82.7867737, 210.8175049, -81.2846832, 205.8714142, -288.6581421, 292.1021729
2: -90.8530884, 194.8759155, -89.3554993, 189.8197327, -280.6728210, 284.2313843
3: -81.8342209, 252.2941589, -80.3610306, 247.1870880, -329.0212708, 332.6551819
4: -131.1201324, 206.4785919, -128.5554962, 202.0467072, -333.1667480, 335.0340881

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3828384, upper bound: 465.3872694
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3828384, upper bound: 465.3875011
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -115.1088486, 338.7149963, -106.8015060, 311.6741333, -426.7829895, 445.5164795
1: -82.3785629, 211.1726837, -76.3749695, 194.6269226, -277.0054321, 287.5476685
2: -90.8334427, 196.1108398, -84.2978134, 179.9353180, -270.7687378, 280.4085999
3: -81.6075897, 252.8109283, -75.7442474, 233.3562622, -314.9638367, 328.5551758
4: -131.5858765, 206.9515533, -121.5232086, 190.9100494, -322.4959106, 328.4747009

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3836105, upper bound: 465.3851518
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3835940, upper bound: 465.3848990
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -115.1088486, 338.7149963, -115.4606857, 338.6555481, -453.7644043, 454.1756287
1: -82.3785629, 211.1726837, -82.5575485, 211.4223328, -293.8009033, 293.7302246
2: -90.8334427, 196.1108398, -91.2431717, 196.2529144, -287.0863342, 287.3539429
3: -81.6075897, 252.8109283, -81.9095535, 253.1731567, -334.7807617, 334.7204285
4: -131.5858765, 206.9515533, -131.8067932, 207.2912903, -338.8771362, 338.7583618

Time for backsubstitution: 1.86 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.88 + 417.83 = 421.71 seconds
