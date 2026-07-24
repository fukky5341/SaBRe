## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 1406.026249396902


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-225.0351715, 567.3841553, -225.0351715, 567.3841553, -792.4193115, 792.4193115)
1: (-555.2101440, 854.5823364, -555.2101440, 854.5823364, -1409.7924805, 1409.7924805)
2: (-364.9949951, 830.7340698, -364.9949951, 830.7340698, -1195.7290039, 1195.7290039)
3: (-596.9047241, 986.7219238, -596.9047241, 986.7219238, -1583.6267090, 1583.6267090)
4: (-532.2448120, 946.0938721, -532.2448120, 946.0938721, -1478.3386230, 1478.3386230)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.88 + 2.37 = 3.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -1406.0403098, upper bound: 1406.0403098

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0400633, upper bound: 1406.0394374
time: 1.03 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0393564, upper bound: 1406.0393564
time: 1.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.19 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.19
Output dim: 3, lower bound: -1406.0400633, upper bound: 1406.0394374
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.19
Output dim: 3, lower bound: -1406.0393564, upper bound: 1406.0393564

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -210.3482971, 532.3909302, -221.3350677, 558.2429810, -768.5911255, 753.7258911
1: -518.5936890, 802.3444214, -545.9274902, 841.0584717, -1359.6519775, 1348.2717285
2: -341.0028992, 779.7360840, -358.9181519, 817.4517212, -1158.4545898, 1138.6540527
3: -557.2022095, 926.2874756, -587.0917358, 971.0841064, -1528.2862549, 1513.3791504
4: -498.0922852, 887.8120117, -523.5852051, 931.0083618, -1429.1005859, 1411.3972168

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0382724, upper bound: 1406.0361783
time: 1.14 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0389943, upper bound: 1406.0385224
time: 0.93 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -224.3145142, 565.5430908, -224.5402679, 566.1205444, -790.4349365, 790.0833740
1: -553.4265137, 851.8437500, -553.9850464, 852.7025757, -1406.1291504, 1405.8288574
2: -363.8313599, 828.0565186, -364.1953430, 828.8972168, -1192.7285156, 1192.2518311
3: -595.0252686, 983.5686646, -595.6121826, 984.5571899, -1579.5825195, 1579.1809082
4: -530.5634155, 943.0631104, -531.0895386, 944.0128174, -1474.5761719, 1474.1525879

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0266861, upper bound: 1406.0286023
time: 1.14 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0266339, upper bound: 1406.0266339
time: 1.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.52 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -1406.0382724, upper bound: 1406.0361783
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -1406.0389943, upper bound: 1406.0385224
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -1406.0266861, upper bound: 1406.0286023
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -1406.0266339, upper bound: 1406.0266339

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -205.7486877, 520.4184570, -215.5290222, 542.8892822, -748.6379395, 735.9475098
1: -507.2195129, 784.1943970, -531.4324951, 817.6435547, -1324.8627930, 1315.6269531
2: -333.5585022, 761.9496460, -349.5570984, 794.6088867, -1128.1672363, 1111.5067139
3: -545.1074829, 905.3496094, -571.7969971, 944.0959473, -1489.2033691, 1477.1466064
4: -487.2612610, 867.7173462, -510.0206604, 905.0498657, -1392.3110352, 1377.7380371

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0269750, upper bound: 1406.0243910
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0151810, upper bound: 1406.0179545
time: 1.30 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -207.6932526, 525.7196655, -217.5018158, 548.6289673, -756.3221436, 743.2214966
1: -511.9517822, 792.3364258, -536.3890381, 826.6774902, -1338.6292725, 1328.7254639
2: -336.6435852, 769.9976196, -352.6528320, 803.4030151, -1140.0466309, 1122.6503906
3: -550.0789185, 914.7443237, -576.7852783, 954.5651855, -1504.6440430, 1491.5292969
4: -491.7932434, 876.7706909, -514.4527588, 915.1572266, -1406.9504395, 1391.2233887

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0192292, upper bound: 1406.0152958
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228
time: 1.15 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -221.4534912, 558.3242798, -220.3337860, 555.5056763, -776.9591675, 778.6580811
1: -546.1316528, 840.9113770, -543.3676758, 836.6107788, -1382.7423096, 1384.2789307
2: -359.1257935, 817.4265747, -357.2923584, 813.2608032, -1172.3863525, 1174.7189941
3: -587.2861938, 971.0214844, -584.2808838, 966.0902710, -1553.3763428, 1555.3023682
4: -523.7177124, 931.0462646, -521.0192871, 926.3430786, -1450.0603027, 1452.0655518

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0242945, upper bound: 1406.0263574
time: 1.21 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0152958, upper bound: 1406.0192292
time: 0.89 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -217.9383240, 549.5402222, -239.8104095, 604.6921997, -822.6304321, 789.3506470
1: -536.7916260, 827.8152466, -589.7391968, 912.1450195, -1448.9362793, 1417.5544434
2: -353.3766785, 804.2095337, -388.8984375, 885.3978271, -1238.7744141, 1193.1079102
3: -578.1560669, 955.5751343, -635.5425415, 1052.9440918, -1631.1000977, 1591.1173096
4: -515.7151489, 916.0606079, -567.3358154, 1008.4573364, -1524.1724854, 1483.3964844

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0179545, upper bound: 1406.0151810
time: 1.22 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228
time: 0.84 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.94 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -1406.0269750, upper bound: 1406.0243910
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.94
Output dim: 3, lower bound: -1406.0151810, upper bound: 1406.0179545
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.94
Output dim: 3, lower bound: -1406.0192292, upper bound: 1406.0152958
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.94
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.94
Output dim: 3, lower bound: -1406.0242945, upper bound: 1406.0263574
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.94
Output dim: 3, lower bound: -1406.0152958, upper bound: 1406.0192292
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.94
Output dim: 3, lower bound: -1406.0179545, upper bound: 1406.0151810
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.94
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -201.1527252, 508.8288269, -212.6513519, 535.5696411, -736.7223511, 721.4801636
1: -495.5373230, 766.6124878, -524.0734863, 806.4884033, -1302.0255127, 1290.6860352
2: -325.9846497, 744.9215698, -344.8157043, 783.8341064, -1109.8187256, 1089.7373047
3: -532.6974487, 885.1605835, -563.9888306, 931.2999268, -1463.9973145, 1449.1491699
4: -476.2731628, 848.4094849, -503.1363220, 892.8137207, -1369.0869141, 1351.5456543

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0233555, upper bound: 1406.0213441
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0269750, upper bound: 1406.0243224
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -215.4508209, 542.4515381, -215.2767334, 542.4143066, -757.8651123, 757.7281494
1: -531.1887207, 816.6318970, -530.8089600, 816.7133179, -1347.9019775, 1347.4404297
2: -349.4425354, 793.8420410, -349.0822144, 793.8723755, -1143.3149414, 1142.9239502
3: -571.4398193, 943.1051636, -570.9362183, 943.1737671, -1514.6135254, 1514.0411377
4: -509.6448975, 904.2141113, -509.1067505, 904.3125000, -1413.9573975, 1413.3208008

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0231929, upper bound: 1406.0252757
time: 1.27 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0242238, upper bound: 1406.0263574
time: 1.08 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.22 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 3, lower bound: -1406.0233555, upper bound: 1406.0213441
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -1406.0269750, upper bound: 1406.0243224
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 3, lower bound: -1406.0231929, upper bound: 1406.0252757
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 3, lower bound: -1406.0242238, upper bound: 1406.0263574

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -200.1634369, 506.4707947, -211.3369141, 532.4796143, -732.6430054, 717.8074951
1: -493.1672668, 763.0691528, -520.9604492, 801.8482666, -1295.0155029, 1284.0295410
2: -324.4069214, 741.4686890, -342.7275696, 779.3166504, -1103.7236328, 1084.1962891
3: -530.1071777, 881.0844727, -560.5358276, 925.9533081, -1456.0603027, 1441.6202393
4: -473.9879456, 844.5063477, -500.0948486, 887.7079468, -1361.6956787, 1344.6010742

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0262782, upper bound: 1406.0240595
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0263796, upper bound: 1406.0238812
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -214.4959259, 540.1928101, -213.9555206, 539.2822876, -753.7781982, 754.1483154
1: -528.9188843, 813.2450562, -527.6546021, 811.9982910, -1340.9171143, 1340.8992920
2: -347.9216919, 790.5354004, -346.9831543, 789.2886353, -1137.2100830, 1137.5185547
3: -568.9384766, 939.2000122, -567.4717407, 937.7504883, -1506.6888428, 1506.6717529
4: -507.4324036, 900.4807129, -506.0665283, 899.1182861, -1406.5506592, 1406.5472412

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0237888, upper bound: 1406.0261993
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0238158, upper bound: 1406.0259141
time: 1.28 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.18 seconds
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 3, lower bound: -1406.0262782, upper bound: 1406.0240595
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.18
Output dim: 3, lower bound: -1406.0263796, upper bound: 1406.0238812
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.18
Output dim: 3, lower bound: -1406.0237888, upper bound: 1406.0261993
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.18
Output dim: 3, lower bound: -1406.0238158, upper bound: 1406.0259141

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -197.2379913, 499.0087585, -209.4458771, 527.6596680, -724.8975220, 708.4546509
1: -486.0411072, 751.8049927, -516.3704224, 794.5749512, -1280.6159668, 1268.1754150
2: -319.6806335, 730.4655151, -339.6562195, 772.2304688, -1091.9108887, 1070.1217041
3: -522.3820801, 868.1046753, -555.5464478, 917.5773315, -1439.9589844, 1423.6511230
4: -467.0731201, 831.9949341, -495.6141052, 879.6511841, -1346.7243652, 1327.6090088

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0248720, upper bound: 1406.0194583
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0260577, upper bound: 1406.0234225
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -200.2473602, 508.1918030, -208.1399994, 524.6597290, -724.9070435, 716.3316650
1: -493.6811523, 765.7886353, -513.3057251, 790.0888672, -1283.7700195, 1279.0941162
2: -324.6778564, 744.4235840, -337.6492004, 767.9367676, -1092.6146240, 1082.0727539
3: -530.1417236, 884.1505127, -552.0212402, 912.3814697, -1442.5230713, 1436.1717529
4: -474.1539307, 847.8868408, -492.4948120, 874.7452393, -1348.8991699, 1340.3815918

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0249121, upper bound: 1406.0194115
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0261268, upper bound: 1406.0232858
time: 1.46 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.88 seconds
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.88
Output dim: 3, lower bound: -1406.0248720, upper bound: 1406.0194583
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.88
Output dim: 3, lower bound: -1406.0260577, upper bound: 1406.0234225
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.88
Output dim: 3, lower bound: -1406.0249121, upper bound: 1406.0194115
NS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.88
Output dim: 3, lower bound: -1406.0261268, upper bound: 1406.0232858

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.25 + 40.03 = 43.28 seconds
