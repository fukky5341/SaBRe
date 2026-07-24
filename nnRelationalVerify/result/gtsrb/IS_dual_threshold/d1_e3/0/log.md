## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 1800 seconds
Split limit: 100
Threshold: 16.9602713514


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=59, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-27.3553810, 8.0059156, -27.3553810, 8.0059156, -35.1541176, 35.1541214)
1: (-14.3738527, 13.1456423, -14.3738527, 13.1456423, -27.5194950, 27.5194950)
2: (-12.0486050, 13.7512474, -12.0486050, 13.7512474, -25.7295609, 25.7295647)
3: (-8.4818811, 19.2768917, -8.4818811, 19.2768917, -27.7587738, 27.7587738)
4: (-15.3400288, 12.7616272, -15.3400288, 12.7616272, -28.1016560, 28.1016560)
5: (-8.4925900, 21.8095169, -8.4925900, 21.8095169, -30.3021069, 30.3021069)
6: (-22.5647430, 6.0934381, -22.5647430, 6.0934381, -24.2854004, 24.2854004)
7: (-14.2568521, 20.5986767, -14.2568521, 20.5986767, -34.8555298, 34.8555298)
8: (-15.1975794, 17.3974724, -15.1975794, 17.3974724, -32.5941238, 32.5941238)
9: (-16.2352715, 14.0409336, -16.2352715, 14.0409336, -29.6555939, 29.6555939)
10: (-22.3045483, 24.2655678, -22.3045483, 24.2655678, -46.5701141, 46.5701141)
11: (-26.9328022, 14.1697321, -26.9328022, 14.1697321, -41.1025352, 41.1025352)
12: (-25.9868717, 13.1568661, -25.9868717, 13.1568661, -38.3463821, 38.3463821)
13: (-28.1229744, 8.8471899, -28.1229744, 8.8471899, -36.9701653, 36.9701653)
14: (-49.2634850, 3.5757933, -49.2634850, 3.5757933, -48.3050995, 48.3050919)
15: (-18.0831642, 10.8290596, -18.0831642, 10.8290596, -28.8070183, 28.8070183)
16: (-22.6058140, 18.6420555, -22.6058140, 18.6420555, -41.2478714, 41.2478714)
17: (-44.0130959, 23.8884239, -44.0130959, 23.8884239, -66.7394104, 66.7394180)
18: (-18.6798954, 9.0962238, -18.6798954, 9.0962238, -27.7761192, 27.7761192)
19: (-22.8621407, 3.1672647, -22.8621407, 3.1672647, -26.0294056, 26.0294056)
20: (-14.4835548, 9.1810989, -14.4835548, 9.1810989, -23.6646538, 23.6646538)
21: (-22.1676178, 9.7627296, -22.1676178, 9.7627296, -31.9303474, 31.9303474)
22: (-27.2011833, 9.8212547, -27.2011833, 9.8212547, -37.0224380, 37.0224380)
23: (-20.7062416, 5.7638655, -20.7062416, 5.7638655, -26.4701080, 26.4701080)
24: (-27.9682446, -0.0671892, -27.9682446, -0.0671892, -27.8639679, 27.8639679)
25: (-19.8669052, 7.6443977, -19.8669052, 7.6443977, -27.5113029, 27.5113029)
26: (-33.2248726, 7.2380404, -33.2248726, 7.2380404, -40.4629135, 40.4629135)
27: (-22.6988029, 9.5920048, -22.6988029, 9.5920048, -31.9442368, 31.9442368)
28: (-20.6286469, 7.0542479, -20.6286469, 7.0542479, -27.6828957, 27.6828957)
29: (-33.0641098, 9.0293598, -33.0641098, 9.0293598, -41.5258026, 41.5257950)
30: (-22.4276619, 8.0514975, -22.4276619, 8.0514975, -30.4791603, 30.4791603)
31: (-20.4918900, 9.0954580, -20.4918900, 9.0954580, -29.5873489, 29.5873489)
32: (-19.6942158, 9.6822872, -19.6942158, 9.6822872, -28.2969055, 28.2969055)
33: (-42.4827957, 5.5264988, -42.4827957, 5.5264988, -45.7264938, 45.7264938)
34: (-31.4582138, 7.5270596, -31.4582138, 7.5270596, -37.2376175, 37.2376175)
35: (-31.8877182, 7.7160897, -31.8877182, 7.7160897, -38.8506241, 38.8506241)
36: (-32.2013626, 7.3777037, -32.2013626, 7.3777037, -38.9149323, 38.9149246)
37: (-49.8116226, -2.3671541, -49.8116226, -2.3671541, -44.0720825, 44.0720978)
38: (-40.5607605, 8.2751951, -40.5607605, 8.2751951, -47.7802429, 47.7802429)
39: (-54.6296539, -1.3925772, -54.6296539, -1.3925772, -52.6431427, 52.6431427)
40: (-40.1360474, 2.8698897, -40.1360474, 2.8698897, -40.2175522, 40.2175522)
41: (-25.8314590, 4.1538486, -25.8314590, 4.1538486, -26.4098854, 26.4098892)
42: (-16.4646549, 8.0450420, -16.4646549, 8.0450420, -23.0334396, 23.0334396)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.76 + 48.33 = 51.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -16.9772486, upper bound: 16.9772486

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 636
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 636

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9753418, upper bound: 16.9483425
time: 24.89 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9753418, upper bound: 16.9765846
time: 32.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 57.27 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 57.27
Output dim: 7, lower bound: -16.9753418, upper bound: 16.9483425
IS_B2, status: Status.UNKNOWN, split count: 1, time: 57.27
Output dim: 7, lower bound: -16.9753418, upper bound: 16.9765846

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -27.3329544, 8.0022383, -27.3032646, 7.9883246, -35.1138382, 35.0980682
1: -14.3521156, 13.1427374, -14.3275595, 13.1194763, -27.4715919, 27.4702969
2: -12.0273056, 13.7486143, -12.0039587, 13.7283335, -25.6843109, 25.6824226
3: -8.4591570, 19.2740746, -8.4348755, 19.2483463, -27.7075043, 27.7089500
4: -15.3178234, 12.7586298, -15.2947998, 12.7390251, -28.0568485, 28.0534286
5: -8.4651518, 21.8057556, -8.4364872, 21.7746372, -30.2397881, 30.2422428
6: -22.5515442, 6.0871267, -22.5322914, 6.0861502, -24.2439079, 24.2375641
7: -14.2269001, 20.5959892, -14.1951733, 20.5670319, -34.7939301, 34.7911606
8: -15.1761570, 17.3946342, -15.1527576, 17.3747025, -32.5501022, 32.5457497
9: -16.2209454, 14.0372448, -16.2043419, 14.0253181, -29.6269836, 29.6228943
10: -22.2946053, 24.2583961, -22.2802219, 24.2461014, -46.5407066, 46.5386200
11: -26.9228287, 14.1573095, -26.9052200, 14.1426144, -41.0654449, 41.0625305
12: -25.9829998, 13.1424961, -25.9659042, 13.1257353, -38.3108521, 38.3113556
13: -28.1057854, 8.8405952, -28.0855904, 8.8207083, -36.9264946, 36.9261856
14: -49.2527084, 3.5710039, -49.2357750, 3.5636883, -48.2764511, 48.2707977
15: -18.0701962, 10.8235703, -18.0558720, 10.8133640, -28.7768059, 28.7733955
16: -22.5764694, 18.6373024, -22.5392418, 18.6134453, -41.1899147, 41.1765442
17: -43.9920883, 23.8828430, -43.9661102, 23.8580780, -66.6772461, 66.6826477
18: -18.6743202, 9.0815887, -18.6604214, 9.0645132, -27.7388344, 27.7420101
19: -22.8570747, 3.1497881, -22.8332901, 3.1318169, -25.9888916, 25.9830780
20: -14.4789419, 9.1699762, -14.4645557, 9.1564007, -23.6353416, 23.6345329
21: -22.1610241, 9.7490635, -22.1424065, 9.7346430, -31.8956680, 31.8914700
22: -27.1948433, 9.8022118, -27.1755276, 9.7814169, -36.9762611, 36.9777374
23: -20.7003632, 5.7413616, -20.6681099, 5.7176166, -26.4179802, 26.4094715
24: -27.9642677, -0.0899911, -27.9382496, -0.1147118, -27.8114090, 27.8036575
25: -19.8625374, 7.6264691, -19.8467560, 7.6067958, -27.4693336, 27.4732246
26: -33.2189140, 7.2074428, -33.1848221, 7.1728544, -40.3917694, 40.3922653
27: -22.6925240, 9.5726547, -22.6699963, 9.5519381, -31.8956451, 31.8865738
28: -20.6249809, 7.0339022, -20.6017857, 7.0111017, -27.6360817, 27.6356888
29: -33.0564880, 9.0119362, -33.0363007, 8.9926167, -41.4804840, 41.4800415
30: -22.4242458, 8.0414448, -22.4147758, 8.0292339, -30.4534798, 30.4562206
31: -20.4848633, 9.0804920, -20.4597492, 9.0651741, -29.5500374, 29.5402412
32: -19.6874161, 9.6665201, -19.6649055, 9.6503582, -28.2570038, 28.2482109
33: -42.4765930, 5.5077868, -42.4578705, 5.4868059, -45.6805267, 45.6832352
34: -31.4520626, 7.4962153, -31.4214096, 7.4640231, -37.1674042, 37.1659012
35: -31.8832474, 7.6918154, -31.8575630, 7.6667395, -38.7970886, 38.7963867
36: -32.1953163, 7.3513093, -32.1629753, 7.3244643, -38.8556595, 38.8489914
37: -49.8031616, -2.3948531, -49.7605476, -2.4246340, -44.0043106, 43.9923019
38: -40.5531578, 8.2427864, -40.5145454, 8.2067509, -47.7037048, 47.6987000
39: -54.6209068, -1.4183750, -54.5822105, -1.4456024, -52.5810242, 52.5695801
40: -40.1293297, 2.8586526, -40.1112633, 2.8457599, -40.1881332, 40.1827011
41: -25.8247643, 4.1351018, -25.8006535, 4.1155252, -26.3655548, 26.3593292
42: -16.4576550, 8.0364552, -16.4442120, 8.0262794, -23.0081177, 23.0030136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=58, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1678

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9730646, upper bound: 16.9281936
time: 27.43 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9730646, upper bound: 16.9460595
time: 27.37 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -27.3541718, 8.0057354, -27.3528595, 8.0054846, -35.1524506, 35.1491547
1: -14.3731127, 13.1454325, -14.3722734, 13.1452065, -27.5183182, 27.5177059
2: -12.0478029, 13.7510948, -12.0468979, 13.7509356, -25.7289772, 25.7274590
3: -8.4810190, 19.2766991, -8.4800825, 19.2764740, -27.7574921, 27.7567825
4: -15.3389997, 12.7613783, -15.3378572, 12.7611122, -28.1001129, 28.0992355
5: -8.4915714, 21.8093262, -8.4904604, 21.8090820, -30.3006535, 30.2997856
6: -22.5621338, 6.0930166, -22.5590897, 6.0925665, -24.2871399, 24.2822456
7: -14.2558365, 20.5984344, -14.2546921, 20.5981483, -34.8539848, 34.8531265
8: -15.1965055, 17.3972511, -15.1953545, 17.3970242, -32.5925903, 32.5890198
9: -16.2345333, 14.0407171, -16.2337112, 14.0404720, -29.6537590, 29.6505280
10: -22.3037491, 24.2630997, -22.3028774, 24.2602291, -46.5639801, 46.5659790
11: -26.9323578, 14.1676846, -26.9318504, 14.1653309, -41.0976868, 41.0995331
12: -25.9866409, 13.1557026, -25.9863510, 13.1545267, -38.3363419, 38.3442078
13: -28.1222267, 8.8468695, -28.1214218, 8.8464985, -36.9687271, 36.9682922
14: -49.2627335, 3.5750113, -49.2619476, 3.5740175, -48.3065033, 48.3012924
15: -18.0822487, 10.8263741, -18.0813084, 10.8233471, -28.8013687, 28.8028336
16: -22.6045036, 18.6416626, -22.6030235, 18.6411934, -41.2456970, 41.2446861
17: -44.0118866, 23.8878708, -44.0106201, 23.8872585, -66.7435760, 66.7342682
18: -18.6795349, 9.0953236, -18.6791306, 9.0943260, -27.7738609, 27.7744541
19: -22.8618851, 3.1665735, -22.8615894, 3.1657603, -26.0276451, 26.0281639
20: -14.4832792, 9.1802654, -14.4829216, 9.1792564, -23.6625366, 23.6631870
21: -22.1672325, 9.7609377, -22.1668110, 9.7588787, -31.9261112, 31.9277496
22: -27.2007484, 9.8183784, -27.2002792, 9.8149557, -37.0157051, 37.0186577
23: -20.7059860, 5.7630253, -20.7056885, 5.7621002, -26.4680862, 26.4687138
24: -27.9680042, -0.0680461, -27.9677086, -0.0690551, -27.8600159, 27.8676834
25: -19.8666687, 7.6435089, -19.8664017, 7.6424799, -27.5091476, 27.5099106
26: -33.2245026, 7.2366590, -33.2240753, 7.2350912, -40.4595947, 40.4607353
27: -22.6984673, 9.5910816, -22.6980705, 9.5900307, -31.9397430, 31.9491882
28: -20.6284714, 7.0534258, -20.6282234, 7.0525265, -27.6809978, 27.6816483
29: -33.0634995, 9.0264387, -33.0628510, 9.0229530, -41.5173035, 41.5215454
30: -22.4273567, 8.0501146, -22.4270325, 8.0485516, -30.4759083, 30.4771461
31: -20.4915180, 9.0945883, -20.4911098, 9.0937433, -29.5852623, 29.5856972
32: -19.6937599, 9.6814613, -19.6932354, 9.6805267, -28.2823792, 28.2949066
33: -42.4824181, 5.5257921, -42.4820328, 5.5250177, -45.7128220, 45.7249146
34: -31.4579086, 7.5260057, -31.4575367, 7.5248017, -37.2100983, 37.2357178
35: -31.8874207, 7.7152967, -31.8870964, 7.7144032, -38.8336334, 38.8492661
36: -32.2009583, 7.3768201, -32.2005081, 7.3758807, -38.9125824, 38.9135895
37: -49.8108864, -2.3680320, -49.8101234, -2.3690066, -44.0684204, 44.0696945
38: -40.5603790, 8.2740946, -40.5598946, 8.2727957, -47.7771378, 47.7787323
39: -54.6291771, -1.3933439, -54.6285210, -1.3943186, -52.6407318, 52.6412048
40: -40.1354942, 2.8694444, -40.1348572, 2.8689761, -40.2131119, 40.2152939
41: -25.8311176, 4.1531467, -25.8307152, 4.1522999, -26.3994904, 26.4083481
42: -16.4643192, 8.0442219, -16.4639301, 8.0433006, -23.0315781, 23.0321293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=59, inp2_unstable=58, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1678
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1678

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9730646, upper bound: 16.9281936
time: 24.68 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9730646, upper bound: 16.9460595
time: 35.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 62.04 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 62.04
Output dim: 7, lower bound: -16.9730646, upper bound: 16.9281936
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 62.04
Output dim: 7, lower bound: -16.9730646, upper bound: 16.9460595
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 62.04
Output dim: 7, lower bound: -16.9730646, upper bound: 16.9281936
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 62.04
Output dim: 7, lower bound: -16.9730646, upper bound: 16.9460595

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -27.3267975, 7.9888449, -27.3005791, 7.9824533, -35.1019058, 35.0821075
1: -14.3489752, 13.1181316, -14.3261690, 13.1087141, -27.4576893, 27.4443016
2: -12.0256529, 13.7371025, -12.0032234, 13.7232780, -25.6772118, 25.6693535
3: -8.4580059, 19.2526970, -8.4343624, 19.2389984, -27.6970043, 27.6870594
4: -15.3150959, 12.7380419, -15.2935944, 12.7300196, -28.0451164, 28.0316353
5: -8.4636192, 21.7866459, -8.4358082, 21.7662735, -30.2298927, 30.2224541
6: -22.5486488, 6.0835400, -22.5310097, 6.0845680, -24.2383728, 24.2325554
7: -14.2240601, 20.5734901, -14.1939611, 20.5571899, -34.7812500, 34.7674522
8: -15.1731701, 17.3729095, -15.1514502, 17.3652039, -32.5375519, 32.5226440
9: -16.2173805, 14.0203772, -16.2027988, 14.0179396, -29.6160469, 29.6043854
10: -22.2890282, 24.2486420, -22.2777977, 24.2418137, -46.5308418, 46.5264397
11: -26.9133148, 14.1549053, -26.9010563, 14.1415701, -41.0548859, 41.0559616
12: -25.9655972, 13.1396255, -25.9582787, 13.1244488, -38.2908211, 38.3001442
13: -28.0995083, 8.8367672, -28.0828285, 8.8190346, -36.9185410, 36.9195938
14: -49.2413406, 3.5696878, -49.2307587, 3.5631371, -48.2648773, 48.2623749
15: -18.0669327, 10.8105354, -18.0544434, 10.8076572, -28.7666168, 28.7568665
16: -22.5695763, 18.6091995, -22.5362129, 18.6011772, -41.1707535, 41.1454124
17: -43.9805298, 23.8730927, -43.9609795, 23.8537712, -66.6582794, 66.6613770
18: -18.6692238, 9.0791559, -18.6581898, 9.0634575, -27.7326813, 27.7373466
19: -22.8432884, 3.1489060, -22.8272324, 3.1314685, -25.9747562, 25.9761391
20: -14.4555559, 9.1677265, -14.4543200, 9.1554270, -23.6109829, 23.6220474
21: -22.1464977, 9.7470083, -22.1360283, 9.7337399, -31.8802376, 31.8830376
22: -27.1704636, 9.8004141, -27.1648750, 9.7806396, -36.9511032, 36.9652901
23: -20.6830158, 5.7395773, -20.6605186, 5.7168388, -26.3998547, 26.4000969
24: -27.9428864, -0.0909739, -27.9288940, -0.1151323, -27.7880554, 27.7924652
25: -19.8406219, 7.6251836, -19.8371620, 7.6062298, -27.4468517, 27.4623451
26: -33.2008514, 7.2051272, -33.1769180, 7.1718450, -40.3726959, 40.3820457
27: -22.6876831, 9.5708838, -22.6678448, 9.5511580, -31.8898621, 31.8825836
28: -20.6059380, 7.0319829, -20.5934601, 7.0102353, -27.6161728, 27.6254425
29: -33.0339394, 9.0104218, -33.0264549, 8.9919529, -41.4570847, 41.4686203
30: -22.4044800, 8.0388517, -22.4061394, 8.0281143, -30.4325943, 30.4449921
31: -20.4699478, 9.0786486, -20.4531975, 9.0643768, -29.5343246, 29.5318451
32: -19.6731071, 9.6634941, -19.6586494, 9.6490154, -28.2360306, 28.2365036
33: -42.4641037, 5.5052843, -42.4523087, 5.4856968, -45.6662674, 45.6748962
34: -31.4416199, 7.4934158, -31.4168587, 7.4627991, -37.1552963, 37.1582947
35: -31.8674355, 7.6899300, -31.8505936, 7.6659331, -38.7804413, 38.7874451
36: -32.1746712, 7.3495221, -32.1539612, 7.3236904, -38.8318558, 38.8369293
37: -49.7843475, -2.3961964, -49.7523499, -2.4252162, -43.9820480, 43.9812851
38: -40.5301437, 8.2387772, -40.5045052, 8.2049341, -47.6732407, 47.6818695
39: -54.5993881, -1.4195490, -54.5728111, -1.4461546, -52.5578918, 52.5584869
40: -40.1241341, 2.8568225, -40.1090088, 2.8449368, -40.1792984, 40.1768188
41: -25.8209763, 4.1315341, -25.7990036, 4.1139560, -26.3604507, 26.3547745
42: -16.4535027, 8.0328722, -16.4423943, 8.0246639, -22.9958954, 22.9948158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=58, inp2_unstable=58, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1628

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9649120, upper bound: 16.9012098
time: 35.41 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9649120, upper bound: 16.9200579
time: 32.32 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -27.3760681, 8.0046806, -27.3027515, 7.9873567, -35.1560669, 35.0996284
1: -14.3928833, 13.1429701, -14.3273335, 13.1178303, -27.5107136, 27.4703026
2: -12.0500860, 13.7494602, -12.0037746, 13.7275352, -25.7076874, 25.6815910
3: -8.4896011, 19.2746716, -8.4347477, 19.2469292, -27.7365303, 27.7094193
4: -15.3541555, 12.7592583, -15.2945480, 12.7375774, -28.0917320, 28.0538063
5: -8.4950018, 21.8063984, -8.4363155, 21.7732925, -30.2682953, 30.2427139
6: -22.5672760, 6.0928645, -22.5320568, 6.0849576, -24.2470589, 24.2553825
7: -14.2649622, 20.5963345, -14.1949558, 20.5655499, -34.8305130, 34.7912903
8: -15.2197208, 17.3955975, -15.1524630, 17.3732834, -32.5921059, 32.5458298
9: -16.2488136, 14.0400133, -16.2040901, 14.0241585, -29.6535835, 29.6247368
10: -22.3172493, 24.2645683, -22.2797680, 24.2455330, -46.5627823, 46.5443344
11: -26.9308472, 14.1723433, -26.9041176, 14.1424084, -41.0732574, 41.0764618
12: -25.9863052, 13.1599255, -25.9647598, 13.1252842, -38.3114700, 38.3297310
13: -28.1117477, 8.8500347, -28.0841656, 8.8202820, -36.9320297, 36.9342003
14: -49.2652359, 3.5748768, -49.2341995, 3.5635252, -48.2982330, 48.2656326
15: -18.1037693, 10.8293533, -18.0555382, 10.8124523, -28.8117752, 28.7761688
16: -22.6204567, 18.6413994, -22.5387001, 18.6116409, -41.2320976, 41.1800995
17: -44.0086899, 23.8873940, -43.9651451, 23.8575649, -66.7041321, 66.6751404
18: -18.6788597, 9.0897655, -18.6599064, 9.0643406, -27.7432003, 27.7496719
19: -22.8601646, 3.1729817, -22.8322868, 3.1317971, -25.9919624, 26.0052681
20: -14.4812031, 9.1983318, -14.4629526, 9.1561832, -23.6373863, 23.6612854
21: -22.1670990, 9.7668171, -22.1414108, 9.7344360, -31.9015350, 31.9082279
22: -27.1990833, 9.8323441, -27.1738777, 9.7812567, -36.9803391, 37.0062218
23: -20.7023048, 5.7664404, -20.6666832, 5.7174611, -26.4197655, 26.4331245
24: -27.9652939, -0.0589662, -27.9366779, -0.1148171, -27.8099060, 27.8357086
25: -19.8652763, 7.6584907, -19.8451920, 7.6066408, -27.4719162, 27.5036831
26: -33.2216911, 7.2371621, -33.1824341, 7.1726589, -40.3943481, 40.4195976
27: -22.6960716, 9.5801802, -22.6694717, 9.5518169, -31.8986359, 31.8945160
28: -20.6258984, 7.0645504, -20.6003914, 7.0109329, -27.6368313, 27.6649418
29: -33.0632019, 9.0379963, -33.0347862, 8.9924812, -41.4851303, 41.5045853
30: -22.4270134, 8.0692329, -22.4132786, 8.0289650, -30.4559784, 30.4825115
31: -20.4903984, 9.1072407, -20.4586067, 9.0650253, -29.5554237, 29.5658474
32: -19.6917248, 9.6866970, -19.6637249, 9.6500387, -28.2517014, 28.2773514
33: -42.4821548, 5.5297012, -42.4568787, 5.4865999, -45.6848831, 45.7058563
34: -31.4532661, 7.5153837, -31.4205971, 7.4637990, -37.1687927, 37.1842651
35: -31.8852863, 7.7145205, -31.8563957, 7.6665955, -38.7982025, 38.8181076
36: -32.1978416, 7.3784418, -32.1615677, 7.3243237, -38.8545685, 38.8777542
37: -49.8105278, -2.3797536, -49.7591438, -2.4247570, -44.0092010, 44.0071182
38: -40.5579376, 8.2685127, -40.5130005, 8.2063961, -47.7002640, 47.7308502
39: -54.6275406, -1.3933210, -54.5806808, -1.4457235, -52.5867004, 52.5931549
40: -40.1351395, 2.8669167, -40.1108131, 2.8456149, -40.1884995, 40.1960678
41: -25.8329353, 4.1404963, -25.8002434, 4.1145844, -26.3711319, 26.3684349
42: -16.4686565, 8.0400333, -16.4437523, 8.0253477, -22.9962959, 23.0210514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=58, inp2_unstable=58, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1628

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9649120, upper bound: 16.9190663
time: 31.67 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9649120, upper bound: 16.9378873
time: 37.20 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -27.3480110, 7.9923306, -27.3501511, 7.9996214, -35.1405067, 35.1332092
1: -14.3699589, 13.1208324, -14.3708954, 13.1344242, -27.5043831, 27.4917278
2: -12.0461416, 13.7395916, -12.0461750, 13.7458839, -25.7219276, 25.7143517
3: -8.4799061, 19.2553120, -8.4795771, 19.2671185, -27.7470245, 27.7348900
4: -15.3362780, 12.7408094, -15.3366623, 12.7521152, -28.0883942, 28.0774727
5: -8.4900208, 21.7902241, -8.4897757, 21.8007317, -30.2907524, 30.2799988
6: -22.5592060, 6.0894375, -22.5578117, 6.0909872, -24.2816124, 24.2772350
7: -14.2529945, 20.5759411, -14.2534485, 20.5883369, -34.8413315, 34.8293915
8: -15.1935320, 17.3755188, -15.1940603, 17.3875160, -32.5800247, 32.5659561
9: -16.2309647, 14.0238266, -16.2321453, 14.0330944, -29.6428032, 29.6320915
10: -22.2982025, 24.2532997, -22.3004417, 24.2559395, -46.5541420, 46.5537415
11: -26.9228706, 14.1652565, -26.9277000, 14.1642790, -41.0871506, 41.0929565
12: -25.9692154, 13.1527977, -25.9787407, 13.1532784, -38.3163071, 38.3330002
13: -28.1159496, 8.8430367, -28.1186409, 8.8447742, -36.9607239, 36.9616776
14: -49.2514229, 3.5736427, -49.2569656, 3.5734596, -48.2949066, 48.2928238
15: -18.0789833, 10.8133411, -18.0798664, 10.8176317, -28.7911797, 28.7863045
16: -22.5976257, 18.6135426, -22.5999985, 18.6289177, -41.2265434, 41.2135391
17: -44.0003357, 23.8781586, -44.0055428, 23.8829536, -66.7245941, 66.7130966
18: -18.6744461, 9.0929012, -18.6769066, 9.0932837, -27.7677307, 27.7698078
19: -22.8481007, 3.1657054, -22.8555412, 3.1653819, -26.0134830, 26.0212460
20: -14.4598684, 9.1780167, -14.4727087, 9.1782904, -23.6381588, 23.6507263
21: -22.1527100, 9.7588692, -22.1604347, 9.7579823, -31.9106922, 31.9193039
22: -27.1763687, 9.8165436, -27.1896133, 9.8141537, -36.9905243, 37.0061569
23: -20.6886520, 5.7612610, -20.6980858, 5.7613306, -26.4499817, 26.4593468
24: -27.9466133, -0.0690141, -27.9583626, -0.0694203, -27.8366776, 27.8564758
25: -19.8447571, 7.6422043, -19.8568153, 7.6419215, -27.4866791, 27.4990196
26: -33.2064667, 7.2343216, -33.2162094, 7.2340808, -40.4405479, 40.4505310
27: -22.6935940, 9.5893326, -22.6959400, 9.5892830, -31.9339752, 31.9452362
28: -20.6094398, 7.0515299, -20.6198959, 7.0517111, -27.6611519, 27.6714249
29: -33.0409355, 9.0249472, -33.0529633, 9.0222998, -41.4938965, 41.5101318
30: -22.4076157, 8.0475368, -22.4183617, 8.0474148, -30.4550304, 30.4658985
31: -20.4765854, 9.0927210, -20.4845715, 9.0929356, -29.5695210, 29.5772934
32: -19.6794395, 9.6784725, -19.6869907, 9.6791992, -28.2614288, 28.2832527
33: -42.4699326, 5.5233297, -42.4764786, 5.5239182, -45.6985168, 45.7166290
34: -31.4474659, 7.5231957, -31.4529877, 7.5235472, -37.1980209, 37.2281113
35: -31.8716049, 7.7134018, -31.8801670, 7.7135968, -38.8169708, 38.8402939
36: -32.1802979, 7.3750405, -32.1914444, 7.3750930, -38.8889008, 38.9015656
37: -49.7920876, -2.3693542, -49.8019066, -2.3696165, -44.0461426, 44.0586777
38: -40.5374146, 8.2700253, -40.5498734, 8.2710276, -47.7467041, 47.7618027
39: -54.6075974, -1.3946314, -54.6191635, -1.3948507, -52.6175995, 52.6300354
40: -40.1302643, 2.8675900, -40.1326256, 2.8681302, -40.2042618, 40.2093735
41: -25.8273048, 4.1495857, -25.8290539, 4.1507430, -26.3943710, 26.4037857
42: -16.4601631, 8.0406265, -16.4621086, 8.0416889, -23.0193634, 23.0239429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=58, inp2_unstable=58, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1628

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9460777, upper bound: 16.9483741
time: 25.82 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9460777, upper bound: 16.9200582
time: 33.17 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -27.3972702, 8.0081549, -27.3523216, 8.0045280, -35.1947098, 35.1507568
1: -14.4138861, 13.1456451, -14.3720369, 13.1435204, -27.5574074, 27.5176811
2: -12.0705948, 13.7519360, -12.0467281, 13.7501297, -25.7524147, 25.7265854
3: -8.5114613, 19.2772675, -8.4799623, 19.2750359, -27.7864971, 27.7572289
4: -15.3753386, 12.7620258, -15.3376122, 12.7596798, -28.1350174, 28.0996380
5: -8.5214396, 21.8099899, -8.4903030, 21.8077316, -30.3291702, 30.3002930
6: -22.5778637, 6.0987473, -22.5588646, 6.0913773, -24.2902870, 24.3000565
7: -14.2938833, 20.5987625, -14.2544746, 20.5966873, -34.8905716, 34.8532372
8: -15.2400465, 17.3981895, -15.1950817, 17.3955803, -32.6345634, 32.5890656
9: -16.2623978, 14.0434933, -16.2334557, 14.0393133, -29.6803513, 29.6524353
10: -22.3264046, 24.2692261, -22.3024311, 24.2596321, -46.5860367, 46.5716553
11: -26.9403763, 14.1827478, -26.9307404, 14.1651449, -41.1055222, 41.1134872
12: -25.9899521, 13.1731071, -25.9851627, 13.1540813, -38.3369331, 38.3626060
13: -28.1281929, 8.8563108, -28.1199875, 8.8460808, -36.9742737, 36.9762993
14: -49.2752724, 3.5788431, -49.2603874, 3.5738697, -48.3282928, 48.2961273
15: -18.1158218, 10.8321457, -18.0809708, 10.8224335, -28.8363266, 28.8055763
16: -22.6485176, 18.6457253, -22.6024895, 18.6393929, -41.2879105, 41.2482147
17: -44.0284958, 23.8924618, -44.0096893, 23.8867321, -66.7704773, 66.7268600
18: -18.6840916, 9.1034956, -18.6786194, 9.0941677, -27.7782593, 27.7821159
19: -22.8649559, 3.1897883, -22.8606071, 3.1657028, -26.0306587, 26.0503960
20: -14.4855146, 9.2086277, -14.4813328, 9.1790485, -23.6645622, 23.6899605
21: -22.1733284, 9.7786789, -22.1657944, 9.7586699, -31.9319992, 31.9444733
22: -27.2049751, 9.8484974, -27.1986122, 9.8147745, -37.0197487, 37.0471115
23: -20.7079639, 5.7881460, -20.7042694, 5.7619514, -26.4699154, 26.4924164
24: -27.9690247, -0.0369973, -27.9661255, -0.0691190, -27.8585510, 27.8997192
25: -19.8694191, 7.6755276, -19.8648338, 7.6423235, -27.5117416, 27.5403614
26: -33.2273178, 7.2663803, -33.2217216, 7.2349138, -40.4622307, 40.4881020
27: -22.7019939, 9.5986414, -22.6975689, 9.5899258, -31.9427109, 31.9571915
28: -20.6293869, 7.0840883, -20.6268539, 7.0523767, -27.6817627, 27.7109413
29: -33.0701790, 9.0524693, -33.0613327, 9.0228157, -41.5219421, 41.5460815
30: -22.4301529, 8.0779123, -22.4255047, 8.0482845, -30.4784374, 30.5034180
31: -20.4970531, 9.1213388, -20.4899788, 9.0936079, -29.5906601, 29.6113167
32: -19.6980648, 9.7016392, -19.6920681, 9.6802254, -28.2770691, 28.3240585
33: -42.4880066, 5.5477343, -42.4810410, 5.5247793, -45.7171631, 45.7475510
34: -31.4591293, 7.5451941, -31.4567432, 7.5245724, -37.2114792, 37.2541199
35: -31.8894730, 7.7380118, -31.8859863, 7.7142696, -38.8348083, 38.8709412
36: -32.2034073, 7.4039850, -32.1990967, 7.3757548, -38.9115677, 38.9423218
37: -49.8182487, -2.3529348, -49.8087082, -2.3691635, -44.0733566, 44.0845337
38: -40.5651016, 8.2998247, -40.5583992, 8.2724676, -47.7737274, 47.8108292
39: -54.6357231, -1.3683348, -54.6269302, -1.3944130, -52.6464539, 52.6647949
40: -40.1413422, 2.8776970, -40.1343994, 2.8688045, -40.2134399, 40.2286377
41: -25.8392982, 4.1585336, -25.8303261, 4.1513600, -26.4050674, 26.4174614
42: -16.4753208, 8.0477667, -16.4634857, 8.0423603, -23.0197678, 23.0502014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=58, inp2_unstable=58, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1628

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9460777, upper bound: 16.9662240
time: 35.40 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9662242, upper bound: 16.9662240
time: 33.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 71.50 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 71.50
Output dim: 7, lower bound: -16.9649120, upper bound: 16.9012098
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 71.50
Output dim: 7, lower bound: -16.9649120, upper bound: 16.9200579
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 71.50
Output dim: 7, lower bound: -16.9649120, upper bound: 16.9190663
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 71.50
Output dim: 7, lower bound: -16.9649120, upper bound: 16.9378873
IS_B2_A1_B1, status: Status.VERIFIED, split count: 3, time: 71.50
Output dim: 7, lower bound: -16.9460777, upper bound: 16.9483741
IS_B2_A1_B2, status: Status.VERIFIED, split count: 3, time: 71.50
Output dim: 7, lower bound: -16.9460777, upper bound: 16.9200582
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 71.50
Output dim: 7, lower bound: -16.9460777, upper bound: 16.9662240
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 71.50
Output dim: 7, lower bound: -16.9662242, upper bound: 16.9662240

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -27.3203735, 7.9765773, -27.2988567, 7.9791503, -35.0921631, 35.0680771
1: -14.3462114, 13.1032019, -14.3254223, 13.1047201, -27.4509315, 27.4286232
2: -12.0237350, 13.7277212, -12.0027199, 13.7207823, -25.6726303, 25.6588287
3: -8.4568882, 19.2366962, -8.4340649, 19.2347221, -27.6916103, 27.6707611
4: -15.3126545, 12.7306061, -15.2929630, 12.7280397, -28.0406952, 28.0235691
5: -8.4621849, 21.7690563, -8.4354315, 21.7615852, -30.2237701, 30.2044868
6: -22.5457745, 6.0815692, -22.5302391, 6.0840392, -24.2346725, 24.2299042
7: -14.2223024, 20.5574265, -14.1934948, 20.5528851, -34.7751884, 34.7509232
8: -15.1713676, 17.3634148, -15.1509581, 17.3626633, -32.5332451, 32.5129700
9: -16.2145462, 14.0036964, -16.2020264, 14.0134621, -29.6087570, 29.5869675
10: -22.2846756, 24.2441559, -22.2766266, 24.2405987, -46.5252762, 46.5207825
11: -26.9068832, 14.1527472, -26.8993225, 14.1409798, -41.0478630, 41.0520706
12: -25.9602242, 13.1367254, -25.9568443, 13.1236792, -38.2856750, 38.2960014
13: -28.0971680, 8.8119736, -28.0822220, 8.8123760, -36.9095459, 36.8941956
14: -49.2372131, 3.5675030, -49.2296219, 3.5625591, -48.2602692, 48.2588959
15: -18.0644913, 10.8014946, -18.0537891, 10.8052177, -28.7615623, 28.7470627
16: -22.5626259, 18.5872498, -22.5343590, 18.5952930, -41.1579208, 41.1216087
17: -43.9777069, 23.8522148, -43.9602203, 23.8481312, -66.6488190, 66.6364822
18: -18.6538582, 9.0773611, -18.6540680, 9.0629854, -27.7168427, 27.7314301
19: -22.8316116, 3.1483767, -22.8241100, 3.1312997, -25.9629116, 25.9724865
20: -14.4349213, 9.1658449, -14.4487820, 9.1549196, -23.5898399, 23.6146278
21: -22.1301231, 9.7460117, -22.1316452, 9.7334900, -31.8636131, 31.8776569
22: -27.1493835, 9.7992373, -27.1591949, 9.7803402, -36.9297256, 36.9584312
23: -20.6685638, 5.7378693, -20.6566620, 5.7163582, -26.3849220, 26.3945312
24: -27.9178391, -0.0922503, -27.9221745, -0.1154861, -27.7613525, 27.7840652
25: -19.8234406, 7.6238079, -19.8325539, 7.6058583, -27.4292984, 27.4563618
26: -33.1820526, 7.2024798, -33.1718864, 7.1711192, -40.3531723, 40.3743668
27: -22.6679306, 9.5693455, -22.6625519, 9.5507650, -31.8693237, 31.8756638
28: -20.5894947, 7.0301571, -20.5890560, 7.0097580, -27.5992527, 27.6192131
29: -33.0169487, 9.0089302, -33.0218658, 8.9915543, -41.4395294, 41.4624405
30: -22.3876953, 8.0364714, -22.4016438, 8.0274887, -30.4151840, 30.4381142
31: -20.4498138, 9.0775423, -20.4478111, 9.0640879, -29.5139008, 29.5253525
32: -19.6595631, 9.6622734, -19.6550121, 9.6487236, -28.2226105, 28.2316933
33: -42.4525261, 5.5039978, -42.4492340, 5.4853420, -45.6552048, 45.6706390
34: -31.4225960, 7.4912210, -31.4117699, 7.4622030, -37.1359253, 37.1509857
35: -31.8564129, 7.6888051, -31.8476543, 7.6656313, -38.7696304, 38.7834396
36: -32.1649475, 7.3483815, -32.1513290, 7.3233976, -38.8216019, 38.8331223
37: -49.7710533, -2.3976879, -49.7488213, -2.4256573, -43.9682236, 43.9762192
38: -40.5087357, 8.2343445, -40.4987411, 8.2037830, -47.6485138, 47.6710358
39: -54.5825005, -1.4206457, -54.5682945, -1.4464207, -52.5403290, 52.5527191
40: -40.1101379, 2.8553033, -40.1052246, 2.8445263, -40.1649704, 40.1714249
41: -25.8157749, 4.1301069, -25.7975922, 4.1135998, -26.3546944, 26.3518105
42: -16.4501667, 8.0312347, -16.4414997, 8.0242252, -22.9906044, 22.9916801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=57, inp2_unstable=58, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1694

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9547104, upper bound: 16.8998179
time: 34.50 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9547104, upper bound: 16.8998179
time: 34.42 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -27.3995953, 7.9936681, -27.2996597, 7.9809947, -35.1734085, 35.0855637
1: -14.4024668, 13.1195526, -14.3256836, 13.1070175, -27.5094833, 27.4452362
2: -12.0655937, 13.7379370, -12.0027084, 13.7221880, -25.7181015, 25.6679268
3: -8.5093126, 19.2554169, -8.4337482, 19.2371960, -27.7465096, 27.6891651
4: -15.3647118, 12.7397480, -15.2931185, 12.7288151, -28.0935268, 28.0328674
5: -8.5201035, 21.7910919, -8.4351587, 21.7642727, -30.2843761, 30.2262497
6: -22.5514545, 6.1060863, -22.5301552, 6.0841222, -24.2510147, 24.2452202
7: -14.2819939, 20.5764561, -14.1932688, 20.5554085, -34.8374023, 34.7697258
8: -15.2168493, 17.3761253, -15.1508923, 17.3640938, -32.5796165, 32.5244370
9: -16.2810898, 14.0220442, -16.2023983, 14.0159216, -29.6775665, 29.6052704
10: -22.3230343, 24.2565002, -22.2771740, 24.2398796, -46.5629120, 46.5336761
11: -26.9238930, 14.1744499, -26.8997974, 14.1411705, -41.0650635, 41.0742493
12: -25.9696426, 13.1702156, -25.9573402, 13.1238041, -38.2962646, 38.3277779
13: -28.1734314, 8.8440676, -28.0824699, 8.8160763, -36.9895096, 36.9265366
14: -49.2669601, 3.5777683, -49.2299080, 3.5627480, -48.2922821, 48.2696533
15: -18.1165581, 10.8180666, -18.0539322, 10.8064976, -28.8143539, 28.7633133
16: -22.6445389, 18.6115379, -22.5351620, 18.5987129, -41.2432518, 41.1466980
17: -44.0500603, 23.8809700, -43.9599419, 23.8514118, -66.7372742, 66.6597366
18: -18.6784801, 9.1238794, -18.6562080, 9.0630131, -27.7414932, 27.7800865
19: -22.8534069, 3.1870351, -22.8258934, 3.1311624, -25.9845695, 26.0129280
20: -14.4609203, 9.2279940, -14.4518986, 9.1550961, -23.6160164, 23.6798935
21: -22.1583309, 9.7955132, -22.1339626, 9.7334528, -31.8917847, 31.9294758
22: -27.1803284, 9.8605566, -27.1624146, 9.7802782, -36.9606056, 37.0229721
23: -20.6877861, 5.7780199, -20.6586075, 5.7163801, -26.4041672, 26.4366264
24: -27.9478951, -0.0219016, -27.9258881, -0.1156135, -27.7889328, 27.8621063
25: -19.8493862, 7.6781454, -19.8351173, 7.6057816, -27.4551678, 27.5132637
26: -33.2106552, 7.2692518, -33.1745567, 7.1712070, -40.3818626, 40.4438095
27: -22.6948795, 9.6369543, -22.6653214, 9.5508289, -31.8954315, 31.9475250
28: -20.6109791, 7.0802517, -20.5914574, 7.0097699, -27.6207485, 27.6717091
29: -33.0442009, 9.0603590, -33.0243835, 8.9916401, -41.4660034, 41.5148163
30: -22.4080105, 8.0896530, -22.4037457, 8.0276775, -30.4356880, 30.4933987
31: -20.4844017, 9.1330595, -20.4509583, 9.0640917, -29.5484924, 29.5840187
32: -19.6814156, 9.7135143, -19.6570015, 9.6486130, -28.2424316, 28.2863960
33: -42.4710159, 5.5381603, -42.4488907, 5.4853964, -45.6752319, 45.7022858
34: -31.4444809, 7.5571833, -31.4145412, 7.4623852, -37.1563263, 37.2199783
35: -31.8714294, 7.7283010, -31.8485641, 7.6655273, -38.7835846, 38.8205566
36: -32.1790733, 7.3881717, -32.1518936, 7.3232489, -38.8356781, 38.8752823
37: -49.7953529, -2.3664904, -49.7504425, -2.4258132, -43.9926758, 44.0096664
38: -40.5365334, 8.3085756, -40.5004120, 8.2040796, -47.6788483, 47.7536087
39: -54.6126060, -1.3852501, -54.5696945, -1.4466562, -52.5702820, 52.5915070
40: -40.1349869, 2.8883681, -40.1071091, 2.8446746, -40.1899338, 40.2076187
41: -25.8236828, 4.1512389, -25.7976494, 4.1135745, -26.3702736, 26.3704147
42: -16.4569397, 8.0472698, -16.4412098, 8.0242662, -22.9954872, 23.0134010

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=57, inp2_unstable=58, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1694

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9186694
time: 29.72 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9186694
time: 33.05 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -27.3696327, 7.9923940, -27.3010063, 7.9840617, -35.1463470, 35.0856361
1: -14.3901291, 13.1280327, -14.3266087, 13.1138268, -27.5039558, 27.4546413
2: -12.0481911, 13.7400732, -12.0032558, 13.7250185, -25.7031212, 25.6710434
3: -8.4884720, 19.2586517, -8.4344406, 19.2426300, -27.7311020, 27.6930923
4: -15.3517189, 12.7518120, -15.2938919, 12.7356110, -28.0873299, 28.0457039
5: -8.4936085, 21.7888145, -8.4359484, 21.7685795, -30.2621880, 30.2247620
6: -22.5644035, 6.0908756, -22.5312958, 6.0844436, -24.2433701, 24.2527237
7: -14.2632170, 20.5802612, -14.1944561, 20.5612488, -34.8244667, 34.7747192
8: -15.2178907, 17.3861160, -15.1519871, 17.3707237, -32.5877495, 32.5360870
9: -16.2459679, 14.0233412, -16.2033405, 14.0196781, -29.6463356, 29.6073227
10: -22.3128834, 24.2600708, -22.2786026, 24.2443180, -46.5572014, 46.5386734
11: -26.9243870, 14.1702290, -26.9023781, 14.1418352, -41.0662231, 41.0726089
12: -25.9809113, 13.1570520, -25.9632950, 13.1245232, -38.3063240, 38.3255997
13: -28.1094456, 8.8252439, -28.0835533, 8.8136387, -36.9230843, 36.9087982
14: -49.2610092, 3.5726871, -49.2330971, 3.5629234, -48.2936020, 48.2622147
15: -18.1013603, 10.8203068, -18.0548954, 10.8100233, -28.8067169, 28.7663612
16: -22.6135216, 18.6194649, -22.5368385, 18.6057472, -41.2192688, 41.1563034
17: -44.0058784, 23.8665848, -43.9644241, 23.8519974, -66.6946716, 66.6501846
18: -18.6634998, 9.0879736, -18.6557770, 9.0638847, -27.7273846, 27.7437515
19: -22.8484802, 3.1724541, -22.8291550, 3.1316385, -25.9801178, 26.0016098
20: -14.4605732, 9.1964550, -14.4574280, 9.1556845, -23.6162567, 23.6538830
21: -22.1507320, 9.7658300, -22.1370068, 9.7341862, -31.8849182, 31.9028358
22: -27.1779747, 9.8311911, -27.1682091, 9.7809477, -36.9589233, 36.9994011
23: -20.6878700, 5.7647285, -20.6628113, 5.7170191, -26.4048882, 26.4275398
24: -27.9402504, -0.0602345, -27.9299374, -0.1151357, -27.7832336, 27.8272934
25: -19.8481197, 7.6571169, -19.8405838, 7.6062684, -27.4543877, 27.4976997
26: -33.2029266, 7.2345486, -33.1774368, 7.1719570, -40.3748856, 40.4119873
27: -22.6763153, 9.5786648, -22.6641693, 9.5514088, -31.8780899, 31.8876190
28: -20.6094589, 7.0627174, -20.5959911, 7.0104246, -27.6198845, 27.6587086
29: -33.0462341, 9.0364876, -33.0302467, 8.9920540, -41.4675827, 41.4984283
30: -22.4102650, 8.0668659, -22.4087811, 8.0283241, -30.4385891, 30.4756470
31: -20.4702759, 9.1061764, -20.4532166, 9.0647335, -29.5350094, 29.5593929
32: -19.6781864, 9.6854696, -19.6600895, 9.6497135, -28.2382584, 28.2725220
33: -42.4705582, 5.5283527, -42.4538040, 5.4862490, -45.6738739, 45.7016144
34: -31.4342632, 7.5131979, -31.4154930, 7.4632096, -37.1493759, 37.1769867
35: -31.8742905, 7.7133579, -31.8534756, 7.6662846, -38.7873764, 38.8140335
36: -32.1881752, 7.3773046, -32.1589737, 7.3240395, -38.8442993, 38.8738632
37: -49.7972107, -2.3813000, -49.7555656, -2.4251513, -43.9953842, 44.0020599
38: -40.5363922, 8.2641029, -40.5072136, 8.2052259, -47.6756134, 47.7200623
39: -54.6106873, -1.3943796, -54.5761223, -1.4460239, -52.5690918, 52.5874023
40: -40.1211395, 2.8654056, -40.1069984, 2.8452048, -40.1741257, 40.1906738
41: -25.8277378, 4.1390786, -25.7988548, 4.1142044, -26.3653908, 26.3654823
42: -16.4653282, 8.0383797, -16.4428654, 8.0249157, -22.9909897, 23.0179253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=57, inp2_unstable=58, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1694

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9185412
time: 34.07 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9185412
time: 29.52 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -27.4489021, 8.0094814, -27.3018074, 7.9859242, -35.2275963, 35.1031342
1: -14.4463921, 13.1443787, -14.3268490, 13.1161394, -27.5625305, 27.4712276
2: -12.0900631, 13.7502871, -12.0032692, 13.7264280, -25.7485847, 25.6801453
3: -8.5408869, 19.2773743, -8.4341278, 19.2451057, -27.7859917, 27.7115021
4: -15.4037685, 12.7609444, -15.2940445, 12.7363644, -28.1401329, 28.0549889
5: -8.5515537, 21.8108578, -8.4356918, 21.7712784, -30.3228321, 30.2465496
6: -22.5701122, 6.1153898, -22.5312157, 6.0844908, -24.2597084, 24.2680626
7: -14.3229113, 20.5992908, -14.1942825, 20.5637741, -34.8866844, 34.7935715
8: -15.2633905, 17.3987999, -15.1519032, 17.3721695, -32.6341248, 32.5475388
9: -16.3125248, 14.0416708, -16.2037086, 14.0221252, -29.7150993, 29.6256104
10: -22.3512840, 24.2724228, -22.2791519, 24.2435627, -46.5948486, 46.5515747
11: -26.9413719, 14.1918974, -26.9028549, 14.1420250, -41.0833969, 41.0947533
12: -25.9903603, 13.1905565, -25.9637833, 13.1246338, -38.3169403, 38.3573685
13: -28.1856766, 8.8573055, -28.0837936, 8.8173246, -37.0030022, 36.9411011
14: -49.2907333, 3.5829601, -49.2333755, 3.5631580, -48.3256226, 48.2730103
15: -18.1534138, 10.8368683, -18.0550499, 10.8113003, -28.8594971, 28.7826271
16: -22.6954441, 18.6437416, -22.5376472, 18.6091766, -41.3046188, 41.1813889
17: -44.0782089, 23.8952827, -43.9641380, 23.8552189, -66.7831573, 66.6734390
18: -18.6881275, 9.1344872, -18.6579037, 9.0639105, -27.7520370, 27.7923908
19: -22.8702545, 3.2111154, -22.8309517, 3.1314857, -26.0017395, 26.0420666
20: -14.4865351, 9.2585993, -14.4605103, 9.1558475, -23.6423836, 23.7191086
21: -22.1789036, 9.8153152, -22.1393452, 9.7341280, -31.9130325, 31.9546604
22: -27.2089024, 9.8924770, -27.1714325, 9.7808533, -36.9897537, 37.0639114
23: -20.7070827, 5.8048744, -20.6647816, 5.7170086, -26.4240913, 26.4696560
24: -27.9702721, 0.0101218, -27.9336739, -0.1152468, -27.8107834, 27.9053192
25: -19.8740387, 7.7114601, -19.8431530, 7.6061864, -27.4802246, 27.5546131
26: -33.2314606, 7.3012996, -33.1800919, 7.1720123, -40.4034729, 40.4813919
27: -22.7032681, 9.6462688, -22.6669350, 9.5514717, -31.9041672, 31.9594650
28: -20.6309052, 7.1128035, -20.5984116, 7.0104351, -27.6413403, 27.7112160
29: -33.0734406, 9.0878725, -33.0327759, 8.9920883, -41.4940491, 41.5508270
30: -22.4305077, 8.1200409, -22.4108772, 8.0285130, -30.4590206, 30.5309181
31: -20.5048447, 9.1616726, -20.4563599, 9.0647469, -29.5695915, 29.6180325
32: -19.7000046, 9.7366705, -19.6620750, 9.6496296, -28.2580948, 28.3272095
33: -42.4890709, 5.5625448, -42.4534683, 5.4862700, -45.6938477, 45.7332382
34: -31.4561195, 7.5791712, -31.4182663, 7.4634047, -37.1697693, 37.2459564
35: -31.8893757, 7.7529058, -31.8543663, 7.6662130, -38.8013763, 38.8511429
36: -32.2022247, 7.4170737, -32.1595383, 7.3239017, -38.8583527, 38.9161301
37: -49.8215103, -2.3500195, -49.7572327, -2.4253340, -44.0198135, 44.0354843
38: -40.5642090, 8.3383636, -40.5089073, 8.2055101, -47.7059402, 47.8026428
39: -54.6407471, -1.3590269, -54.5774879, -1.4462395, -52.5990753, 52.6261749
40: -40.1459961, 2.8984766, -40.1088524, 2.8453584, -40.1991272, 40.2268600
41: -25.8356419, 4.1601958, -25.7989006, 4.1142106, -26.3809700, 26.3840904
42: -16.4721146, 8.0544271, -16.4425621, 8.0249367, -22.9958916, 23.0396538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=57, inp2_unstable=58, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=41, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1694

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9373629
time: 30.24 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9373629
time: 28.81 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -27.3955574, 8.0048676, -27.3458710, 7.9922667, -35.1806946, 35.1410446
1: -14.4131346, 13.1416464, -14.3693056, 13.1286039, -27.5417385, 27.5109520
2: -12.0700903, 13.7494297, -12.0448189, 13.7407532, -25.7418747, 25.7219887
3: -8.5111408, 19.2729950, -8.4788332, 19.2590199, -27.7701607, 27.7518272
4: -15.3746929, 12.7600069, -15.3351679, 12.7522459, -28.1269379, 28.0951748
5: -8.5210571, 21.8052597, -8.4888830, 21.7901573, -30.3112144, 30.2941437
6: -22.5771046, 6.0982194, -22.5559959, 6.0893965, -24.2876511, 24.2963600
7: -14.2934265, 20.5944576, -14.2527027, 20.5806179, -34.8740463, 34.8471603
8: -15.2395487, 17.3956432, -15.1932621, 17.3860931, -32.6248474, 32.5847397
9: -16.2616158, 14.0390081, -16.2305984, 14.0226097, -29.6629143, 29.6451836
10: -22.3252411, 24.2680244, -22.2980652, 24.2551289, -46.5803680, 46.5660896
11: -26.9386330, 14.1821823, -26.9243202, 14.1629906, -41.1016235, 41.1065025
12: -25.9884777, 13.1723385, -25.9797897, 13.1512337, -38.3328133, 38.3574257
13: -28.1275501, 8.8496628, -28.1176491, 8.8212414, -36.9487915, 36.9673119
14: -49.2741814, 3.5782661, -49.2562408, 3.5716295, -48.3248749, 48.2914886
15: -18.1151714, 10.8297205, -18.0785389, 10.8133812, -28.8265266, 28.8005295
16: -22.6466713, 18.6398468, -22.5955105, 18.6174526, -41.2641220, 41.2353592
17: -44.0277328, 23.8868713, -44.0068779, 23.8659191, -66.7454834, 66.7173767
18: -18.6799679, 9.1030140, -18.6632538, 9.0923615, -27.7723293, 27.7662678
19: -22.8618202, 3.1896417, -22.8488960, 3.1651781, -26.0269985, 26.0385380
20: -14.4799795, 9.2081032, -14.4606953, 9.1771688, -23.6571484, 23.6687984
21: -22.1689091, 9.7784262, -22.1494179, 9.7577066, -31.9266167, 31.9278450
22: -27.1992912, 9.8481798, -27.1775284, 9.8136272, -37.0129166, 37.0257072
23: -20.7040920, 5.7876682, -20.6898079, 5.7602429, -26.4643345, 26.4774761
24: -27.9623070, -0.0373397, -27.9410839, -0.0703964, -27.8501663, 27.8729935
25: -19.8648109, 7.6751471, -19.8476620, 7.6409059, -27.5057163, 27.5228081
26: -33.2222672, 7.2656593, -33.2029266, 7.2322993, -40.4545670, 40.4685860
27: -22.6966934, 9.5982227, -22.6778030, 9.5883884, -31.9357910, 31.9366226
28: -20.6249809, 7.0835943, -20.6104317, 7.0505466, -27.6755276, 27.6940269
29: -33.0656395, 9.0520630, -33.0443459, 9.0212641, -41.5158234, 41.5285416
30: -22.4256382, 8.0772839, -22.4087372, 8.0458794, -30.4715176, 30.4860210
31: -20.4916515, 9.1210384, -20.4698410, 9.0925064, -29.5841579, 29.5908794
32: -19.6943855, 9.7013149, -19.6785164, 9.6790028, -28.2722549, 28.3106117
33: -42.4848785, 5.5473328, -42.4694824, 5.5234489, -45.7129211, 45.7364502
34: -31.4540462, 7.5445795, -31.4377022, 7.5223656, -37.2041855, 37.2346878
35: -31.8865089, 7.7376885, -31.8749332, 7.7131271, -38.8307190, 38.8600693
36: -32.2008057, 7.4036870, -32.1894302, 7.3746324, -38.9077148, 38.9320908
37: -49.8147011, -2.3533058, -49.7954178, -2.3706694, -44.0682144, 44.0707397
38: -40.5593719, 8.2986259, -40.5368652, 8.2680483, -47.7629471, 47.7860336
39: -54.6312103, -1.3686609, -54.6100769, -1.3955317, -52.6406860, 52.6471405
40: -40.1375198, 2.8772731, -40.1203766, 2.8673100, -40.2080536, 40.2142563
41: -25.8378983, 4.1581564, -25.8251400, 4.1499338, -26.4020958, 26.4117470
42: -16.4744091, 8.0473194, -16.4601688, 8.0407009, -23.0166283, 23.0448723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=58, inp2_unstable=57, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1694

## Relational analysis of IS_B2_A2_B1_B1

### Relational analysis result of IS_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9358626, upper bound: 16.9373632
time: 31.51 seconds

## Relational analysis of IS_B2_A2_B1_B2

### Relational analysis result of IS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9358626, upper bound: 16.9657159
time: 40.89 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -27.3963203, 8.0066814, -27.4251137, 8.0093403, -35.1981506, 35.2222214
1: -14.4133806, 13.1439457, -14.4255276, 13.1449080, -27.5582886, 27.5694733
2: -12.0700722, 13.7508240, -12.0866547, 13.7509451, -25.7509689, 25.7674332
3: -8.5108280, 19.2754574, -8.5311699, 19.2777309, -27.7885590, 27.8066273
4: -15.3748245, 12.7607765, -15.3872023, 12.7613935, -28.1362190, 28.1479797
5: -8.5207682, 21.8079605, -8.5467901, 21.8121891, -30.3329582, 30.3547516
6: -22.5770187, 6.0983052, -22.5617294, 6.1138973, -24.3029404, 24.3126030
7: -14.2931995, 20.5969887, -14.3123493, 20.5996437, -34.8928452, 34.9093399
8: -15.2394743, 17.3971004, -15.2387304, 17.3987942, -32.6362762, 32.6310501
9: -16.2619858, 14.0414534, -16.2971573, 14.0409737, -29.6812286, 29.7139359
10: -22.3257885, 24.2672997, -22.3364487, 24.2675209, -46.5933075, 46.6037483
11: -26.9391060, 14.1823387, -26.9412899, 14.1846523, -41.1237564, 41.1236267
12: -25.9889736, 13.1724491, -25.9892426, 13.1847515, -38.3645859, 38.3680229
13: -28.1278114, 8.8533363, -28.1939125, 8.8533173, -36.9811287, 37.0472488
14: -49.2744370, 3.5784655, -49.2859650, 3.5819244, -48.3356171, 48.3234711
15: -18.1153297, 10.8309813, -18.1305771, 10.8299503, -28.8427925, 28.8532867
16: -22.6474609, 18.6432686, -22.6774025, 18.6417274, -41.2891884, 41.3206711
17: -44.0274582, 23.8900566, -44.0791702, 23.8946609, -66.7687531, 66.8057404
18: -18.6820984, 9.1030388, -18.6878376, 9.1388483, -27.8209457, 27.7908764
19: -22.8636055, 3.1895080, -22.8707027, 3.2037923, -26.0673981, 26.0602112
20: -14.4830828, 9.2082691, -14.4866810, 9.2392807, -23.7223625, 23.6949501
21: -22.1712513, 9.7783976, -22.1776142, 9.8071823, -31.9784336, 31.9560127
22: -27.2024879, 9.8480835, -27.2084484, 9.8748922, -37.0773811, 37.0565338
23: -20.7060204, 5.7876291, -20.7090302, 5.8003712, -26.5063915, 26.4966583
24: -27.9660110, -0.0374637, -27.9710999, -0.0000858, -27.9281082, 27.9005356
25: -19.8673744, 7.6750484, -19.8735886, 7.6952257, -27.5625992, 27.5486374
26: -33.2249107, 7.2656922, -33.2314453, 7.2989993, -40.5239105, 40.4971390
27: -22.6994591, 9.5982914, -22.7047577, 9.6559811, -32.0076065, 31.9626694
28: -20.6273708, 7.0836000, -20.6318588, 7.1005688, -27.7279396, 27.7154579
29: -33.0681534, 9.0521564, -33.0715942, 9.0726728, -41.5681686, 41.5550232
30: -22.4277420, 8.0774717, -22.4290276, 8.0990505, -30.5267925, 30.5065002
31: -20.4948101, 9.1210346, -20.5044117, 9.1480007, -29.6428108, 29.6254463
32: -19.6964283, 9.7012329, -19.7003365, 9.7302217, -28.3269653, 28.3304291
33: -42.4845581, 5.5473728, -42.4880333, 5.5576124, -45.7445221, 45.7564926
34: -31.4567738, 7.5447817, -31.4595795, 7.5883312, -37.2731171, 37.2551117
35: -31.8874550, 7.7376256, -31.8900375, 7.7526183, -38.8677979, 38.8740616
36: -32.2013626, 7.4035368, -32.2035217, 7.4143310, -38.9498978, 38.9461670
37: -49.8163185, -2.3534994, -49.8196640, -2.3394947, -44.1016159, 44.0951767
38: -40.5610428, 8.2989216, -40.5647240, 8.3422241, -47.8454895, 47.8164368
39: -54.6326103, -1.3688965, -54.6401558, -1.3601694, -52.6794434, 52.6771088
40: -40.1393852, 2.8774395, -40.1452179, 2.9003725, -40.2442245, 40.2392654
41: -25.8379402, 4.1581383, -25.8330421, 4.1710901, -26.4207268, 26.4272461
42: -16.4741268, 8.0473518, -16.4669647, 8.0567646, -23.0383606, 23.0497780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=58, inp2_unstable=57, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1694

## Relational analysis of IS_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9358626, upper bound: 16.9657159
time: 37.81 seconds

## Relational analysis of IS_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9657161, upper bound: 16.9657159
time: 30.11 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 70.29 seconds
IS_B1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9547104, upper bound: 16.8998179
IS_B1_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9547104, upper bound: 16.8998179
IS_B1_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9186694
IS_B1_A1_A2_B2, status: Status.VERIFIED, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9186694
IS_B1_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9185412
IS_B1_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9185412
IS_B1_A2_A2_B1, status: Status.VERIFIED, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9373629
IS_B1_A2_A2_B2, status: Status.VERIFIED, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9547104, upper bound: 16.9373629
IS_B2_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9358626, upper bound: 16.9373632
IS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9358626, upper bound: 16.9657159
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9358626, upper bound: 16.9657159
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 70.29
Output dim: 7, lower bound: -16.9657161, upper bound: 16.9657159

## BFS IS instance: IS_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -27.3949795, 8.0040722, -27.3448792, 7.9907327, -35.1734467, 35.1387558
1: -14.4129524, 13.1405258, -14.3689489, 13.1262712, -27.5392227, 27.5094757
2: -12.0698853, 13.7487392, -12.0444756, 13.7393866, -25.7370300, 25.7207756
3: -8.5108891, 19.2720909, -8.4783831, 19.2573833, -27.7682724, 27.7504730
4: -15.3743382, 12.7591343, -15.3345490, 12.7504206, -28.1247597, 28.0936832
5: -8.5207520, 21.8045406, -8.4883518, 21.7888069, -30.3095589, 30.2928925
6: -22.5767365, 6.0979848, -22.5553265, 6.0889211, -24.2899208, 24.2900810
7: -14.2931519, 20.5936432, -14.2521706, 20.5789261, -34.8720779, 34.8458138
8: -15.2392826, 17.3943825, -15.1927328, 17.3835258, -32.6149254, 32.5828857
9: -16.2613888, 14.0381861, -16.2301407, 14.0209227, -29.6560669, 29.6441994
10: -22.3249397, 24.2673702, -22.2974586, 24.2537918, -46.5787315, 46.5648270
11: -26.9381065, 14.1819859, -26.9232750, 14.1626654, -41.1007729, 41.1052628
12: -25.9873734, 13.1719532, -25.9776497, 13.1505070, -38.3331299, 38.3543739
13: -28.1261654, 8.8491993, -28.1147919, 8.8203754, -36.9465408, 36.9639893
14: -49.2734871, 3.5767155, -49.2549057, 3.5686369, -48.3183975, 48.2891464
15: -18.1148529, 10.8288155, -18.0779152, 10.8115139, -28.8229675, 28.8023376
16: -22.6461639, 18.6387558, -22.5945625, 18.6151505, -41.2613144, 41.2333183
17: -44.0263443, 23.8862572, -44.0043106, 23.8647137, -66.7409210, 66.7185898
18: -18.6794472, 9.1022539, -18.6622257, 9.0907869, -27.7702332, 27.7644806
19: -22.8607864, 3.1895905, -22.8468361, 3.1650677, -26.0258541, 26.0364265
20: -14.4790268, 9.2079153, -14.4589396, 9.1768389, -23.6558647, 23.6668549
21: -22.1681976, 9.7783260, -22.1479797, 9.7574883, -31.9256859, 31.9263058
22: -27.1980934, 9.8480873, -27.1750984, 9.8133955, -37.0114899, 37.0231857
23: -20.7031174, 5.7874713, -20.6877899, 5.7599344, -26.4630508, 26.4752617
24: -27.9613152, -0.0374823, -27.9391289, -0.0706592, -27.8480606, 27.8580627
25: -19.8638020, 7.6750183, -19.8458252, 7.6406450, -27.5044479, 27.5208435
26: -33.2215843, 7.2646685, -33.2015762, 7.2302794, -40.4518623, 40.4662437
27: -22.6960983, 9.5973816, -22.6766739, 9.5871143, -31.9294586, 31.9342346
28: -20.6238403, 7.0833769, -20.6082630, 7.0501709, -27.6740112, 27.6916389
29: -33.0644951, 9.0519543, -33.0420494, 9.0211487, -41.5144501, 41.5195160
30: -22.4247303, 8.0770073, -22.4068565, 8.0454073, -30.4701385, 30.4838638
31: -20.4902611, 9.1209469, -20.4670639, 9.0923309, -29.5825920, 29.5880108
32: -19.6933060, 9.7011013, -19.6762409, 9.6785831, -28.2787170, 28.3049011
33: -42.4835968, 5.5471392, -42.4668007, 5.5230036, -45.7110977, 45.7252045
34: -31.4533615, 7.5444069, -31.4363708, 7.5220618, -37.2031326, 37.2233124
35: -31.8853474, 7.7376099, -31.8724556, 7.7129374, -38.8292694, 38.8476715
36: -32.1994400, 7.4035559, -32.1865845, 7.3744388, -38.9075012, 38.9284973
37: -49.8133163, -2.3534718, -49.7925529, -2.3709855, -44.0664673, 44.0604095
38: -40.5582619, 8.2983646, -40.5348396, 8.2675323, -47.7705994, 47.7796631
39: -54.6294785, -1.3687954, -54.6064186, -1.3958321, -52.6385651, 52.6340179
40: -40.1368866, 2.8771420, -40.1190758, 2.8670597, -40.2070923, 40.2124786
41: -25.8374653, 4.1576414, -25.8243790, 4.1489191, -26.4001083, 26.4093513
42: -16.4741459, 8.0466709, -16.4596634, 8.0393724, -23.0262871, 23.0359840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=58, inp2_unstable=56, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1592

## Relational analysis of IS_B2_A2_B1_B2_A1

### Relational analysis result of IS_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9233554, upper bound: 16.9647520
time: 34.62 seconds

## Relational analysis of IS_B2_A2_B1_B2_A2

### Relational analysis result of IS_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9462677, upper bound: 16.9651152
time: 61.38 seconds

## BFS IS instance: IS_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -27.3906555, 8.0002098, -27.3882942, 7.9934812, -35.1776733, 35.1818314
1: -14.4110756, 13.1247177, -14.3928776, 13.1041317, -27.5152073, 27.5175953
2: -12.0684252, 13.7444239, -12.0716248, 13.7358665, -25.7345810, 25.7471123
3: -8.5092659, 19.2615032, -8.5118322, 19.2477627, -27.7570286, 27.7733345
4: -15.3712454, 12.7458477, -15.3593149, 12.7290306, -28.1002769, 28.1051636
5: -8.5184917, 21.7984371, -8.5273170, 21.7909870, -30.3094788, 30.3257542
6: -22.5726624, 6.0959697, -22.5452747, 6.1041574, -24.2802429, 24.2969360
7: -14.2902155, 20.5835266, -14.2873640, 20.5703850, -34.8605995, 34.8708916
8: -15.2359896, 17.3761826, -15.1965656, 17.3536587, -32.5872383, 32.5678558
9: -16.2592373, 14.0282993, -16.2759972, 14.0103416, -29.6478386, 29.6788063
10: -22.3212242, 24.2555122, -22.3085861, 24.2369308, -46.5581551, 46.5640984
11: -26.9339790, 14.1805315, -26.9241371, 14.1750612, -41.1090393, 41.1046677
12: -25.9747219, 13.1688004, -25.9568062, 13.1668015, -38.3282318, 38.3309517
13: -28.1033382, 8.8492250, -28.1377850, 8.8093891, -36.9127274, 36.9870110
14: -49.2651825, 3.5709620, -49.2549706, 3.5634918, -48.3065720, 48.2793655
15: -18.1107655, 10.8154488, -18.0911655, 10.7934494, -28.8006630, 28.7938652
16: -22.6423588, 18.6239452, -22.6377487, 18.5968647, -41.2392235, 41.2616959
17: -44.0137138, 23.8841343, -44.0406723, 23.8762589, -66.7371979, 66.7599106
18: -18.6772079, 9.0902195, -18.6586838, 9.1074123, -27.7846203, 27.7489033
19: -22.8536968, 3.1885586, -22.8467178, 3.1881833, -26.0418797, 26.0352764
20: -14.4680471, 9.2063885, -14.4532671, 9.2215290, -23.6895752, 23.6596565
21: -22.1625404, 9.7766533, -22.1541176, 9.7982845, -31.9608250, 31.9307709
22: -27.1841412, 9.8466873, -27.1661682, 9.8525887, -37.0367279, 37.0128555
23: -20.6999245, 5.7860041, -20.6936340, 5.7882404, -26.4881649, 26.4796371
24: -27.9533958, -0.0386224, -27.9429169, -0.0177369, -27.8995590, 27.8711243
25: -19.8535805, 7.6736259, -19.8415909, 7.6746216, -27.5282021, 27.5152168
26: -33.2175102, 7.2612681, -33.2115517, 7.2840295, -40.5015411, 40.4728203
27: -22.6941681, 9.5888577, -22.6764946, 9.6334114, -31.9806976, 31.9248962
28: -20.6189384, 7.0814590, -20.6119843, 7.0806913, -27.6996307, 27.6934433
29: -33.0507431, 9.0514326, -33.0303574, 9.0558472, -41.5337982, 41.5128021
30: -22.4134331, 8.0747299, -22.3965187, 8.0727844, -30.4862175, 30.4712486
31: -20.4848595, 9.1196117, -20.4777031, 9.1298027, -29.6146622, 29.5973148
32: -19.6781483, 9.6983299, -19.6592522, 9.7020254, -28.2710571, 28.2861595
33: -42.4625168, 5.5451288, -42.4377289, 5.5220814, -45.6861420, 45.7031708
34: -31.4472332, 7.5424652, -31.4377251, 7.5707531, -37.2456894, 37.2307434
35: -31.8679886, 7.7362032, -31.8469601, 7.7282448, -38.8236389, 38.8294144
36: -32.1792603, 7.4020228, -32.1540337, 7.3896933, -38.9006348, 38.8943481
37: -49.7934647, -2.3544989, -49.7649040, -2.3616424, -44.0553589, 44.0369263
38: -40.5444794, 8.2962227, -40.5245399, 8.3280449, -47.8005371, 47.7703094
39: -54.6026840, -1.3702660, -54.5710220, -1.4012508, -52.6070709, 52.6052399
40: -40.1299019, 2.8759489, -40.1199188, 2.8822594, -40.2153931, 40.2116470
41: -25.8337555, 4.1559348, -25.8195515, 4.1618519, -26.4055023, 26.4121666
42: -16.4704952, 8.0419559, -16.4483776, 8.0423298, -23.0136299, 23.0354958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=58, inp2_unstable=56, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1592

## Relational analysis of IS_B2_A2_B2_B1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9434943, upper bound: 16.9647520
time: 35.21 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9554122, upper bound: 16.9651152
time: 25.90 seconds

## BFS IS instance: IS_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -27.3957806, 8.0058994, -27.4241009, 8.0078363, -35.1908951, 35.2199402
1: -14.4131870, 13.1428261, -14.4251995, 13.1426182, -27.5558052, 27.5680256
2: -12.0698795, 13.7501478, -12.0863085, 13.7496033, -25.7461166, 25.7662315
3: -8.5105686, 19.2745686, -8.5307226, 19.2761059, -27.7866745, 27.8052902
4: -15.3745031, 12.7598858, -15.3865767, 12.7595482, -28.1340523, 28.1464615
5: -8.5204811, 21.8072205, -8.5462255, 21.8108521, -30.3313332, 30.3534470
6: -22.5766525, 6.0980558, -22.5610600, 6.1134052, -24.3052139, 24.3063393
7: -14.2929230, 20.5961723, -14.3118305, 20.5979557, -34.8908768, 34.9080048
8: -15.2391882, 17.3958149, -15.2381926, 17.3962288, -32.6263351, 32.6292572
9: -16.2617416, 14.0406361, -16.2966690, 14.0392742, -29.6743698, 29.7129745
10: -22.3254795, 24.2666397, -22.3358154, 24.2661705, -46.5916519, 46.6024551
11: -26.9385757, 14.1821756, -26.9402542, 14.1843348, -41.1229095, 41.1224289
12: -25.9878788, 13.1720591, -25.9871025, 13.1840162, -38.3648834, 38.3649673
13: -28.1264305, 8.8529043, -28.1910267, 8.8524342, -36.9788666, 37.0439301
14: -49.2737350, 3.5769358, -49.2846107, 3.5788994, -48.3291473, 48.3211212
15: -18.1150169, 10.8300705, -18.1299782, 10.8280754, -28.8392334, 28.8550682
16: -22.6469650, 18.6421700, -22.6764374, 18.6394367, -41.2863998, 41.3186073
17: -44.0260963, 23.8895054, -44.0765839, 23.8933601, -66.7642212, 66.8069458
18: -18.6815529, 9.1022778, -18.6868134, 9.1372719, -27.8188248, 27.7890911
19: -22.8625755, 3.1894393, -22.8686371, 3.2036781, -26.0662537, 26.0580769
20: -14.4821224, 9.2080832, -14.4849072, 9.2389593, -23.7210808, 23.6929893
21: -22.1705189, 9.7782726, -22.1761875, 9.8069410, -31.9774590, 31.9544601
22: -27.2013111, 9.8480072, -27.2060299, 9.8746996, -37.0760117, 37.0540390
23: -20.7050514, 5.7874680, -20.7069969, 5.8000522, -26.5051041, 26.4944649
24: -27.9650230, -0.0376277, -27.9691792, -0.0003624, -27.9260178, 27.8856125
25: -19.8663750, 7.6748981, -19.8717632, 7.6949577, -27.5613327, 27.5466614
26: -33.2242393, 7.2646985, -33.2301331, 7.2969804, -40.5212212, 40.4948311
27: -22.6988640, 9.5974455, -22.7036285, 9.6547203, -32.0012665, 31.9603271
28: -20.6262512, 7.0833850, -20.6297417, 7.1002150, -27.7264671, 27.7131271
29: -33.0670357, 9.0520420, -33.0692863, 9.0725164, -41.5668488, 41.5460358
30: -22.4268417, 8.0771961, -22.4271374, 8.0985804, -30.5254211, 30.5043335
31: -20.4933968, 9.1209421, -20.5016155, 9.1478271, -29.6412239, 29.6225586
32: -19.6953011, 9.7010155, -19.6980762, 9.7298183, -28.3334198, 28.3247223
33: -42.4832840, 5.5471420, -42.4853096, 5.5572300, -45.7426834, 45.7451935
34: -31.4561272, 7.5446048, -31.4582214, 7.5879936, -37.2720871, 37.2436676
35: -31.8862324, 7.7374811, -31.8875389, 7.7524304, -38.8663025, 38.8616333
36: -32.2000084, 7.4034376, -32.2006950, 7.4141045, -38.9496918, 38.9425812
37: -49.8149567, -2.3536639, -49.8168602, -2.3397975, -44.0998459, 44.0848770
38: -40.5599442, 8.2986498, -40.5626450, 8.3417225, -47.8531342, 47.8101196
39: -54.6308212, -1.3690567, -54.6364899, -1.3604908, -52.6772461, 52.6639557
40: -40.1387367, 2.8773122, -40.1439285, 2.9001250, -40.2433167, 40.2374649
41: -25.8375301, 4.1576328, -25.8322983, 4.1700602, -26.4187355, 26.4248772
42: -16.4738731, 8.0467167, -16.4664345, 8.0554085, -23.0480118, 23.0409069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=58, inp2_unstable=56, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1592
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1592

## Relational analysis of IS_B2_A2_B2_B2_A1

### Relational analysis result of IS_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9531983, upper bound: 16.9647520
time: 39.15 seconds

## Relational analysis of IS_B2_A2_B2_B2_A2

### Relational analysis result of IS_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9651152, upper bound: 16.9651152
time: 40.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 81.74 seconds
IS_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 81.74
Output dim: 7, lower bound: -16.9233554, upper bound: 16.9647520
IS_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 81.74
Output dim: 7, lower bound: -16.9462677, upper bound: 16.9651152
IS_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 81.74
Output dim: 7, lower bound: -16.9434943, upper bound: 16.9647520
IS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 81.74
Output dim: 7, lower bound: -16.9554122, upper bound: 16.9651152
IS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 81.74
Output dim: 7, lower bound: -16.9531983, upper bound: 16.9647520
IS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 81.74
Output dim: 7, lower bound: -16.9651152, upper bound: 16.9651152

## BFS IS instance: IS_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -27.3842392, 7.9911523, -27.3397751, 7.9846344, -35.1559219, 35.1199570
1: -14.4053774, 13.1362104, -14.3653307, 13.1242151, -27.5295925, 27.5015411
2: -12.0577831, 13.7449398, -12.0386744, 13.7375975, -25.7190323, 25.7092972
3: -8.4877424, 19.2686672, -8.4674263, 19.2557602, -27.7435036, 27.7360935
4: -15.3573494, 12.7544622, -15.3265038, 12.7481918, -28.1055412, 28.0809669
5: -8.4949579, 21.8008900, -8.4761524, 21.7870541, -30.2820129, 30.2770424
6: -22.5408287, 6.0923653, -22.5383224, 6.0862551, -24.2493210, 24.2645779
7: -14.2670631, 20.5904388, -14.2398129, 20.5774002, -34.8444633, 34.8302536
8: -15.2293825, 17.3896122, -15.1880436, 17.3812618, -32.6018143, 32.5724335
9: -16.2417088, 14.0342989, -16.2206955, 14.0191031, -29.6341705, 29.6308174
10: -22.2989445, 24.2600517, -22.2851391, 24.2503052, -46.5492477, 46.5451889
11: -26.9259949, 14.1602955, -26.9174957, 14.1523781, -41.0783730, 41.0777893
12: -25.9765244, 13.1649199, -25.9724770, 13.1471863, -38.3187256, 38.3419495
13: -28.0886669, 8.8415375, -28.0970173, 8.8167629, -36.9054298, 36.9385529
14: -49.2644882, 3.5440035, -49.2506104, 3.5531349, -48.2934418, 48.2517395
15: -18.1077747, 10.8038597, -18.0745621, 10.7996798, -28.8035202, 28.7743301
16: -22.6109085, 18.6333275, -22.5778580, 18.6125908, -41.2234993, 41.2111855
17: -44.0185013, 23.8579369, -44.0005951, 23.8512955, -66.7185669, 66.6861725
18: -18.6747265, 9.0760765, -18.6599846, 9.0781784, -27.7529049, 27.7360611
19: -22.8547363, 3.1712348, -22.8439713, 3.1563725, -26.0111084, 26.0152054
20: -14.4730501, 9.1943111, -14.4560919, 9.1703405, -23.6433907, 23.6504021
21: -22.1589127, 9.7620277, -22.1435890, 9.7497749, -31.9086876, 31.9056168
22: -27.1912670, 9.8032246, -27.1718693, 9.7921591, -36.9834251, 36.9750938
23: -20.6971817, 5.7602425, -20.6849842, 5.7470264, -26.4442081, 26.4452267
24: -27.9563713, -0.0757236, -27.9367828, -0.0887570, -27.8250275, 27.8174515
25: -19.8586044, 7.6470213, -19.8433571, 7.6274147, -27.4860191, 27.4903793
26: -33.2148247, 7.2193909, -33.1983414, 7.2086730, -40.4234962, 40.4177322
27: -22.6877232, 9.5580063, -22.6727142, 9.5682869, -31.9017944, 31.8891296
28: -20.6200981, 7.0489807, -20.6065254, 7.0338802, -27.6539783, 27.6555061
29: -33.0574379, 9.0023842, -33.0386963, 8.9976044, -41.4841309, 41.4666061
30: -22.4169693, 8.0547485, -22.4031773, 8.0348587, -30.4518280, 30.4579258
31: -20.4824753, 9.1060982, -20.4633713, 9.0852957, -29.5677719, 29.5694695
32: -19.6697006, 9.6964283, -19.6650429, 9.6763697, -28.2522507, 28.2875633
33: -42.4541664, 5.5394735, -42.4526443, 5.5194359, -45.6766052, 45.7027664
34: -31.4399166, 7.5390286, -31.4298782, 7.5194621, -37.1859360, 37.2084351
35: -31.8704071, 7.7336931, -31.8653316, 7.7110519, -38.8117065, 38.8361206
36: -32.1869965, 7.3999290, -32.1806602, 7.3727021, -38.8919678, 38.9170685
37: -49.8010330, -2.3592587, -49.7866440, -2.3736691, -44.0497208, 44.0438080
38: -40.5425949, 8.2944164, -40.5274200, 8.2656898, -47.7525482, 47.7678757
39: -54.5957108, -1.3723822, -54.5903969, -1.3975239, -52.6025085, 52.6142731
40: -40.1131058, 2.8730764, -40.1078415, 2.8651748, -40.1836090, 40.1974182
41: -25.8183270, 4.1520081, -25.8153133, 4.1462383, -26.3771515, 26.3912354
42: -16.4548492, 8.0405426, -16.4504967, 8.0364599, -23.0083351, 23.0181046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=57, inp2_unstable=56, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1592
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 636

## Relational analysis of IS_B2_A2_B1_B2_A1_A1

### Relational analysis result of IS_B2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9634249
time: 36.77 seconds

## Relational analysis of IS_B2_A2_B1_B2_A1_A2

### Relational analysis result of IS_B2_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9384340
time: 26.52 seconds

## BFS IS instance: IS_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -27.3991699, 8.0081320, -27.3435326, 7.9899435, -35.1769485, 35.1411819
1: -14.4144087, 13.1439114, -14.3675718, 13.1258869, -27.5402946, 27.5114822
2: -12.0709572, 13.7538395, -12.0435543, 13.7391396, -25.7334442, 25.7308197
3: -8.5117092, 19.2877979, -8.4770908, 19.2570343, -27.7687435, 27.7648888
4: -15.3750229, 12.7692299, -15.3332424, 12.7501106, -28.1251335, 28.1024723
5: -8.5196085, 21.8278008, -8.4868517, 21.7884750, -30.3080826, 30.3146515
6: -22.5784798, 6.1274834, -22.5534058, 6.0883951, -24.2853508, 24.3179359
7: -14.2949724, 20.6012650, -14.2506542, 20.5785713, -34.8735428, 34.8519211
8: -15.2397404, 17.3975601, -15.1911201, 17.3829403, -32.6144409, 32.5839920
9: -16.2627220, 14.0532751, -16.2289257, 14.0206394, -29.6564140, 29.6581192
10: -22.3263817, 24.2925911, -22.2958870, 24.2532463, -46.5796280, 46.5884781
11: -26.9629478, 14.1816807, -26.9225903, 14.1611767, -41.1241226, 41.1042709
12: -25.9885597, 13.1821604, -25.9764862, 13.1499557, -38.3319092, 38.3630562
13: -28.1260529, 8.8828392, -28.1121712, 8.8197823, -36.9458351, 36.9950104
14: -49.3052063, 3.5761318, -49.2538643, 3.5669403, -48.3499908, 48.2807693
15: -18.1289062, 10.8331108, -18.0771332, 10.8101654, -28.8338318, 28.8050766
16: -22.6537819, 18.6683979, -22.5926266, 18.6146698, -41.2684517, 41.2610245
17: -44.0550156, 23.8886948, -44.0035553, 23.8629837, -66.7649536, 66.7147980
18: -18.7059937, 9.1057587, -18.6616135, 9.0894012, -27.7953949, 27.7673721
19: -22.8813324, 3.1886282, -22.8464470, 3.1639376, -26.0452690, 26.0350761
20: -14.4901371, 9.2082615, -14.4584332, 9.1760025, -23.6661396, 23.6666946
21: -22.1896687, 9.7769928, -22.1474228, 9.7563858, -31.9460545, 31.9244156
22: -27.2390881, 9.8477955, -27.1745186, 9.8110886, -37.0501785, 37.0223160
23: -20.7364635, 5.7868395, -20.6873703, 5.7583523, -26.4948158, 26.4742088
24: -28.0048866, -0.0395999, -27.9387798, -0.0727129, -27.8894882, 27.8513031
25: -19.8889961, 7.6747270, -19.8454514, 7.6390743, -27.5280704, 27.5201778
26: -33.2738914, 7.2664104, -33.2009201, 7.2278247, -40.5017166, 40.4673309
27: -22.7378883, 9.5954952, -22.6760597, 9.5849628, -31.9705276, 31.9300461
28: -20.6645889, 7.0829644, -20.6079769, 7.0482216, -27.7128105, 27.6909409
29: -33.1123848, 9.0497141, -33.0415154, 9.0182495, -41.5595856, 41.5143433
30: -22.4574757, 8.0764637, -22.4064293, 8.0439739, -30.5014496, 30.4828930
31: -20.5078354, 9.1204195, -20.4664612, 9.0913181, -29.5991535, 29.5868797
32: -19.6972275, 9.7150288, -19.6748390, 9.6782455, -28.2780457, 28.3141441
33: -42.4873810, 5.5656157, -42.4650040, 5.5224085, -45.7106247, 45.7433929
34: -31.4579163, 7.5518994, -31.4354172, 7.5216393, -37.2028809, 37.2255402
35: -31.8903618, 7.7439499, -31.8712502, 7.7125945, -38.8315582, 38.8535461
36: -32.2029419, 7.4036164, -32.1855202, 7.3733425, -38.9066849, 38.9264374
37: -49.8181992, -2.3479972, -49.7910728, -2.3713608, -44.0715485, 44.0564804
38: -40.5601120, 8.3133297, -40.5334549, 8.2673025, -47.7691956, 47.7920609
39: -54.6374283, -1.3368797, -54.6044579, -1.3960676, -52.6451111, 52.6643219
40: -40.1396599, 2.8906856, -40.1170502, 2.8667450, -40.2062683, 40.2194443
41: -25.8423462, 4.1706924, -25.8233376, 4.1484523, -26.4008026, 26.4091339
42: -16.4762154, 8.0568466, -16.4586678, 8.0387897, -23.0252342, 23.0322151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=57, inp2_unstable=56, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 636

## Relational analysis of IS_B2_A2_B1_B2_A2_A1

### Relational analysis result of IS_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9637879
time: 32.84 seconds

## Relational analysis of IS_B2_A2_B1_B2_A2_A2

### Relational analysis result of IS_B2_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9367593
time: 34.69 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -27.3799019, 7.9872856, -27.3831978, 7.9873524, -35.1601486, 35.1630287
1: -14.4034853, 13.1203918, -14.3892689, 13.1020746, -27.5055599, 27.5096607
2: -12.0563183, 13.7406254, -12.0658150, 13.7340527, -25.7166290, 25.7355957
3: -8.4860916, 19.2580814, -8.5008755, 19.2461433, -27.7322350, 27.7589569
4: -15.3542767, 12.7411938, -15.3512726, 12.7267990, -28.0810757, 28.0924664
5: -8.4926777, 21.7947922, -8.5150776, 21.7892647, -30.2819424, 30.3098698
6: -22.5367432, 6.0903463, -22.5282631, 6.1014729, -24.2396393, 24.2714462
7: -14.2641401, 20.5802784, -14.2750092, 20.5688667, -34.8330078, 34.8552856
8: -15.2260847, 17.3714218, -15.1918612, 17.3513966, -32.5741425, 32.5574112
9: -16.2395611, 14.0244064, -16.2665558, 14.0085011, -29.6259232, 29.6653938
10: -22.2952805, 24.2481709, -22.2962494, 24.2334366, -46.5287170, 46.5444183
11: -26.9218712, 14.1588421, -26.9183731, 14.1647902, -41.0866623, 41.0772171
12: -25.9638672, 13.1617622, -25.9516296, 13.1634636, -38.3138504, 38.3185043
13: -28.0658493, 8.8415813, -28.1199894, 8.8057766, -36.8716278, 36.9615707
14: -49.2561684, 3.5382671, -49.2506943, 3.5479679, -48.2816162, 48.2419891
15: -18.1036644, 10.7904911, -18.0877914, 10.7816315, -28.7812119, 28.7658310
16: -22.6070862, 18.6185131, -22.6210480, 18.5942841, -41.2013702, 41.2395630
17: -44.0058899, 23.8558311, -44.0369873, 23.8629093, -66.7148132, 66.7275543
18: -18.6724606, 9.0640621, -18.6564560, 9.0948153, -27.7672768, 27.7205181
19: -22.8476448, 3.1701810, -22.8438454, 3.1794639, -26.0271091, 26.0140266
20: -14.4620581, 9.1927929, -14.4504175, 9.2150431, -23.6771011, 23.6432114
21: -22.1532593, 9.7603865, -22.1497307, 9.7905769, -31.9438362, 31.9101181
22: -27.1773376, 9.8018169, -27.1629543, 9.8313494, -37.0086861, 36.9647713
23: -20.6940193, 5.7587614, -20.6908035, 5.7753181, -26.4693375, 26.4495659
24: -27.9484482, -0.0768209, -27.9405785, -0.0358205, -27.8765030, 27.8305054
25: -19.8483639, 7.6456318, -19.8390999, 7.6613584, -27.5097218, 27.4847317
26: -33.2107506, 7.2159944, -33.2083435, 7.2624497, -40.4732018, 40.4243393
27: -22.6857872, 9.5495090, -22.6725121, 9.6145611, -31.9530640, 31.8797836
28: -20.6151886, 7.0470715, -20.6101856, 7.0644045, -27.6795921, 27.6572571
29: -33.0436821, 9.0017929, -33.0269852, 9.0323582, -41.5034561, 41.4598999
30: -22.4056568, 8.0524836, -22.3928032, 8.0622425, -30.4678993, 30.4452858
31: -20.4770622, 9.1047268, -20.4740067, 9.1227779, -29.5998402, 29.5787334
32: -19.6545334, 9.6936541, -19.6481075, 9.6998043, -28.2445526, 28.2687836
33: -42.4330597, 5.5374956, -42.4235687, 5.5184622, -45.6516190, 45.6806870
34: -31.4337559, 7.5370860, -31.4312134, 7.5681953, -37.2285156, 37.2159195
35: -31.8530293, 7.7322736, -31.8398342, 7.7263985, -38.8061218, 38.8179092
36: -32.1668015, 7.3984213, -32.1480942, 7.3879704, -38.8851089, 38.8829803
37: -49.7811356, -2.3602872, -49.7590103, -2.3643460, -44.0386353, 44.0203400
38: -40.5287933, 8.2923355, -40.5170326, 8.3261156, -47.7825623, 47.7584381
39: -54.5689583, -1.3737946, -54.5550079, -1.4029455, -52.5710449, 52.5854034
40: -40.1061440, 2.8718715, -40.1086426, 2.8803444, -40.1919098, 40.1965790
41: -25.8145962, 4.1503010, -25.8104572, 4.1591749, -26.3825378, 26.3940086
42: -16.4511833, 8.0358162, -16.4392147, 8.0393906, -22.9956818, 23.0176296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=57, inp2_unstable=56, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 636

## Relational analysis of IS_B2_A2_B2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9634249
time: 49.35 seconds

## Relational analysis of IS_B2_A2_B2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9384340
time: 31.01 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -27.3947983, 8.0042629, -27.3869247, 7.9926882, -35.1811676, 35.1842422
1: -14.4125500, 13.1280756, -14.3915014, 13.1037416, -27.5162926, 27.5195770
2: -12.0695105, 13.7495241, -12.0707188, 13.7356052, -25.7310104, 25.7571526
3: -8.5100670, 19.2772141, -8.5105381, 19.2473965, -27.7574635, 27.7877522
4: -15.3719368, 12.7559528, -15.3579941, 12.7287216, -28.1006584, 28.1139469
5: -8.5173244, 21.8216991, -8.5258198, 21.7906818, -30.3080063, 30.3475189
6: -22.5743904, 6.1254797, -22.5433769, 6.1036177, -24.2756615, 24.3247948
7: -14.2920532, 20.5911617, -14.2858381, 20.5700455, -34.8620987, 34.8769989
8: -15.2364273, 17.3793831, -15.1949511, 17.3530788, -32.5867996, 32.5689468
9: -16.2605858, 14.0433693, -16.2747841, 14.0100307, -29.6481781, 29.6927109
10: -22.3227139, 24.2807159, -22.3070221, 24.2364197, -46.5591354, 46.5877380
11: -26.9587955, 14.1802235, -26.9234524, 14.1736078, -41.1324043, 41.1036758
12: -25.9759045, 13.1789846, -25.9556389, 13.1662407, -38.3270149, 38.3396187
13: -28.1032448, 8.8828611, -28.1351452, 8.8088398, -36.9120865, 37.0180054
14: -49.2968941, 3.5703602, -49.2539406, 3.5617523, -48.3381653, 48.2710342
15: -18.1248093, 10.8197651, -18.0903778, 10.7920876, -28.8115234, 28.7966270
16: -22.6499615, 18.6535797, -22.6358337, 18.5963593, -41.2463226, 41.2894135
17: -44.0423737, 23.8865776, -44.0399666, 23.8745880, -66.7612000, 66.7561493
18: -18.7037582, 9.0937405, -18.6580830, 9.1060181, -27.8097763, 27.7518234
19: -22.8742256, 3.1875913, -22.8463287, 3.1870713, -26.0612965, 26.0339203
20: -14.4791269, 9.2067385, -14.4527626, 9.2207136, -23.6998405, 23.6595001
21: -22.1840153, 9.7753353, -22.1535435, 9.7971907, -31.9812050, 31.9288788
22: -27.2251511, 9.8464127, -27.1656036, 9.8502579, -37.0754089, 37.0120163
23: -20.7332993, 5.7853518, -20.6932201, 5.7866459, -26.5199451, 26.4785728
24: -27.9969864, -0.0407176, -27.9425735, -0.0198131, -27.9409790, 27.8643951
25: -19.8787746, 7.6733475, -19.8412323, 7.6730494, -27.5518246, 27.5145798
26: -33.2697983, 7.2630177, -33.2108879, 7.2815666, -40.5513649, 40.4739075
27: -22.7359390, 9.5869560, -22.6758881, 9.6312313, -32.0217896, 31.9206772
28: -20.6596680, 7.0810757, -20.6116428, 7.0787411, -27.7384090, 27.6927185
29: -33.0986366, 9.0491333, -33.0297966, 9.0529385, -41.5789032, 41.5075607
30: -22.4461975, 8.0741568, -22.3960381, 8.0713634, -30.5175610, 30.4701958
31: -20.5024185, 9.1190777, -20.4771194, 9.1288109, -29.6312294, 29.5961971
32: -19.6820545, 9.7122374, -19.6578770, 9.7017107, -28.2703247, 28.2953987
33: -42.4662933, 5.5636196, -42.4359131, 5.5214672, -45.6856384, 45.7213364
34: -31.4517727, 7.5499339, -31.4367676, 7.5703621, -37.2454605, 37.2330170
35: -31.8729935, 7.7425523, -31.8457413, 7.7278695, -38.8259583, 38.8353653
36: -32.1828117, 7.4020338, -32.1529503, 7.3885770, -38.8998718, 38.8922882
37: -49.7983284, -2.3490696, -49.7634201, -2.3620467, -44.0604706, 44.0330048
38: -40.5463028, 8.3112345, -40.5230713, 8.3278093, -47.7991638, 47.7825928
39: -54.6107140, -1.3383102, -54.5690536, -1.4014788, -52.6136475, 52.6354828
40: -40.1326981, 2.8895006, -40.1178474, 2.8819437, -40.2145309, 40.2186356
41: -25.8386345, 4.1690116, -25.8185081, 4.1613965, -26.4061852, 26.4119225
42: -16.4725361, 8.0521240, -16.4473953, 8.0417175, -23.0125656, 23.0317364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=57, inp2_unstable=56, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 636
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 719
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1777
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 636

## Relational analysis of IS_B2_A2_B2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9270892, upper bound: 16.9637879
time: 36.63 seconds

## Relational analysis of IS_B2_A2_B2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.9270892, upper bound: 16.9387961
time: 34.05 seconds

## BFS IS instance: IS_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -27.3850117, 7.9929872, -27.4190140, 8.0016861, -35.1733704, 35.2011604
1: -14.4056234, 13.1385136, -14.4215813, 13.1405497, -27.5461731, 27.5600948
2: -12.0577431, 13.7463579, -12.0805025, 13.7477751, -25.7281265, 25.7547073
3: -8.4874086, 19.2711544, -8.5197582, 19.2744713, -27.7618790, 27.7909126
4: -15.3575115, 12.7552128, -15.3785191, 12.7573423, -28.1148529, 28.1337318
5: -8.4946613, 21.8035736, -8.5340080, 21.8091011, -30.3037624, 30.3375816
6: -22.5407410, 6.0924349, -22.5440407, 6.1107421, -24.2646217, 24.2808342
7: -14.2668495, 20.5929451, -14.2994671, 20.5964317, -34.8632812, 34.8924103
8: -15.2292852, 17.3910656, -15.2335281, 17.3939762, -32.6132660, 32.6187553
9: -16.2420826, 14.0367355, -16.2872200, 14.0374374, -29.6524887, 29.6995659
10: -22.2994766, 24.2593346, -22.3234978, 24.2627068, -46.5621834, 46.5828323
11: -26.9264641, 14.1604576, -26.9344921, 14.1740532, -41.1005173, 41.0949478
12: -25.9770126, 13.1650352, -25.9819355, 13.1806803, -38.3504944, 38.3525352
13: -28.0888977, 8.8452454, -28.1732693, 8.8488483, -36.9377441, 37.0185165
14: -49.2647552, 3.5442429, -49.2803574, 3.5634174, -48.3041992, 48.2837372
15: -18.1079006, 10.8051329, -18.1265869, 10.8162651, -28.8197708, 28.8270798
16: -22.6116848, 18.6367569, -22.6597424, 18.6368637, -41.2485504, 41.2965012
17: -44.0182343, 23.8611870, -44.0728760, 23.8799629, -66.7418518, 66.7745819
18: -18.6768112, 9.0760927, -18.6845741, 9.1246634, -27.8014755, 27.7606659
19: -22.8565292, 3.1710761, -22.8657608, 3.1949723, -26.0515022, 26.0368366
20: -14.4761276, 9.1944542, -14.4820833, 9.2324886, -23.7086163, 23.6765366
21: -22.1612530, 9.7619972, -22.1717892, 9.7992420, -31.9604950, 31.9337864
22: -27.1944981, 9.8031197, -27.2028027, 9.8534613, -37.0479584, 37.0059204
23: -20.6991215, 5.7602205, -20.7042122, 5.7871456, -26.4862671, 26.4644318
24: -27.9600945, -0.0758381, -27.9668274, -0.0184669, -27.9029999, 27.8450241
25: -19.8611488, 7.6468863, -19.8692951, 7.6817050, -27.5428543, 27.5161819
26: -33.2174568, 7.2194424, -33.2269173, 7.2753940, -40.4928513, 40.4463577
27: -22.6904774, 9.5580730, -22.6996574, 9.6358452, -31.9736328, 31.9152298
28: -20.6225033, 7.0489964, -20.6279716, 7.0839100, -27.7064133, 27.6769676
29: -33.0599594, 9.0024147, -33.0659485, 9.0490150, -41.5365143, 41.4931030
30: -22.4190483, 8.0549364, -22.4234657, 8.0880442, -30.5070915, 30.4784012
31: -20.4856224, 9.1060991, -20.4979439, 9.1407862, -29.6264076, 29.6040421
32: -19.6716824, 9.6963081, -19.6869087, 9.7275686, -28.3069305, 28.3073502
33: -42.4538536, 5.5395336, -42.4711914, 5.5535955, -45.7081909, 45.7227249
34: -31.4426880, 7.5391989, -31.4517307, 7.5854340, -37.2548752, 37.2288818
35: -31.8713322, 7.7336092, -31.8804245, 7.7505674, -38.8487396, 38.8500977
36: -32.1875343, 7.3998280, -32.1947746, 7.4124236, -38.9341583, 38.9311829
37: -49.8026733, -2.3594227, -49.8109665, -2.3424692, -44.0831146, 44.0682831
38: -40.5442657, 8.2947292, -40.5551872, 8.3398666, -47.8350754, 47.7981949
39: -54.5970764, -1.3726301, -54.6204185, -1.3622150, -52.6412659, 52.6442261
40: -40.1149597, 2.8732228, -40.1326752, 2.8982263, -40.2198639, 40.2223969
41: -25.8183670, 4.1519780, -25.8232002, 4.1673722, -26.3957825, 26.4067459
42: -16.4545555, 8.0405636, -16.4573097, 8.0525093, -23.0300751, 23.0230064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=57, inp2_unstable=56, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 636

## Relational analysis of IS_B2_A2_B2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9634245
time: 38.20 seconds

## Relational analysis of IS_B2_A2_B2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9384340
time: 40.76 seconds

## BFS IS instance: IS_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -27.3999557, 8.0099640, -27.4227505, 8.0070066, -35.1944427, 35.2223969
1: -14.4146671, 13.1462240, -14.4238071, 13.1422358, -27.5569038, 27.5700302
2: -12.0709476, 13.7552462, -12.0853748, 13.7493258, -25.7425537, 25.7762566
3: -8.5113792, 19.2902660, -8.5294304, 19.2757225, -27.7871017, 27.8196964
4: -15.3751726, 12.7699842, -15.3852673, 12.7592468, -28.1344185, 28.1552505
5: -8.5193253, 21.8305054, -8.5446997, 21.8105164, -30.3298416, 30.3752060
6: -22.5783882, 6.1275706, -22.5591660, 6.1129060, -24.3006439, 24.3341885
7: -14.2947636, 20.6037846, -14.3103008, 20.5975952, -34.8923569, 34.9140854
8: -15.2396336, 17.3990269, -15.2366085, 17.3956470, -32.6258659, 32.6303406
9: -16.2630844, 14.0557280, -16.2954693, 14.0390043, -29.6747284, 29.7268982
10: -22.3269539, 24.2918549, -22.3342323, 24.2656536, -46.5926056, 46.6260872
11: -26.9634056, 14.1818542, -26.9395847, 14.1828671, -41.1462708, 41.1214371
12: -25.9890137, 13.1822615, -25.9859295, 13.1834641, -38.3636665, 38.3736115
13: -28.1263161, 8.8865509, -28.1883869, 8.8518705, -36.9781876, 37.0749359
14: -49.3054428, 3.5763760, -49.2835693, 3.5771961, -48.3607407, 48.3127594
15: -18.1290474, 10.8343821, -18.1291885, 10.8267412, -28.8501053, 28.8578415
16: -22.6545677, 18.6718082, -22.6745338, 18.6389523, -41.2935181, 41.3463440
17: -44.0547066, 23.8918991, -44.0758629, 23.8917656, -66.7882843, 66.8031845
18: -18.7080917, 9.1057625, -18.6862030, 9.1358833, -27.8439751, 27.7919655
19: -22.8831177, 3.1884773, -22.8682480, 3.2025752, -26.0856934, 26.0567245
20: -14.4932213, 9.2084513, -14.4844189, 9.2381554, -23.7313766, 23.6928711
21: -22.1919842, 9.7769384, -22.1756058, 9.8058491, -31.9978333, 31.9525452
22: -27.2422791, 9.8477192, -27.2054672, 9.8723526, -37.1146317, 37.0531845
23: -20.7384033, 5.7867966, -20.7066002, 5.7984638, -26.5368671, 26.4933968
24: -28.0086002, -0.0397253, -27.9688129, -0.0024176, -27.9674759, 27.8788834
25: -19.8915596, 7.6746140, -19.8713799, 7.6933904, -27.5849495, 27.5459938
26: -33.2765236, 7.2664366, -33.2294655, 7.2945094, -40.5710335, 40.4959030
27: -22.7406464, 9.5955696, -22.7030029, 9.6525402, -32.0423508, 31.9561386
28: -20.6669769, 7.0829802, -20.6294231, 7.0982685, -27.7652454, 27.7124023
29: -33.1149063, 9.0497608, -33.0687447, 9.0696287, -41.6119232, 41.5407791
30: -22.4595718, 8.0766649, -22.4267063, 8.0971546, -30.5567265, 30.5033722
31: -20.5109653, 9.1204205, -20.5010338, 9.1468163, -29.6577816, 29.6214542
32: -19.6992245, 9.7149334, -19.6966991, 9.7294750, -28.3327255, 28.3339806
33: -42.4870453, 5.5656118, -42.4835434, 5.5565615, -45.7421722, 45.7633591
34: -31.4607010, 7.5520611, -31.4572830, 7.5875912, -37.2718430, 37.2459412
35: -31.8912811, 7.7438579, -31.8863544, 7.7520723, -38.8686523, 38.8675766
36: -32.2035217, 7.4034667, -32.1996002, 7.4130416, -38.9488678, 38.9405060
37: -49.8198280, -2.3481774, -49.8153572, -2.3401856, -44.1049576, 44.0809097
38: -40.5618057, 8.3136368, -40.5612450, 8.3414764, -47.8516998, 47.8224182
39: -54.6388817, -1.3371544, -54.6344566, -1.3607569, -52.6838531, 52.6942444
40: -40.1415558, 2.8908253, -40.1419067, 2.8998494, -40.2424698, 40.2444611
41: -25.8423939, 4.1706967, -25.8312416, 4.1695971, -26.4194107, 26.4246597
42: -16.4759331, 8.0568895, -16.4654655, 8.0548201, -23.0469780, 23.0371361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=57, inp2_unstable=56, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1373

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 636

## Relational analysis of IS_B2_A2_B2_B2_A2_A1

### Relational analysis result of IS_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9637879
time: 39.14 seconds

## Relational analysis of IS_B2_A2_B2_B2_A2_A2

### Relational analysis result of IS_B2_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9363911
time: 35.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 77.59 seconds
IS_B2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9634249
IS_B2_A2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9384340
IS_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9637879
IS_B2_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9367593
IS_B2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9634249
IS_B2_A2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9384340
IS_B2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.9270892, upper bound: 16.9637879
IS_B2_A2_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.9270892, upper bound: 16.9387961
IS_B2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9634245
IS_B2_A2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9384340
IS_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9637879
IS_B2_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 77.59
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9363911

## BFS IS instance: IS_B2_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -27.3333359, 7.9737439, -27.3397751, 7.9846344, -35.1047668, 35.1046143
1: -14.3598108, 13.1102858, -14.3653307, 13.1242151, -27.4840260, 27.4756165
2: -12.0139370, 13.7221594, -12.0386744, 13.7375975, -25.6741333, 25.6860886
3: -8.4415922, 19.2403393, -8.4674263, 19.2557602, -27.6973534, 27.7077656
4: -15.3131618, 12.7321167, -15.3265038, 12.7481918, -28.0613537, 28.0586205
5: -8.4398870, 21.7662392, -8.4761524, 21.7870541, -30.2269402, 30.2423916
6: -22.5109863, 6.0854931, -22.5383224, 6.0862551, -24.2029686, 24.2444439
7: -14.2064466, 20.5590172, -14.2398129, 20.5774002, -34.7838478, 34.7988281
8: -15.1856327, 17.3670940, -15.1880436, 17.3812618, -32.5574646, 32.5529633
9: -16.2115269, 14.0189123, -16.2206955, 14.0191031, -29.6065674, 29.6175003
10: -22.2753906, 24.2430668, -22.2851391, 24.2503052, -46.5256958, 46.5282059
11: -26.8988590, 14.1352415, -26.9174957, 14.1523781, -41.0512390, 41.0527382
12: -25.9558449, 13.1349850, -25.9724770, 13.1471863, -38.3042831, 38.3125305
13: -28.0520535, 8.8153772, -28.0970173, 8.8167629, -36.8688164, 36.9123955
14: -49.2375221, 3.5327263, -49.2506104, 3.5531349, -48.2592926, 48.2383957
15: -18.0814190, 10.7908545, -18.0745621, 10.7996798, -28.7772369, 28.7592125
16: -22.5456161, 18.6051579, -22.5778580, 18.6125908, -41.1582069, 41.1830139
17: -43.9726486, 23.8281803, -44.0005951, 23.8512955, -66.6581726, 66.6518860
18: -18.6555920, 9.0452843, -18.6599846, 9.0781784, -27.7337704, 27.7052689
19: -22.8261604, 3.1364920, -22.8439713, 3.1563725, -25.9825325, 25.9804630
20: -14.4543552, 9.1704683, -14.4560919, 9.1703405, -23.6246948, 23.6265602
21: -22.1341057, 9.7357407, -22.1435890, 9.7497749, -31.8838806, 31.8793297
22: -27.1660900, 9.7663155, -27.1718693, 9.7921591, -36.9582481, 36.9381866
23: -20.6592846, 5.7148342, -20.6849842, 5.7470264, -26.4063110, 26.3998184
24: -27.9266224, -0.1223679, -27.9367828, -0.0887570, -27.7925339, 27.7612228
25: -19.8386955, 7.6103582, -19.8433571, 7.6274147, -27.4661102, 27.4537163
26: -33.1751595, 7.1555667, -33.1983414, 7.2086730, -40.3838310, 40.3539085
27: -22.6592598, 9.5188694, -22.6727142, 9.5682869, -31.8694763, 31.8374405
28: -20.5934334, 7.0066319, -20.6065254, 7.0338802, -27.6273136, 27.6131573
29: -33.0302315, 8.9685555, -33.0386963, 8.9976044, -41.4583130, 41.4318466
30: -22.4043922, 8.0339050, -22.4031773, 8.0348587, -30.4392509, 30.4370823
31: -20.4507065, 9.0766821, -20.4633713, 9.0852957, -29.5360031, 29.5400543
32: -19.6408386, 9.6653004, -19.6650429, 9.6763697, -28.2323380, 28.2555504
33: -42.4295692, 5.5005302, -42.4526443, 5.5194359, -45.6640701, 45.6639252
34: -31.4034424, 7.4770594, -31.4298782, 7.5194621, -37.1703415, 37.1456909
35: -31.8404999, 7.6851225, -31.8653316, 7.7110519, -38.7969208, 38.7878418
36: -32.1490097, 7.3475876, -32.1806602, 7.3727021, -38.8531189, 38.8638916
37: -49.7507172, -2.4158134, -49.7866440, -2.3736691, -44.0004120, 43.9853363
38: -40.4968147, 8.2270927, -40.5274200, 8.2656898, -47.7044220, 47.6993484
39: -54.5487900, -1.4246044, -54.5903969, -1.3975239, -52.5554810, 52.5617828
40: -40.0889206, 2.8493838, -40.1078415, 2.8651748, -40.1613312, 40.1758194
41: -25.7878685, 4.1144023, -25.8153133, 4.1462383, -26.3538055, 26.3540230
42: -16.4347363, 8.0225925, -16.4504967, 8.0364599, -22.9870300, 22.9994316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=56, inp2_unstable=56, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 584
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1783

## Relational analysis of IS_B2_A2_B1_B2_A1_A1_A1

### Relational analysis result of IS_B2_A2_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.8907247, upper bound: 16.9510769
time: 31.02 seconds

## Relational analysis of IS_B2_A2_B1_B2_A1_A1_A2

### Relational analysis result of IS_B2_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.8907247, upper bound: 16.9629850
time: 56.48 seconds

## BFS IS instance: IS_B2_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -27.3482876, 7.9907107, -27.3435326, 7.9899435, -35.1258011, 35.1258392
1: -14.3688526, 13.1179848, -14.3675718, 13.1258869, -27.4947395, 27.4855576
2: -12.0271215, 13.7310772, -12.0435543, 13.7391396, -25.6885414, 25.7076149
3: -8.4655638, 19.2594662, -8.4770908, 19.2570343, -27.7225990, 27.7365570
4: -15.3308334, 12.7468681, -15.3332424, 12.7501106, -28.0809441, 28.0801105
5: -8.4645376, 21.7931023, -8.4868517, 21.7884750, -30.2530136, 30.2799530
6: -22.5486259, 6.1206207, -22.5534058, 6.0883951, -24.2389755, 24.2978077
7: -14.2343235, 20.5698738, -14.2506542, 20.5785713, -34.8128967, 34.8205261
8: -15.1959887, 17.3750553, -15.1911201, 17.3829403, -32.5700836, 32.5645218
9: -16.2325306, 14.0378904, -16.2289257, 14.0206394, -29.6287956, 29.6448021
10: -22.3028164, 24.2756004, -22.2958870, 24.2532463, -46.5560608, 46.5714874
11: -26.9357986, 14.1566057, -26.9225903, 14.1611767, -41.0969772, 41.0791969
12: -25.9678516, 13.1522160, -25.9764862, 13.1499557, -38.3174934, 38.3336372
13: -28.0894718, 8.8566895, -28.1121712, 8.8197823, -36.9092560, 36.9688606
14: -49.2782249, 3.5648460, -49.2538643, 3.5669403, -48.3157959, 48.2674255
15: -18.1024971, 10.8201122, -18.0771332, 10.8101654, -28.8075790, 28.7899742
16: -22.5884781, 18.6402397, -22.5926266, 18.6146698, -41.2031479, 41.2328644
17: -44.0091820, 23.8588963, -44.0035553, 23.8629837, -66.7045135, 66.6805344
18: -18.6868668, 9.0749636, -18.6616135, 9.0894012, -27.7762680, 27.7365761
19: -22.8527641, 3.1538894, -22.8464470, 3.1639376, -26.0167007, 26.0003357
20: -14.4714289, 9.1844254, -14.4584332, 9.1760025, -23.6474304, 23.6428585
21: -22.1648502, 9.7507038, -22.1474228, 9.7563858, -31.9212360, 31.8981266
22: -27.2138824, 9.8108854, -27.1745186, 9.8110886, -37.0249710, 36.9854050
23: -20.6985989, 5.7414303, -20.6873703, 5.7583523, -26.4569511, 26.4288006
24: -27.9751511, -0.0862432, -27.9387798, -0.0727129, -27.8570175, 27.7950821
25: -19.8690872, 7.6380415, -19.8454514, 7.6390743, -27.5081615, 27.4834938
26: -33.2342339, 7.2026234, -33.2009201, 7.2278247, -40.4620590, 40.4035416
27: -22.7094116, 9.5563517, -22.6760597, 9.5849628, -31.9382324, 31.8783569
28: -20.6379070, 7.0406055, -20.6079769, 7.0482216, -27.6861286, 27.6485825
29: -33.0852280, 9.0159035, -33.0415154, 9.0182495, -41.5337601, 41.4795761
30: -22.4449177, 8.0555878, -22.4064293, 8.0439739, -30.4888916, 30.4620171
31: -20.4760761, 9.0909977, -20.4664612, 9.0913181, -29.5673943, 29.5574589
32: -19.6683559, 9.6839085, -19.6748390, 9.6782455, -28.2581177, 28.2821465
33: -42.4627838, 5.5266676, -42.4650040, 5.5224085, -45.6980591, 45.7045593
34: -31.4214249, 7.4898982, -31.4354172, 7.5216393, -37.1873398, 37.1627960
35: -31.8604336, 7.6953650, -31.8712502, 7.7125945, -38.8167419, 38.8052673
36: -32.1649895, 7.3512416, -32.1855202, 7.3733425, -38.8678360, 38.8732376
37: -49.7678680, -2.4045987, -49.7910728, -2.3713608, -44.0222473, 43.9979782
38: -40.5142822, 8.2459755, -40.5334549, 8.2673025, -47.7210999, 47.7235336
39: -54.5905228, -1.3891048, -54.6044579, -1.3960676, -52.5981140, 52.6118317
40: -40.1154709, 2.8670049, -40.1170502, 2.8667450, -40.1839294, 40.1978607
41: -25.8118649, 4.1330843, -25.8233376, 4.1484523, -26.3774300, 26.3719292
42: -16.4560852, 8.0389280, -16.4586678, 8.0387897, -23.0039101, 23.0135536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=56, inp2_unstable=56, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=246, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1336
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1711
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 690
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 584
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 673
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1783

## Relational analysis of IS_B2_A2_B1_B2_A2_A1_A1

### Relational analysis result of IS_B2_A2_B1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.8907247, upper bound: 16.9514352
time: 43.93 seconds

## Relational analysis of IS_B2_A2_B1_B2_A2_A1_A2

### Relational analysis result of IS_B2_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9175061, upper bound: 16.9633503
time: 39.08 seconds

## BFS IS instance: IS_B2_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -27.3289814, 7.9698849, -27.3831978, 7.9873524, -35.1089935, 35.1476974
1: -14.3579416, 13.0944500, -14.3892689, 13.1020746, -27.4600163, 27.4837189
2: -12.0124941, 13.7178555, -12.0658150, 13.7340527, -25.6717453, 25.7123985
3: -8.4399605, 19.2297668, -8.5008755, 19.2461433, -27.6861038, 27.7306423
4: -15.3100796, 12.7188435, -15.3512726, 12.7267990, -28.0368786, 28.0701160
5: -8.4376326, 21.7601070, -8.5150776, 21.7892647, -30.2268982, 30.2751846
6: -22.5068836, 6.0835075, -22.5282631, 6.1014729, -24.1932907, 24.2513065
7: -14.2035017, 20.5488853, -14.2750092, 20.5688667, -34.7723694, 34.8238945
8: -15.1823740, 17.3488960, -15.1918612, 17.3513966, -32.5297928, 32.5379028
9: -16.2093773, 14.0089960, -16.2665558, 14.0085011, -29.5983200, 29.6520805
10: -22.2717171, 24.2311916, -22.2962494, 24.2334366, -46.5051537, 46.5274429
11: -26.8947353, 14.1337471, -26.9183731, 14.1647902, -41.0595245, 41.0521202
12: -25.9431820, 13.1318150, -25.9516296, 13.1634636, -38.2994003, 38.2891083
13: -28.0292187, 8.8154345, -28.1199894, 8.8057766, -36.8349953, 36.9354248
14: -49.2292099, 3.5270061, -49.2506943, 3.5479679, -48.2474365, 48.2286377
15: -18.0773201, 10.7774858, -18.0877914, 10.7816315, -28.7549591, 28.7507248
16: -22.5418129, 18.5903454, -22.6210480, 18.5942841, -41.1360970, 41.2113953
17: -43.9599686, 23.8260555, -44.0369873, 23.8629093, -66.6544189, 66.6932373
18: -18.6533489, 9.0332527, -18.6564560, 9.0948153, -27.7481651, 27.6897087
19: -22.8190479, 3.1354620, -22.8438454, 3.1794639, -25.9985123, 25.9793072
20: -14.4433498, 9.1689463, -14.4504175, 9.2150431, -23.6583939, 23.6193638
21: -22.1284294, 9.7341022, -22.1497307, 9.7905769, -31.9190063, 31.8838329
22: -27.1521416, 9.7649193, -27.1629543, 9.8313494, -36.9834900, 36.9278717
23: -20.6561222, 5.7133694, -20.6908035, 5.7753181, -26.4314404, 26.4041729
24: -27.9187107, -0.1234555, -27.9405785, -0.0358205, -27.8440170, 27.7742844
25: -19.8284416, 7.6089859, -19.8390999, 7.6613584, -27.4897995, 27.4480858
26: -33.1711082, 7.1522231, -33.2083435, 7.2624497, -40.4335594, 40.3605652
27: -22.6573105, 9.5103397, -22.6725121, 9.6145611, -31.9207382, 31.8280869
28: -20.5885334, 7.0047283, -20.6101856, 7.0644045, -27.6529388, 27.6149139
29: -33.0164719, 8.9680042, -33.0269852, 9.0323582, -41.4776001, 41.4251404
30: -22.3930702, 8.0316105, -22.3928032, 8.0622425, -30.4553127, 30.4244137
31: -20.4452896, 9.0753632, -20.4740067, 9.1227779, -29.5680676, 29.5493698
32: -19.6256790, 9.6625586, -19.6481075, 9.6998043, -28.2246170, 28.2367973
33: -42.4084549, 5.4985199, -42.4235687, 5.5184622, -45.6391144, 45.6418991
34: -31.3972855, 7.4750824, -31.4312134, 7.5681953, -37.2129593, 37.1531677
35: -31.8231468, 7.6837435, -31.8398342, 7.7263985, -38.7913055, 38.7696075
36: -32.1288376, 7.3460622, -32.1480942, 7.3879704, -38.8462448, 38.8297882
37: -49.7308197, -2.4169002, -49.7590103, -2.3643460, -43.9892807, 43.9618530
38: -40.4830208, 8.2249851, -40.5170326, 8.3261156, -47.7344360, 47.6899338
39: -54.5220909, -1.4259939, -54.5550079, -1.4029455, -52.5240173, 52.5330048
40: -40.0819778, 2.8481898, -40.1086426, 2.8803444, -40.1696167, 40.1749802
41: -25.7841625, 4.1126919, -25.8104572, 4.1591749, -26.3591576, 26.3568230
42: -16.4310684, 8.0178719, -16.4392147, 8.0393906, -22.9743156, 22.9989643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=56, inp2_unstable=56, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=246, inp2_unstable=247, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=20, inp2_unstable=20, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=40, inp2_unstable=40, delta_unstable=43

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 721
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1678
type: A, layer: 1, pos: 627
type: B, layer: 1, pos: 627
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 719
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 705
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1780
type: A, layer: 1, pos: 1780
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1777
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1711
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 704
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 704
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 690
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 612
type: A, layer: 1, pos: 612
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1715
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1715
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1336
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1335
type: A, layer: 1, pos: 674
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 674
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 673
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1362
type: B, layer: 1, pos: 1362
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1783

## Relational analysis of IS_B2_A2_B2_B1_A1_A1_B1

### Relational analysis result of IS_B2_A2_B2_B1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -16.8839367, upper bound: 16.9578352
time: 36.07 seconds

## Relational analysis of IS_B2_A2_B2_B1_A1_A1_B2

### Relational analysis result of IS_B2_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -16.9147286, upper bound: 16.9629847
time: 31.36 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 70.04 seconds
IS_B2_A2_B1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 70.04
Output dim: 7, lower bound: -16.8907247, upper bound: 16.9510769
IS_B2_A2_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 70.04
Output dim: 7, lower bound: -16.8907247, upper bound: 16.9629850
IS_B2_A2_B1_B2_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 70.04
Output dim: 7, lower bound: -16.8907247, upper bound: 16.9514352
IS_B2_A2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 70.04
Output dim: 7, lower bound: -16.9175061, upper bound: 16.9633503
IS_B2_A2_B2_B1_A1_A1_B1, status: Status.VERIFIED, split count: 7, time: 70.04
Output dim: 7, lower bound: -16.8839367, upper bound: 16.9578352
IS_B2_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 70.04
Output dim: 7, lower bound: -16.9147286, upper bound: 16.9629847
IS_B2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 70.04
Output dim: 7, lower bound: -16.9270892, upper bound: 16.9637879
IS_B2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 70.04
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9634245
IS_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 70.04
Output dim: 7, lower bound: -16.8963439, upper bound: 16.9637879

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 51.09 + 1797.79 = 1848.88 seconds
