## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 7)
Time budget: 3600 seconds
Split limit: 100
Threshold: 51.2181686794


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=48, inp2_unstable=48, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-62.1904907, 22.0116215, -62.1904907, 22.0116215, -84.2021103, 84.2021103)
1: (-34.7129898, 24.8510170, -34.7129898, 24.8510170, -59.5640068, 59.5640030)
2: (-27.8456993, 22.9933567, -27.8456993, 22.9933567, -50.8390579, 50.8390541)
3: (-32.7335243, 29.1557751, -32.7335243, 29.1557751, -61.8892975, 61.8892975)
4: (-34.7061081, 27.2610359, -34.7061081, 27.2610359, -61.9671440, 61.9671402)
5: (-30.8502083, 32.4295158, -30.8502083, 32.4295158, -63.2797241, 63.2797241)
6: (-37.9479370, 29.4828987, -37.9479370, 29.4828987, -67.4308319, 67.4308319)
7: (-40.0783234, 31.7114029, -40.0783234, 31.7114029, -71.7897186, 71.7897263)
8: (-38.7026634, 33.1363564, -38.7026634, 33.1363564, -71.8390198, 71.8390198)
9: (-29.9568024, 31.6364326, -29.9568024, 31.6364326, -61.5932350, 61.5932350)
10: (-43.7579842, 47.9829826, -43.7579842, 47.9829826, -91.7409668, 91.7409668)
11: (-44.8636398, 26.0650711, -44.8636398, 26.0650711, -70.9287109, 70.9287109)
12: (-42.2284355, 34.2841911, -42.2284355, 34.2841911, -76.5126190, 76.5126190)
13: (-45.5958176, 40.1284027, -45.5958176, 40.1284027, -85.7242203, 85.7242203)
14: (-77.7727966, 23.1575775, -77.7727966, 23.1575775, -100.9303741, 100.9303741)
15: (-37.0161057, 28.0862560, -37.0161057, 28.0862560, -65.1023636, 65.1023636)
16: (-48.1769371, 35.7739182, -48.1769371, 35.7739182, -83.9508514, 83.9508514)
17: (-77.0876617, 35.4324455, -77.0876617, 35.4324455, -112.5201111, 112.5201035)
18: (-40.6185570, 28.6590481, -40.6185570, 28.6590481, -69.2776031, 69.2776031)
19: (-30.9227695, 16.3407841, -30.9227695, 16.3407841, -47.2635536, 47.2635536)
20: (-31.8609276, 19.5329990, -31.8609276, 19.5329990, -51.3939209, 51.3939285)
21: (-43.7039795, 18.6627541, -43.7039795, 18.6627541, -62.3667297, 62.3667336)
22: (-51.3969116, 17.5496368, -51.3969116, 17.5496368, -68.9465485, 68.9465485)
23: (-31.9608231, 23.2057343, -31.9608231, 23.2057343, -55.1665573, 55.1665573)
24: (-44.3418083, 22.8910675, -44.3418083, 22.8910675, -67.2328720, 67.2328796)
25: (-34.0386887, 25.6217136, -34.0386887, 25.6217136, -59.6603966, 59.6604004)
26: (-49.9049988, 33.9512100, -49.9049988, 33.9512100, -83.8562088, 83.8562088)
27: (-47.8394241, 21.3874321, -47.8394241, 21.3874321, -69.2268524, 69.2268524)
28: (-33.9242477, 23.8886127, -33.9242477, 23.8886127, -57.8128548, 57.8128586)
29: (-57.2876282, 16.4622059, -57.2876282, 16.4622059, -73.7498322, 73.7498245)
30: (-42.2382278, 25.3053398, -42.2382278, 25.3053398, -67.5435638, 67.5435638)
31: (-39.6838112, 26.1002426, -39.6838112, 26.1002426, -65.7840500, 65.7840576)
32: (-44.9297256, 22.9506683, -44.9297256, 22.9506683, -67.8803940, 67.8803864)
33: (-59.6186180, 38.0911942, -59.6186180, 38.0911942, -97.7098083, 97.7098083)
34: (-52.2903061, 25.4515991, -52.2903061, 25.4515991, -77.7419052, 77.7419052)
35: (-53.5936279, 29.3012886, -53.5936279, 29.3012886, -82.8949051, 82.8949127)
36: (-53.2807541, 28.9255981, -53.2807541, 28.9255981, -82.2063522, 82.2063522)
37: (-71.7362900, 29.5404472, -71.7362900, 29.5404472, -101.2767258, 101.2767334)
38: (-62.8544884, 33.8173866, -62.8544884, 33.8173866, -96.6718674, 96.6718750)
39: (-72.9953918, 33.9001846, -72.9953918, 33.9001846, -106.8955688, 106.8955688)
40: (-59.3336563, 29.4873638, -59.3336563, 29.4873638, -88.8210220, 88.8210144)
41: (-41.8998718, 27.6854591, -41.8998718, 27.6854591, -69.5853271, 69.5853271)
42: (-29.5159225, 24.2377357, -29.5159225, 24.2377357, -53.7536545, 53.7536545)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.39 + 89.22 = 91.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 10, lower bound: -51.3208103, upper bound: 51.3208103

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1685

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -51.3140055, upper bound: 51.0877201
time: 79.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -51.3140240, upper bound: 51.3140235
time: 78.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 158.28 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 158.28
Output dim: 10, lower bound: -51.3140055, upper bound: 51.0877201
IS_A2, status: Status.UNKNOWN, split count: 1, time: 158.28
Output dim: 10, lower bound: -51.3140240, upper bound: 51.3140235

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -62.1167297, 21.9631233, -62.1719360, 21.9966393, -84.1133728, 84.1350555
1: -34.6780930, 24.7865295, -34.7045631, 24.8297482, -59.5078392, 59.4910927
2: -27.8013992, 22.9544010, -27.8333664, 22.9821510, -50.7835503, 50.7877655
3: -32.7007484, 29.0310364, -32.7201614, 29.1166477, -61.8173904, 61.7511978
4: -34.6237221, 27.2082520, -34.6813850, 27.2427483, -61.8664703, 61.8896370
5: -30.7992210, 32.3024979, -30.8385315, 32.3868103, -63.1860313, 63.1410294
6: -37.8490295, 29.4478855, -37.9192238, 29.4742432, -67.3232727, 67.3671112
7: -40.0260010, 31.5813618, -40.0691376, 31.6679783, -71.6939774, 71.6504974
8: -38.6206322, 33.0756226, -38.6767120, 33.1173553, -71.7379913, 71.7523346
9: -29.8554649, 31.4505463, -29.9433517, 31.5748825, -61.4303436, 61.3938980
10: -43.5642624, 47.5592995, -43.7381935, 47.8391914, -91.4034576, 91.2974930
11: -44.7479095, 25.8545437, -44.8469315, 25.9925842, -70.7404938, 70.7014771
12: -42.1491470, 34.2091484, -42.2056160, 34.2598953, -76.4090347, 76.4147644
13: -45.5408478, 40.0383301, -45.5782700, 40.1029663, -85.6438141, 85.6165924
14: -77.6534271, 22.9747028, -77.7552872, 23.0950470, -100.7484741, 100.7299881
15: -36.9692535, 27.9842663, -37.0040894, 28.0551491, -65.0243988, 64.9883575
16: -48.0270691, 35.5125999, -48.1528969, 35.6845856, -83.7116547, 83.6654968
17: -76.9311218, 35.1717300, -77.0669174, 35.3450851, -112.2762070, 112.2386398
18: -40.5010338, 28.5979042, -40.5866928, 28.6398354, -69.1408615, 69.1845932
19: -30.8435879, 16.2917690, -30.9042797, 16.3265858, -47.1701736, 47.1960487
20: -31.8129406, 19.5007820, -31.8485126, 19.5236626, -51.3366013, 51.3492928
21: -43.6106339, 18.6035118, -43.6838951, 18.6445274, -62.2551613, 62.2874069
22: -51.2471161, 17.5104256, -51.3485298, 17.5349483, -68.7820663, 68.8589554
23: -31.8994617, 23.1330147, -31.9446297, 23.1813736, -55.0808334, 55.0776443
24: -44.2125511, 22.8479195, -44.2977753, 22.8810959, -67.0936432, 67.1456909
25: -33.9753265, 25.5792084, -34.0239601, 25.6099472, -59.5852623, 59.6031685
26: -49.7855568, 33.8977928, -49.8683167, 33.9342041, -83.7197571, 83.7661133
27: -47.6595459, 21.3175964, -47.7791176, 21.3760872, -69.0356293, 69.0967102
28: -33.8523102, 23.8484688, -33.9013100, 23.8800697, -57.7323799, 57.7497787
29: -57.1986961, 16.4275360, -57.2620735, 16.4481926, -73.6468887, 73.6896057
30: -42.1855888, 25.2304020, -42.2239151, 25.2831345, -67.4687195, 67.4543152
31: -39.6038437, 26.0535088, -39.6627769, 26.0859089, -65.6897507, 65.7162857
32: -44.8146248, 22.9033737, -44.8932571, 22.9434853, -67.7581024, 67.7966309
33: -59.3721428, 37.9861450, -59.5351944, 38.0830078, -97.4551544, 97.5213394
34: -52.1316452, 25.3784027, -52.2374802, 25.4422302, -77.5738754, 77.6158829
35: -53.3658829, 29.2064152, -53.5164719, 29.2942619, -82.6601410, 82.7228775
36: -53.0503693, 28.8362465, -53.2033081, 28.9193611, -81.9697266, 82.0395508
37: -71.4059448, 29.4298325, -71.6265182, 29.5331135, -100.9390564, 101.0563507
38: -62.5698471, 33.7092476, -62.7600861, 33.8064423, -96.3762894, 96.4693298
39: -72.6715698, 33.7938118, -72.8861237, 33.8936768, -106.5652390, 106.6799316
40: -59.0722809, 29.3980942, -59.2470131, 29.4821663, -88.5544434, 88.6451035
41: -41.7463226, 27.6383743, -41.8502998, 27.6798058, -69.4261322, 69.4886703
42: -29.4088573, 24.0973549, -29.4964504, 24.1918087, -53.6006660, 53.5937996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=47, inp2_unstable=48, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1669

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -51.0858023, upper bound: 51.0740876
time: 71.26 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 10, lower bound: -51.0858023, upper bound: 51.0741164
time: 56.85 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -62.1854401, 22.0091496, -62.1898193, 22.0112972, -84.1967392, 84.1989670
1: -34.7107391, 24.8475609, -34.7126999, 24.8505630, -59.5613022, 59.5602608
2: -27.8403683, 22.9911709, -27.8450089, 22.9930725, -50.8334389, 50.8361778
3: -32.7307434, 29.1378956, -32.7331543, 29.1533813, -61.8841248, 61.8710480
4: -34.6942215, 27.2583694, -34.7045212, 27.2606773, -61.9548988, 61.9628830
5: -30.8479424, 32.4174652, -30.8499222, 32.4279709, -63.2759018, 63.2673874
6: -37.9429016, 29.4805870, -37.9472733, 29.4825859, -67.4254913, 67.4278564
7: -40.0762138, 31.7059746, -40.0780220, 31.7107010, -71.7869034, 71.7839966
8: -38.7001610, 33.1330719, -38.7023354, 33.1359138, -71.8360748, 71.8354034
9: -29.9530792, 31.6303101, -29.9563084, 31.6356392, -61.5887146, 61.5866165
10: -43.7543793, 47.9683151, -43.7575111, 47.9810944, -91.7354736, 91.7258301
11: -44.8598175, 26.0560646, -44.8631172, 26.0639229, -70.9237366, 70.9191818
12: -42.2253189, 34.2763672, -42.2280197, 34.2831268, -76.5084381, 76.5043793
13: -45.5896454, 40.1182861, -45.5949860, 40.1270599, -85.7167053, 85.7132721
14: -77.7684479, 23.1516094, -77.7721939, 23.1568165, -100.9252625, 100.9237976
15: -37.0129890, 28.0814419, -37.0157089, 28.0855865, -65.0985641, 65.0971375
16: -48.1708946, 35.7647057, -48.1761436, 35.7727356, -83.9436340, 83.9408417
17: -77.0837326, 35.4230309, -77.0871277, 35.4312210, -112.5149536, 112.5101547
18: -40.6027985, 28.6548538, -40.6165390, 28.6584969, -69.2612915, 69.2713928
19: -30.9193287, 16.3384380, -30.9223099, 16.3404789, -47.2598076, 47.2607460
20: -31.8582516, 19.5297775, -31.8605442, 19.5325661, -51.3908157, 51.3903198
21: -43.7008667, 18.6594982, -43.7035599, 18.6623249, -62.3631897, 62.3630524
22: -51.3927078, 17.5463753, -51.3963547, 17.5491982, -68.9419022, 68.9427338
23: -31.9576359, 23.2024326, -31.9604053, 23.2052956, -55.1629333, 55.1628342
24: -44.3269196, 22.8879700, -44.3398819, 22.8906574, -67.2175751, 67.2278519
25: -34.0343475, 25.6188793, -34.0381088, 25.6213169, -59.6556625, 59.6569862
26: -49.8941307, 33.9464874, -49.9035492, 33.9505997, -83.8447266, 83.8500290
27: -47.8296165, 21.3833370, -47.8381653, 21.3869057, -69.2165222, 69.2214966
28: -33.9200287, 23.8865852, -33.9236908, 23.8883476, -57.8083725, 57.8102722
29: -57.2783165, 16.4590931, -57.2864037, 16.4617920, -73.7401047, 73.7454987
30: -42.2337952, 25.3005314, -42.2376251, 25.3047180, -67.5385132, 67.5381546
31: -39.6790237, 26.0967999, -39.6831894, 26.0997944, -65.7788162, 65.7799911
32: -44.9244041, 22.9469147, -44.9290237, 22.9501915, -67.8745956, 67.8759384
33: -59.6076202, 38.0891762, -59.6171799, 38.0909195, -97.6985321, 97.7063522
34: -52.2830124, 25.4493561, -52.2893600, 25.4512978, -77.7343140, 77.7387085
35: -53.5833473, 29.3001842, -53.5923042, 29.3011169, -82.8844604, 82.8924866
36: -53.2723808, 28.9246292, -53.2796822, 28.9254570, -82.1978302, 82.2043152
37: -71.7213898, 29.5387936, -71.7343597, 29.5402107, -101.2615967, 101.2731476
38: -62.8445740, 33.8144684, -62.8532257, 33.8170013, -96.6615753, 96.6676941
39: -72.9807892, 33.8982773, -72.9934845, 33.8999176, -106.8806915, 106.8917618
40: -59.3234520, 29.4862251, -59.3323059, 29.4872093, -88.8106537, 88.8185272
41: -41.8937607, 27.6833992, -41.8990631, 27.6851921, -69.5789490, 69.5824585
42: -29.5133190, 24.2316189, -29.5155773, 24.2369461, -53.7502670, 53.7471962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=47, inp2_unstable=48, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1685

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -51.0877200, upper bound: 51.3140056
time: 56.62 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -51.0877200, upper bound: 51.3140241
time: 65.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 123.91 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 123.91
Output dim: 10, lower bound: -51.0858023, upper bound: 51.0740876
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 123.91
Output dim: 10, lower bound: -51.0858023, upper bound: 51.0741164
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 123.91
Output dim: 10, lower bound: -51.0877200, upper bound: 51.3140056
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 123.91
Output dim: 10, lower bound: -51.0877200, upper bound: 51.3140241

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -62.1854401, 22.0091496, -62.1167297, 21.9631233, -84.1485596, 84.1258774
1: -34.7107391, 24.8475609, -34.6780930, 24.7865295, -59.4972687, 59.5256538
2: -27.8403683, 22.9911709, -27.8013992, 22.9544010, -50.7947617, 50.7925720
3: -32.7307434, 29.1378956, -32.7007484, 29.0310364, -61.7617798, 61.8386459
4: -34.6942215, 27.2583694, -34.6237221, 27.2082520, -61.9024734, 61.8820915
5: -30.8479424, 32.4174652, -30.7992210, 32.3024979, -63.1504402, 63.2166862
6: -37.9429016, 29.4805870, -37.8490295, 29.4478855, -67.3907852, 67.3296204
7: -40.0762138, 31.7059746, -40.0260010, 31.5813618, -71.6575623, 71.7319794
8: -38.7001610, 33.1330719, -38.6206322, 33.0756226, -71.7757797, 71.7537079
9: -29.9530792, 31.6303101, -29.8554649, 31.4505463, -61.4036255, 61.4857750
10: -43.7543793, 47.9683151, -43.5642624, 47.5592995, -91.3136749, 91.5325775
11: -44.8598175, 26.0560646, -44.7479095, 25.8545437, -70.7143555, 70.8039703
12: -42.2253189, 34.2763672, -42.1491470, 34.2091484, -76.4344635, 76.4255142
13: -45.5896454, 40.1182861, -45.5408478, 40.0383301, -85.6279755, 85.6591339
14: -77.7684479, 23.1516094, -77.6534271, 22.9747028, -100.7431488, 100.8050385
15: -37.0129890, 28.0814419, -36.9692535, 27.9842663, -64.9972534, 65.0506897
16: -48.1708946, 35.7647057, -48.0270691, 35.5125999, -83.6834869, 83.7917786
17: -77.0837326, 35.4230309, -76.9311218, 35.1717300, -112.2554626, 112.3541412
18: -40.6027985, 28.6548538, -40.5010338, 28.5979042, -69.2006989, 69.1558838
19: -30.9193287, 16.3384380, -30.8435879, 16.2917690, -47.2110977, 47.1820221
20: -31.8582516, 19.5297775, -31.8129406, 19.5007820, -51.3590317, 51.3427200
21: -43.7008667, 18.6594982, -43.6106339, 18.6035118, -62.3043747, 62.2701340
22: -51.3927078, 17.5463753, -51.2471161, 17.5104256, -68.9031372, 68.7934875
23: -31.9576359, 23.2024326, -31.8994617, 23.1330147, -55.0906525, 55.1018944
24: -44.3269196, 22.8879700, -44.2125511, 22.8479195, -67.1748276, 67.1005249
25: -34.0343475, 25.6188793, -33.9753265, 25.5792084, -59.6135559, 59.5942078
26: -49.8941307, 33.9464874, -49.7855568, 33.8977928, -83.7919235, 83.7320404
27: -47.8296165, 21.3833370, -47.6595459, 21.3175964, -69.1472092, 69.0428772
28: -33.9200287, 23.8865852, -33.8523102, 23.8484688, -57.7684975, 57.7388954
29: -57.2783165, 16.4590931, -57.1986961, 16.4275360, -73.7058487, 73.6577835
30: -42.2337952, 25.3005314, -42.1855888, 25.2304020, -67.4641953, 67.4861221
31: -39.6790237, 26.0967999, -39.6038437, 26.0535088, -65.7325287, 65.7006454
32: -44.9244041, 22.9469147, -44.8146248, 22.9033737, -67.8277740, 67.7615356
33: -59.6076202, 38.0891762, -59.3721428, 37.9861450, -97.5937653, 97.4613190
34: -52.2830124, 25.4493561, -52.1316452, 25.3784027, -77.6614151, 77.5810013
35: -53.5833473, 29.3001842, -53.3658829, 29.2064152, -82.7897568, 82.6660690
36: -53.2723808, 28.9246292, -53.0503693, 28.8362465, -82.1086121, 81.9749985
37: -71.7213898, 29.5387936, -71.4059448, 29.4298325, -101.1512222, 100.9447403
38: -62.8445740, 33.8144684, -62.5698471, 33.7092476, -96.5538177, 96.3843155
39: -72.9807892, 33.8982773, -72.6715698, 33.7938118, -106.7745972, 106.5698395
40: -59.3234520, 29.4862251, -59.0722809, 29.3980942, -88.7215424, 88.5585022
41: -41.8937607, 27.6833992, -41.7463226, 27.6383743, -69.5321350, 69.4297180
42: -29.5133190, 24.2316189, -29.4088573, 24.0973549, -53.6106720, 53.6404724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=47, inp2_unstable=47, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1669

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -51.0740875, upper bound: 51.0858010
time: 72.95 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -51.0741163, upper bound: 51.3083667
time: 80.96 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -62.1854401, 22.0091496, -62.1854401, 22.0091496, -84.1945877, 84.1945877
1: -34.7107391, 24.8475609, -34.7107391, 24.8475609, -59.5582962, 59.5582962
2: -27.8403683, 22.9911709, -27.8403683, 22.9911709, -50.8315353, 50.8315392
3: -32.7307434, 29.1378956, -32.7307434, 29.1378956, -61.8686371, 61.8686371
4: -34.6942215, 27.2583694, -34.6942215, 27.2583694, -61.9525909, 61.9525909
5: -30.8479424, 32.4174652, -30.8479424, 32.4174652, -63.2654076, 63.2654076
6: -37.9429016, 29.4805870, -37.9429016, 29.4805870, -67.4234924, 67.4234924
7: -40.0762138, 31.7059746, -40.0762138, 31.7059746, -71.7821884, 71.7821808
8: -38.7001610, 33.1330719, -38.7001610, 33.1330719, -71.8332367, 71.8332367
9: -29.9530792, 31.6303101, -29.9530792, 31.6303101, -61.5833893, 61.5833893
10: -43.7543793, 47.9683151, -43.7543793, 47.9683151, -91.7226944, 91.7226944
11: -44.8598175, 26.0560646, -44.8598175, 26.0560646, -70.9158783, 70.9158783
12: -42.2253189, 34.2763672, -42.2253189, 34.2763672, -76.5016861, 76.5016785
13: -45.5896454, 40.1182861, -45.5896454, 40.1182861, -85.7079315, 85.7079315
14: -77.7684479, 23.1516094, -77.7684479, 23.1516094, -100.9200516, 100.9200592
15: -37.0129890, 28.0814419, -37.0129890, 28.0814419, -65.0944290, 65.0944214
16: -48.1708946, 35.7647057, -48.1708946, 35.7647057, -83.9355774, 83.9355927
17: -77.0837326, 35.4230309, -77.0837326, 35.4230309, -112.5067520, 112.5067596
18: -40.6027985, 28.6548538, -40.6027985, 28.6548538, -69.2576523, 69.2576523
19: -30.9193287, 16.3384380, -30.9193287, 16.3384380, -47.2577667, 47.2577667
20: -31.8582516, 19.5297775, -31.8582516, 19.5297775, -51.3880310, 51.3880310
21: -43.7008667, 18.6594982, -43.7008667, 18.6594982, -62.3603592, 62.3603668
22: -51.3927078, 17.5463753, -51.3927078, 17.5463753, -68.9390793, 68.9390793
23: -31.9576359, 23.2024326, -31.9576359, 23.2024326, -55.1600685, 55.1600685
24: -44.3269196, 22.8879700, -44.3269196, 22.8879700, -67.2148895, 67.2148895
25: -34.0343475, 25.6188793, -34.0343475, 25.6188793, -59.6532288, 59.6532288
26: -49.8941307, 33.9464874, -49.8941307, 33.9464874, -83.8406067, 83.8406143
27: -47.8296165, 21.3833370, -47.8296165, 21.3833370, -69.2129517, 69.2129517
28: -33.9200287, 23.8865852, -33.9200287, 23.8865852, -57.8066101, 57.8066139
29: -57.2783165, 16.4590931, -57.2783165, 16.4590931, -73.7374039, 73.7374039
30: -42.2337952, 25.3005314, -42.2337952, 25.3005314, -67.5343246, 67.5343246
31: -39.6790237, 26.0967999, -39.6790237, 26.0967999, -65.7758255, 65.7758255
32: -44.9244041, 22.9469147, -44.9244041, 22.9469147, -67.8713226, 67.8713226
33: -59.6076202, 38.0891762, -59.6076202, 38.0891762, -97.6967926, 97.6967926
34: -52.2830124, 25.4493561, -52.2830124, 25.4493561, -77.7323608, 77.7323685
35: -53.5833473, 29.3001842, -53.5833473, 29.3001842, -82.8835297, 82.8835297
36: -53.2723808, 28.9246292, -53.2723808, 28.9246292, -82.1970062, 82.1970062
37: -71.7213898, 29.5387936, -71.7213898, 29.5387936, -101.2601852, 101.2601852
38: -62.8445740, 33.8144684, -62.8445740, 33.8144684, -96.6590424, 96.6590424
39: -72.9807892, 33.8982773, -72.9807892, 33.8982773, -106.8790588, 106.8790588
40: -59.3234520, 29.4862251, -59.3234520, 29.4862251, -88.8096771, 88.8096771
41: -41.8937607, 27.6833992, -41.8937607, 27.6833992, -69.5771637, 69.5771637
42: -29.5133190, 24.2316189, -29.5133190, 24.2316189, -53.7449379, 53.7449379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=47, inp2_unstable=47, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1669
type: B, layer: 1, pos: 1669
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1669

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -51.0760104, upper bound: 51.0860395
time: 83.12 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -51.0760104, upper bound: 51.3083986
time: 79.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 164.78 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 164.78
Output dim: 10, lower bound: -51.0740875, upper bound: 51.0858010
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 164.78
Output dim: 10, lower bound: -51.0741163, upper bound: 51.3083667
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 164.78
Output dim: 10, lower bound: -51.0760104, upper bound: 51.0860395
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 164.78
Output dim: 10, lower bound: -51.0760104, upper bound: 51.3083986

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -62.1742134, 22.0076885, -62.1144943, 21.9628086, -84.1370239, 84.1221771
1: -34.7014847, 24.8462486, -34.6763191, 24.7862835, -59.4877701, 59.5225639
2: -27.8346004, 22.9897385, -27.8002014, 22.9541130, -50.7887115, 50.7899361
3: -32.7226753, 29.1355915, -32.6995354, 29.0305748, -61.7532463, 61.8351250
4: -34.6812057, 27.2566681, -34.6212387, 27.2078743, -61.8890800, 61.8779068
5: -30.8412342, 32.4152985, -30.7979088, 32.3020706, -63.1433029, 63.2132072
6: -37.9389343, 29.4791756, -37.8482361, 29.4476204, -67.3865509, 67.3274078
7: -40.0680199, 31.7043133, -40.0245361, 31.5810375, -71.6490555, 71.7288513
8: -38.6973419, 33.1311493, -38.6198273, 33.0752563, -71.7725983, 71.7509766
9: -29.9503651, 31.6250648, -29.8549004, 31.4495392, -61.3999023, 61.4799652
10: -43.7510681, 47.9551773, -43.5636024, 47.5568237, -91.3078918, 91.5187836
11: -44.8568077, 26.0509644, -44.7472458, 25.8535709, -70.7103806, 70.7982101
12: -42.2231636, 34.2694550, -42.1486969, 34.2079201, -76.4310760, 76.4181519
13: -45.5856476, 40.1147385, -45.5400696, 40.0376053, -85.6232452, 85.6548080
14: -77.7652740, 23.1454430, -77.6528320, 22.9735336, -100.7388000, 100.7982712
15: -37.0039749, 28.0781326, -36.9674683, 27.9835892, -64.9875641, 65.0456009
16: -48.1664848, 35.7574310, -48.0261688, 35.5112534, -83.6777344, 83.7835999
17: -77.0813904, 35.4160690, -76.9306412, 35.1703796, -112.2517700, 112.3467102
18: -40.5974426, 28.6491566, -40.5000076, 28.5968056, -69.1942444, 69.1491623
19: -30.9169674, 16.3336887, -30.8431301, 16.2908669, -47.2078323, 47.1768150
20: -31.8563538, 19.5260544, -31.8125629, 19.5000629, -51.3564148, 51.3386154
21: -43.6985703, 18.6536274, -43.6101685, 18.6024303, -62.3010025, 62.2637939
22: -51.3890419, 17.5432835, -51.2463913, 17.5097961, -68.8988342, 68.7896729
23: -31.9556675, 23.1952400, -31.8990650, 23.1314392, -55.0871048, 55.0943069
24: -44.3224335, 22.8861122, -44.2113762, 22.8475266, -67.1699524, 67.0974884
25: -34.0317535, 25.6140099, -33.9748116, 25.5783138, -59.6100655, 59.5888214
26: -49.8912277, 33.9370575, -49.7849808, 33.8963928, -83.7876053, 83.7220306
27: -47.8245468, 21.3811073, -47.6585350, 21.3171711, -69.1417160, 69.0396423
28: -33.9166489, 23.8850098, -33.8516922, 23.8481464, -57.7647934, 57.7367020
29: -57.2728729, 16.4558563, -57.1973724, 16.4269333, -73.6998062, 73.6532288
30: -42.2308884, 25.2944660, -42.1849861, 25.2292290, -67.4601135, 67.4794540
31: -39.6763573, 26.0938644, -39.6033401, 26.0529366, -65.7292938, 65.6971970
32: -44.9213982, 22.9446335, -44.8138428, 22.9028912, -67.8242874, 67.7584763
33: -59.6006546, 38.0872307, -59.3707962, 37.9857292, -97.5863800, 97.4580231
34: -52.2772102, 25.4472046, -52.1305389, 25.3779678, -77.6551819, 77.5777359
35: -53.5759811, 29.2987671, -53.3644714, 29.2061062, -82.7820892, 82.6632385
36: -53.2658081, 28.9235039, -53.0491104, 28.8359909, -82.1017990, 81.9726105
37: -71.7139740, 29.5371513, -71.4044495, 29.4295292, -101.1435013, 100.9416046
38: -62.8361206, 33.8125038, -62.5682602, 33.7088928, -96.5450134, 96.3807678
39: -72.9729309, 33.8969269, -72.6700439, 33.7935905, -106.7665176, 106.5669632
40: -59.3174019, 29.4853172, -59.0710602, 29.3979225, -88.7153168, 88.5563812
41: -41.8897934, 27.6819916, -41.7455254, 27.6381264, -69.5279236, 69.4275208
42: -29.5113716, 24.2259617, -29.4084301, 24.0962715, -53.6076431, 53.6343918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=46, inp2_unstable=47, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1669

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.9314875, upper bound: 51.3083671
time: 86.50 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.9314875, upper bound: 51.3083671
time: 63.94 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -62.1742134, 22.0076885, -62.1831932, 22.0088539, -84.1830673, 84.1908798
1: -34.7014847, 24.8462486, -34.7089577, 24.8473034, -59.5487823, 59.5552025
2: -27.8346004, 22.9897385, -27.8391495, 22.9908848, -50.8254814, 50.8288879
3: -32.7226753, 29.1355915, -32.7291794, 29.1374378, -61.8601074, 61.8647690
4: -34.6812057, 27.2566681, -34.6917343, 27.2580261, -61.9392319, 61.9484024
5: -30.8412342, 32.4152985, -30.8465881, 32.4170380, -63.2582703, 63.2618866
6: -37.9389343, 29.4791756, -37.9421158, 29.4803085, -67.4192429, 67.4212952
7: -40.0680199, 31.7043133, -40.0746231, 31.7056465, -71.7736664, 71.7789307
8: -38.6973419, 33.1311493, -38.6995735, 33.1327057, -71.8300476, 71.8307190
9: -29.9503651, 31.6250648, -29.9525414, 31.6292915, -61.5796432, 61.5776062
10: -43.7510681, 47.9551773, -43.7537308, 47.9657974, -91.7168655, 91.7089081
11: -44.8568077, 26.0509644, -44.8592148, 26.0550728, -70.9118805, 70.9101791
12: -42.2231636, 34.2694550, -42.2248840, 34.2750664, -76.4982300, 76.4943390
13: -45.5856476, 40.1147385, -45.5888596, 40.1175652, -85.7032089, 85.7035980
14: -77.7652740, 23.1454430, -77.7678375, 23.1504307, -100.9157028, 100.9132767
15: -37.0039749, 28.0781326, -37.0111809, 28.0807819, -65.0847549, 65.0893097
16: -48.1664848, 35.7574310, -48.1700096, 35.7633057, -83.9297867, 83.9274292
17: -77.0813904, 35.4160690, -77.0833054, 35.4216881, -112.5030823, 112.4993668
18: -40.5974426, 28.6491566, -40.6017609, 28.6537380, -69.2511826, 69.2509155
19: -30.9169674, 16.3336887, -30.9188766, 16.3375244, -47.2544899, 47.2525635
20: -31.8563538, 19.5260544, -31.8578815, 19.5290489, -51.3854027, 51.3839340
21: -43.6985703, 18.6536274, -43.7004013, 18.6583672, -62.3569298, 62.3540268
22: -51.3890419, 17.5432835, -51.3919868, 17.5457478, -68.9347839, 68.9352722
23: -31.9556675, 23.1952400, -31.9572487, 23.2010441, -55.1567116, 55.1524887
24: -44.3224335, 22.8861122, -44.3260498, 22.8875866, -67.2100220, 67.2121582
25: -34.0317535, 25.6140099, -34.0338516, 25.6179276, -59.6496773, 59.6478615
26: -49.8912277, 33.9370575, -49.8935623, 33.9446030, -83.8358307, 83.8306198
27: -47.8245468, 21.3811073, -47.8286438, 21.3828850, -69.2074280, 69.2097473
28: -33.9166489, 23.8850098, -33.9193802, 23.8862457, -57.8028946, 57.8043900
29: -57.2728729, 16.4558563, -57.2772217, 16.4584389, -73.7313080, 73.7330780
30: -42.2308884, 25.2944660, -42.2332382, 25.2993507, -67.5302429, 67.5277023
31: -39.6763573, 26.0938644, -39.6785011, 26.0962181, -65.7725754, 65.7723541
32: -44.9213982, 22.9446335, -44.9238091, 22.9464703, -67.8678665, 67.8684387
33: -59.6006546, 38.0872307, -59.6062660, 38.0887985, -97.6894531, 97.6934967
34: -52.2772102, 25.4472046, -52.2818985, 25.4489250, -77.7261353, 77.7291031
35: -53.5759811, 29.2987671, -53.5819397, 29.2998981, -82.8758774, 82.8807068
36: -53.2658081, 28.9235039, -53.2711143, 28.9244118, -82.1902161, 82.1946182
37: -71.7139740, 29.5371513, -71.7199478, 29.5384789, -101.2524567, 101.2570953
38: -62.8361206, 33.8125038, -62.8429413, 33.8140869, -96.6502075, 96.6554413
39: -72.9729309, 33.8969269, -72.9792786, 33.8980179, -106.8709488, 106.8761902
40: -59.3174019, 29.4853172, -59.3222733, 29.4860516, -88.8034515, 88.8075867
41: -41.8897934, 27.6819916, -41.8929901, 27.6831360, -69.5729294, 69.5749817
42: -29.5113716, 24.2259617, -29.5129280, 24.2305431, -53.7419128, 53.7388916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=46, inp2_unstable=47, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1669

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.9336899, upper bound: 51.3083991
time: 81.56 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.9314875, upper bound: 51.3083991
time: 70.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 154.77 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 154.77
Output dim: 10, lower bound: -50.9314875, upper bound: 51.3083671
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 154.77
Output dim: 10, lower bound: -50.9314875, upper bound: 51.3083671
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 154.77
Output dim: 10, lower bound: -50.9336899, upper bound: 51.3083991
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 154.77
Output dim: 10, lower bound: -50.9314875, upper bound: 51.3083991

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -62.1742134, 22.0076885, -62.0391388, 21.9375916, -84.1118011, 84.0468292
1: -34.7014847, 24.8462486, -34.6339264, 24.7580395, -59.4595261, 59.4801750
2: -27.8346004, 22.9897385, -27.7396545, 22.9242992, -50.7588997, 50.7293930
3: -32.7226753, 29.1355915, -32.6191597, 28.9807129, -61.7033882, 61.7547493
4: -34.6812057, 27.2566681, -34.5544815, 27.1762276, -61.8574333, 61.8111496
5: -30.8412342, 32.4152985, -30.7425308, 32.2609634, -63.1021957, 63.1578293
6: -37.9389343, 29.4791756, -37.7571144, 29.4157791, -67.3547134, 67.2362823
7: -40.0680199, 31.7043133, -39.9722977, 31.5503960, -71.6184158, 71.6766129
8: -38.6973419, 33.1311493, -38.5400276, 33.0356369, -71.7329788, 71.6711731
9: -29.9503651, 31.6250648, -29.7754574, 31.3227882, -61.2731514, 61.4005203
10: -43.7510681, 47.9551773, -43.3856468, 47.2216530, -90.9727173, 91.3408203
11: -44.8568077, 26.0509644, -44.6585464, 25.6933384, -70.5501404, 70.7095108
12: -42.2231636, 34.2694550, -42.0500565, 34.0593491, -76.2825089, 76.3195114
13: -45.5856476, 40.1147385, -45.4872398, 39.9785538, -85.5642014, 85.6019745
14: -77.7652740, 23.1454430, -77.5319519, 22.8092251, -100.5744934, 100.6773987
15: -37.0039749, 28.0781326, -36.9235153, 27.9086227, -64.9125977, 65.0016479
16: -48.1664848, 35.7574310, -47.9167709, 35.3365250, -83.5030060, 83.6742020
17: -77.0813904, 35.4160690, -76.8369751, 35.0026207, -112.0840149, 112.2530441
18: -40.5974426, 28.6491566, -40.4313736, 28.5317478, -69.1291885, 69.0805283
19: -30.9169674, 16.3336887, -30.7827320, 16.2366219, -47.1535873, 47.1164207
20: -31.8563538, 19.5260544, -31.7691708, 19.4793892, -51.3357430, 51.2952271
21: -43.6985703, 18.6536274, -43.5438004, 18.5362225, -62.2347870, 62.1974258
22: -51.3890419, 17.5432835, -51.1490135, 17.4609070, -68.8499451, 68.6922989
23: -31.9556675, 23.1952400, -31.8587532, 23.0800133, -55.0356827, 55.0539932
24: -44.3224335, 22.8861122, -44.1461182, 22.8122196, -67.1346512, 67.0322266
25: -34.0317535, 25.6140099, -33.9310684, 25.5388832, -59.5706329, 59.5450783
26: -49.8912277, 33.9370575, -49.7229919, 33.7898483, -83.6810608, 83.6600494
27: -47.8245468, 21.3811073, -47.5271263, 21.2532368, -69.0777740, 68.9082336
28: -33.9166489, 23.8850098, -33.7742004, 23.8063202, -57.7229576, 57.6592102
29: -57.2728729, 16.4558563, -57.1277733, 16.3793221, -73.6521912, 73.5836334
30: -42.2308884, 25.2944660, -42.1418571, 25.1801033, -67.4109955, 67.4363251
31: -39.6763573, 26.0938644, -39.5379524, 25.9871712, -65.6635284, 65.6318130
32: -44.9213982, 22.9446335, -44.7383614, 22.8708706, -67.7922592, 67.6829987
33: -59.6006546, 38.0872307, -59.2264977, 37.8997078, -97.5003662, 97.3137283
34: -52.2772102, 25.4472046, -51.9918785, 25.3063030, -77.5835114, 77.4390869
35: -53.5759811, 29.2987671, -53.1999359, 29.1210060, -82.6969910, 82.4987030
36: -53.2658081, 28.9235039, -52.8863144, 28.7650795, -82.0308838, 81.8098145
37: -71.7139740, 29.5371513, -71.2670212, 29.3673897, -101.0813599, 100.8041687
38: -62.8361206, 33.8125038, -62.3673096, 33.6244812, -96.4606018, 96.1798096
39: -72.9729309, 33.8969269, -72.5198135, 33.7304993, -106.7034149, 106.4167404
40: -59.3174019, 29.4853172, -58.9372139, 29.3479347, -88.6653366, 88.4225311
41: -41.8897934, 27.6819916, -41.6519318, 27.5980034, -69.4877930, 69.3339233
42: -29.5113716, 24.2259617, -29.3390007, 24.0118370, -53.5232086, 53.5649605

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=46, inp2_unstable=46, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.9099959, upper bound: 51.0663932
time: 76.11 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.9099959, upper bound: 51.3019303
time: 59.25 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -62.1742134, 22.0076885, -62.1056900, 21.9615192, -84.1357346, 84.1133804
1: -34.7014847, 24.8462486, -34.6688538, 24.7852097, -59.4866943, 59.5150986
2: -27.8346004, 22.9897385, -27.7957802, 22.9529572, -50.7875595, 50.7855186
3: -32.7226753, 29.1355915, -32.6946487, 29.0286827, -61.7513504, 61.8302383
4: -34.6812057, 27.2566681, -34.6107254, 27.2065353, -61.8877411, 61.8673935
5: -30.8412342, 32.4152985, -30.7931404, 32.3002663, -63.1415024, 63.2084389
6: -37.9389343, 29.4791756, -37.8451004, 29.4464340, -67.3853683, 67.3242722
7: -40.0680199, 31.7043133, -40.0183868, 31.5796604, -71.6476822, 71.7227020
8: -38.6973419, 33.1311493, -38.6172409, 33.0737190, -71.7710571, 71.7483902
9: -29.9503651, 31.6250648, -29.8526783, 31.4453640, -61.3957214, 61.4777412
10: -43.7510681, 47.9551773, -43.5609436, 47.5464211, -91.2974854, 91.5161209
11: -44.8568077, 26.0509644, -44.7445831, 25.8495560, -70.7063599, 70.7955475
12: -42.2231636, 34.2694550, -42.1469879, 34.2026062, -76.4257660, 76.4164429
13: -45.5856476, 40.1147385, -45.5368423, 40.0347290, -85.6203766, 85.6515808
14: -77.7652740, 23.1454430, -77.6503830, 22.9686127, -100.7338791, 100.7958221
15: -37.0039749, 28.0781326, -36.9601631, 27.9808922, -64.9848633, 65.0382919
16: -48.1664848, 35.7574310, -48.0225906, 35.5055275, -83.6720123, 83.7800140
17: -77.0813904, 35.4160690, -76.9287109, 35.1648407, -112.2462311, 112.3447800
18: -40.5974426, 28.6491566, -40.4957123, 28.5922756, -69.1897125, 69.1448669
19: -30.9169674, 16.3336887, -30.8412743, 16.2871571, -47.2041245, 47.1749573
20: -31.8563538, 19.5260544, -31.8110409, 19.4970589, -51.3534126, 51.3370972
21: -43.6985703, 18.6536274, -43.6083450, 18.5978184, -62.2963867, 62.2619705
22: -51.3890419, 17.5432835, -51.2433853, 17.5073948, -68.8964386, 68.7866669
23: -31.9556675, 23.1952400, -31.8975124, 23.1253967, -55.0810623, 55.0927505
24: -44.3224335, 22.8861122, -44.2064209, 22.8460655, -67.1684952, 67.0925293
25: -34.0317535, 25.6140099, -33.9726410, 25.5746078, -59.6063614, 59.5866508
26: -49.8912277, 33.9370575, -49.7825050, 33.8915253, -83.7827530, 83.7195587
27: -47.8245468, 21.3811073, -47.6543770, 21.3153934, -69.1399384, 69.0354767
28: -33.9166489, 23.8850098, -33.8490944, 23.8469276, -57.7635689, 57.7341042
29: -57.2728729, 16.4558563, -57.1921501, 16.4244900, -73.6973648, 73.6480103
30: -42.2308884, 25.2944660, -42.1824799, 25.2243881, -67.4552765, 67.4769440
31: -39.6763573, 26.0938644, -39.6012726, 26.0506268, -65.7269821, 65.6951370
32: -44.9213982, 22.9446335, -44.8107185, 22.9009705, -67.8223724, 67.7553558
33: -59.6006546, 38.0872307, -59.3651695, 37.9840927, -97.5847473, 97.4524002
34: -52.2772102, 25.4472046, -52.1259308, 25.3762550, -77.6534653, 77.5731354
35: -53.5759811, 29.2987671, -53.3585510, 29.2049236, -82.7809067, 82.6573181
36: -53.2658081, 28.9235039, -53.0438385, 28.8350525, -82.1008606, 81.9673386
37: -71.7139740, 29.5371513, -71.3983383, 29.4282379, -101.1422119, 100.9354858
38: -62.8361206, 33.8125038, -62.5617218, 33.7073402, -96.5434570, 96.3742218
39: -72.9729309, 33.8969269, -72.6636047, 33.7925720, -106.7654953, 106.5605240
40: -59.3174019, 29.4853172, -59.0660477, 29.3971844, -88.7145844, 88.5513611
41: -41.8897934, 27.6819916, -41.7423172, 27.6370583, -69.5268555, 69.4243088
42: -29.5113716, 24.2259617, -29.4068413, 24.0917339, -53.6031036, 53.6328049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=46, inp2_unstable=46, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.9099959, upper bound: 51.0664844
time: 79.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.9099959, upper bound: 51.3019315
time: 83.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -62.1742134, 22.0076885, -62.1070328, 21.9833565, -84.1575623, 84.1147232
1: -34.7014847, 24.8462486, -34.6662979, 24.8186951, -59.5201721, 59.5125427
2: -27.8346004, 22.9897385, -27.7785301, 22.9606152, -50.7952156, 50.7682648
3: -32.7226753, 29.1355915, -32.6510124, 29.0885868, -61.8112602, 61.7866020
4: -34.6812057, 27.2566681, -34.6227417, 27.2235603, -61.9047661, 61.8794098
5: -30.8412342, 32.4152985, -30.7918472, 32.3773613, -63.2185974, 63.2071457
6: -37.9389343, 29.4791756, -37.8502655, 29.4479408, -67.3868713, 67.3294373
7: -40.0680199, 31.7043133, -40.0233078, 31.6742630, -71.7422791, 71.7276230
8: -38.6973419, 33.1311493, -38.6173477, 33.0929794, -71.7903214, 71.7484970
9: -29.9503651, 31.6250648, -29.8726425, 31.5027466, -61.4531097, 61.4977074
10: -43.7510681, 47.9551773, -43.5759315, 47.6310349, -91.3821030, 91.5311127
11: -44.8568077, 26.0509644, -44.7699699, 25.8949356, -70.7517395, 70.8209305
12: -42.2231636, 34.2694550, -42.1255112, 34.1251450, -76.3483124, 76.3949585
13: -45.5856476, 40.1147385, -45.5353470, 40.0579681, -85.6436157, 85.6500854
14: -77.7652740, 23.1454430, -77.6467590, 22.9862137, -100.7514648, 100.7922058
15: -37.0039749, 28.0781326, -36.9665375, 28.0050201, -65.0089874, 65.0446625
16: -48.1664848, 35.7574310, -48.0604515, 35.5888557, -83.7553406, 83.8178711
17: -77.0813904, 35.4160690, -76.9879608, 35.2535477, -112.3349380, 112.4040298
18: -40.5974426, 28.6491566, -40.5327377, 28.5885620, -69.1860046, 69.1818924
19: -30.9169674, 16.3336887, -30.8587399, 16.2833633, -47.2003326, 47.1924286
20: -31.8563538, 19.5260544, -31.8143826, 19.5083199, -51.3646736, 51.3404350
21: -43.6985703, 18.6536274, -43.6339951, 18.5928669, -62.2914352, 62.2876205
22: -51.3890419, 17.5432835, -51.2916603, 17.4958439, -68.8848877, 68.8349457
23: -31.9556675, 23.1952400, -31.9159794, 23.1453133, -55.1009827, 55.1112137
24: -44.3224335, 22.8861122, -44.2565536, 22.8487549, -67.1711807, 67.1426697
25: -34.0317535, 25.6140099, -33.9878006, 25.5784149, -59.6101646, 59.6018105
26: -49.8912277, 33.9370575, -49.8321838, 33.8416481, -83.7328796, 83.7692337
27: -47.8245468, 21.3811073, -47.6966820, 21.3191109, -69.1436539, 69.0777893
28: -33.9166489, 23.8850098, -33.8420258, 23.8444061, -57.7610474, 57.7270355
29: -57.2728729, 16.4558563, -57.2040901, 16.4110661, -73.6839371, 73.6599426
30: -42.2308884, 25.2944660, -42.1894760, 25.2499809, -67.4808655, 67.4839401
31: -39.6763573, 26.0938644, -39.6135521, 26.0302029, -65.7065582, 65.7074127
32: -44.9213982, 22.9446335, -44.8454666, 22.9142437, -67.8356323, 67.7901001
33: -59.6006546, 38.0872307, -59.4613762, 38.0028038, -97.6034546, 97.5486069
34: -52.2772102, 25.4472046, -52.1432571, 25.3771553, -77.6543655, 77.5904541
35: -53.5759811, 29.2987671, -53.4172249, 29.2149124, -82.7908936, 82.7159882
36: -53.2658081, 28.9235039, -53.1078835, 28.8537216, -82.1195297, 82.0313797
37: -71.7139740, 29.5371513, -71.5810089, 29.4765282, -101.1905060, 101.1181641
38: -62.8361206, 33.8125038, -62.6427002, 33.7295647, -96.5656738, 96.4552002
39: -72.9729309, 33.8969269, -72.8277740, 33.8352280, -106.8081589, 106.7247009
40: -59.3174019, 29.4853172, -59.1870232, 29.4345798, -88.7519760, 88.6723404
41: -41.8897934, 27.6819916, -41.7987480, 27.6430702, -69.5328598, 69.4807434
42: -29.5113716, 24.2259617, -29.4429340, 24.1459618, -53.6573257, 53.6688957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=46, inp2_unstable=46, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.9099959, upper bound: 51.0696808
time: 89.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.9099959, upper bound: 51.3019687
time: 77.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -62.1742134, 22.0076885, -62.1742134, 22.0076885, -84.1818924, 84.1819000
1: -34.7014847, 24.8462486, -34.7014847, 24.8462486, -59.5477295, 59.5477295
2: -27.8346004, 22.9897385, -27.8346004, 22.9897385, -50.8243408, 50.8243370
3: -32.7226753, 29.1355915, -32.7226753, 29.1355915, -61.8582687, 61.8582687
4: -34.6812057, 27.2566681, -34.6812057, 27.2566681, -61.9378738, 61.9378738
5: -30.8412342, 32.4152985, -30.8412342, 32.4152985, -63.2565308, 63.2565308
6: -37.9389343, 29.4791756, -37.9389343, 29.4791756, -67.4180984, 67.4181061
7: -40.0680199, 31.7043133, -40.0680199, 31.7043133, -71.7723312, 71.7723312
8: -38.6973419, 33.1311493, -38.6973419, 33.1311493, -71.8284912, 71.8284912
9: -29.9503651, 31.6250648, -29.9503651, 31.6250648, -61.5754318, 61.5754318
10: -43.7510681, 47.9551773, -43.7510681, 47.9551773, -91.7062454, 91.7062454
11: -44.8568077, 26.0509644, -44.8568077, 26.0509644, -70.9077606, 70.9077682
12: -42.2231636, 34.2694550, -42.2231636, 34.2694550, -76.4926147, 76.4926147
13: -45.5856476, 40.1147385, -45.5856476, 40.1147385, -85.7003860, 85.7003860
14: -77.7652740, 23.1454430, -77.7652740, 23.1454430, -100.9107208, 100.9107208
15: -37.0039749, 28.0781326, -37.0039749, 28.0781326, -65.0821075, 65.0820999
16: -48.1664848, 35.7574310, -48.1664848, 35.7574310, -83.9239120, 83.9239044
17: -77.0813904, 35.4160690, -77.0813904, 35.4160690, -112.4974594, 112.4974594
18: -40.5974426, 28.6491566, -40.5974426, 28.6491566, -69.2465973, 69.2465973
19: -30.9169674, 16.3336887, -30.9169674, 16.3336887, -47.2506561, 47.2506561
20: -31.8563538, 19.5260544, -31.8563538, 19.5260544, -51.3824081, 51.3824081
21: -43.6985703, 18.6536274, -43.6985703, 18.6536274, -62.3521957, 62.3521957
22: -51.3890419, 17.5432835, -51.3890419, 17.5432835, -68.9323273, 68.9323273
23: -31.9556675, 23.1952400, -31.9556675, 23.1952400, -55.1509094, 55.1509018
24: -44.3224335, 22.8861122, -44.3224335, 22.8861122, -67.2085419, 67.2085419
25: -34.0317535, 25.6140099, -34.0317535, 25.6140099, -59.6457596, 59.6457634
26: -49.8912277, 33.9370575, -49.8912277, 33.9370575, -83.8282776, 83.8282776
27: -47.8245468, 21.3811073, -47.8245468, 21.3811073, -69.2056427, 69.2056427
28: -33.9166489, 23.8850098, -33.9166489, 23.8850098, -57.8016586, 57.8016548
29: -57.2728729, 16.4558563, -57.2728729, 16.4558563, -73.7287292, 73.7287292
30: -42.2308884, 25.2944660, -42.2308884, 25.2944660, -67.5253448, 67.5253525
31: -39.6763573, 26.0938644, -39.6763573, 26.0938644, -65.7702179, 65.7702179
32: -44.9213982, 22.9446335, -44.9213982, 22.9446335, -67.8660278, 67.8660278
33: -59.6006546, 38.0872307, -59.6006546, 38.0872307, -97.6878815, 97.6878815
34: -52.2772102, 25.4472046, -52.2772102, 25.4472046, -77.7244110, 77.7244110
35: -53.5759811, 29.2987671, -53.5759811, 29.2987671, -82.8747482, 82.8747482
36: -53.2658081, 28.9235039, -53.2658081, 28.9235039, -82.1893158, 82.1893158
37: -71.7139740, 29.5371513, -71.7139740, 29.5371513, -101.2511292, 101.2511292
38: -62.8361206, 33.8125038, -62.8361206, 33.8125038, -96.6486206, 96.6486206
39: -72.9729309, 33.8969269, -72.9729309, 33.8969269, -106.8698578, 106.8698578
40: -59.3174019, 29.4853172, -59.3174019, 29.4853172, -88.8027191, 88.8027191
41: -41.8897934, 27.6819916, -41.8897934, 27.6819916, -69.5717850, 69.5717850
42: -29.5113716, 24.2259617, -29.5113716, 24.2259617, -53.7373314, 53.7373352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=46, inp2_unstable=46, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1655
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 1770
type: A, layer: 1, pos: 1770

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1653

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.9099959, upper bound: 51.0698384
time: 66.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.9099959, upper bound: 51.0672436
time: 82.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 151.26 seconds
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 151.26
Output dim: 10, lower bound: -50.9099959, upper bound: 51.0663932
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 151.26
Output dim: 10, lower bound: -50.9099959, upper bound: 51.3019303
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 151.26
Output dim: 10, lower bound: -50.9099959, upper bound: 51.0664844
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 151.26
Output dim: 10, lower bound: -50.9099959, upper bound: 51.3019315
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 151.26
Output dim: 10, lower bound: -50.9099959, upper bound: 51.0696808
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 151.26
Output dim: 10, lower bound: -50.9099959, upper bound: 51.3019687
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 151.26
Output dim: 10, lower bound: -50.9099959, upper bound: 51.0698384
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 151.26
Output dim: 10, lower bound: -50.9099959, upper bound: 51.0672436

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -62.1729279, 22.0060177, -62.0391388, 21.9375916, -84.1105194, 84.0451584
1: -34.6991196, 24.8451900, -34.6339264, 24.7580395, -59.4571609, 59.4791183
2: -27.8339081, 22.9875889, -27.7396545, 22.9242992, -50.7582054, 50.7272415
3: -32.7198982, 29.1347408, -32.6191597, 28.9807129, -61.7006111, 61.7538948
4: -34.6792297, 27.2558651, -34.5544815, 27.1762276, -61.8554573, 61.8103409
5: -30.8402157, 32.4133911, -30.7425308, 32.2609634, -63.1011810, 63.1559219
6: -37.9365845, 29.4785309, -37.7571144, 29.4157791, -67.3523560, 67.2356415
7: -40.0671425, 31.7029343, -39.9722977, 31.5503960, -71.6175385, 71.6752319
8: -38.6956558, 33.1294632, -38.5400276, 33.0356369, -71.7312927, 71.6694946
9: -29.9493179, 31.6235123, -29.7754574, 31.3227882, -61.2720985, 61.3989716
10: -43.7500763, 47.9498024, -43.3856468, 47.2216530, -90.9717255, 91.3354492
11: -44.8556442, 26.0496864, -44.6585464, 25.6933384, -70.5489807, 70.7082291
12: -42.2223473, 34.2676086, -42.0500565, 34.0593491, -76.2816849, 76.3176651
13: -45.5842133, 40.1136436, -45.4872398, 39.9785538, -85.5627670, 85.6008835
14: -77.7639694, 23.1423550, -77.5319519, 22.8092251, -100.5731964, 100.6743011
15: -37.0022316, 28.0768013, -36.9235153, 27.9086227, -64.9108429, 65.0003204
16: -48.1647987, 35.7548904, -47.9167709, 35.3365250, -83.5013199, 83.6716614
17: -77.0803680, 35.4127922, -76.8369751, 35.0026207, -112.0829926, 112.2497635
18: -40.5964317, 28.6446819, -40.4313736, 28.5317478, -69.1281815, 69.0760574
19: -30.9161053, 16.3297386, -30.7827320, 16.2366219, -47.1527252, 47.1124649
20: -31.8554802, 19.5254250, -31.7691708, 19.4793892, -51.3348694, 51.2945938
21: -43.6975784, 18.6492004, -43.5438004, 18.5362225, -62.2337952, 62.1930008
22: -51.3876877, 17.5421219, -51.1490135, 17.4609070, -68.8485870, 68.6911316
23: -31.9547729, 23.1906681, -31.8587532, 23.0800133, -55.0347862, 55.0494232
24: -44.3211746, 22.8830948, -44.1461182, 22.8122196, -67.1333923, 67.0292130
25: -34.0308380, 25.6104488, -33.9310684, 25.5388832, -59.5697174, 59.5415192
26: -49.8900337, 33.9330025, -49.7229919, 33.7898483, -83.6798706, 83.6559906
27: -47.8222427, 21.3803520, -47.5271263, 21.2532368, -69.0754700, 68.9074707
28: -33.9148636, 23.8843269, -33.7742004, 23.8063202, -57.7211838, 57.6585236
29: -57.2709541, 16.4546089, -57.1277733, 16.3793221, -73.6502762, 73.5823822
30: -42.2295647, 25.2916260, -42.1418571, 25.1801033, -67.4096680, 67.4334869
31: -39.6752319, 26.0899200, -39.5379524, 25.9871712, -65.6623993, 65.6278687
32: -44.9197998, 22.9436760, -44.7383614, 22.8708706, -67.7906723, 67.6820374
33: -59.5978127, 38.0865669, -59.2264977, 37.8997078, -97.4975204, 97.3130646
34: -52.2743645, 25.4464645, -51.9918785, 25.3063030, -77.5806656, 77.4383392
35: -53.5729523, 29.2983494, -53.1999359, 29.1210060, -82.6939545, 82.4982834
36: -53.2624359, 28.9230442, -52.8863144, 28.7650795, -82.0275116, 81.8093567
37: -71.7106552, 29.5364799, -71.2670212, 29.3673897, -101.0780334, 100.8034973
38: -62.8326988, 33.8117027, -62.3673096, 33.6244812, -96.4571838, 96.1790085
39: -72.9697342, 33.8963966, -72.5198135, 33.7304993, -106.7002258, 106.4162140
40: -59.3143234, 29.4848824, -58.9372139, 29.3479347, -88.6622620, 88.4220963
41: -41.8874130, 27.6813660, -41.6519318, 27.5980034, -69.4854126, 69.3332977
42: -29.5101433, 24.2251816, -29.3390007, 24.0118370, -53.5219803, 53.5641823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=45, inp2_unstable=46, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1654

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.5579133, upper bound: 51.2627223
time: 76.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.5579133, upper bound: 51.2964302
time: 77.10 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -62.1729279, 22.0060177, -62.1056900, 21.9615192, -84.1344376, 84.1117020
1: -34.6991196, 24.8451900, -34.6688538, 24.7852097, -59.4843292, 59.5140457
2: -27.8339081, 22.9875889, -27.7957802, 22.9529572, -50.7868614, 50.7833710
3: -32.7198982, 29.1347408, -32.6946487, 29.0286827, -61.7485809, 61.8293915
4: -34.6792297, 27.2558651, -34.6107254, 27.2065353, -61.8857651, 61.8665924
5: -30.8402157, 32.4133911, -30.7931404, 32.3002663, -63.1404800, 63.2065277
6: -37.9365845, 29.4785309, -37.8451004, 29.4464340, -67.3830185, 67.3236313
7: -40.0671425, 31.7029343, -40.0183868, 31.5796604, -71.6467896, 71.7213211
8: -38.6956558, 33.1294632, -38.6172409, 33.0737190, -71.7693787, 71.7467041
9: -29.9493179, 31.6235123, -29.8526783, 31.4453640, -61.3946762, 61.4761848
10: -43.7500763, 47.9498024, -43.5609436, 47.5464211, -91.2964935, 91.5107422
11: -44.8556442, 26.0496864, -44.7445831, 25.8495560, -70.7052002, 70.7942657
12: -42.2223473, 34.2676086, -42.1469879, 34.2026062, -76.4249573, 76.4145966
13: -45.5842133, 40.1136436, -45.5368423, 40.0347290, -85.6189423, 85.6504822
14: -77.7639694, 23.1423550, -77.6503830, 22.9686127, -100.7325821, 100.7927399
15: -37.0022316, 28.0768013, -36.9601631, 27.9808922, -64.9831238, 65.0369644
16: -48.1647987, 35.7548904, -48.0225906, 35.5055275, -83.6703262, 83.7774734
17: -77.0803680, 35.4127922, -76.9287109, 35.1648407, -112.2452087, 112.3415070
18: -40.5964317, 28.6446819, -40.4957123, 28.5922756, -69.1887054, 69.1403961
19: -30.9161053, 16.3297386, -30.8412743, 16.2871571, -47.2032623, 47.1710129
20: -31.8554802, 19.5254250, -31.8110409, 19.4970589, -51.3525391, 51.3364639
21: -43.6975784, 18.6492004, -43.6083450, 18.5978184, -62.2953949, 62.2575378
22: -51.3876877, 17.5421219, -51.2433853, 17.5073948, -68.8950806, 68.7855072
23: -31.9547729, 23.1906681, -31.8975124, 23.1253967, -55.0801697, 55.0881805
24: -44.3211746, 22.8830948, -44.2064209, 22.8460655, -67.1672287, 67.0895157
25: -34.0308380, 25.6104488, -33.9726410, 25.5746078, -59.6054459, 59.5830917
26: -49.8900337, 33.9330025, -49.7825050, 33.8915253, -83.7815552, 83.7155075
27: -47.8222427, 21.3803520, -47.6543770, 21.3153934, -69.1376343, 69.0347290
28: -33.9148636, 23.8843269, -33.8490944, 23.8469276, -57.7617912, 57.7334213
29: -57.2709541, 16.4546089, -57.1921501, 16.4244900, -73.6954422, 73.6467590
30: -42.2295647, 25.2916260, -42.1824799, 25.2243881, -67.4539490, 67.4741058
31: -39.6752319, 26.0899200, -39.6012726, 26.0506268, -65.7258606, 65.6911926
32: -44.9197998, 22.9436760, -44.8107185, 22.9009705, -67.8207550, 67.7543945
33: -59.5978127, 38.0865669, -59.3651695, 37.9840927, -97.5819092, 97.4517288
34: -52.2743645, 25.4464645, -52.1259308, 25.3762550, -77.6506119, 77.5723953
35: -53.5729523, 29.2983494, -53.3585510, 29.2049236, -82.7778778, 82.6568985
36: -53.2624359, 28.9230442, -53.0438385, 28.8350525, -82.0974808, 81.9668808
37: -71.7106552, 29.5364799, -71.3983383, 29.4282379, -101.1388855, 100.9348145
38: -62.8326988, 33.8117027, -62.5617218, 33.7073402, -96.5400391, 96.3734207
39: -72.9697342, 33.8963966, -72.6636047, 33.7925720, -106.7623062, 106.5599976
40: -59.3143234, 29.4848824, -59.0660477, 29.3971844, -88.7115021, 88.5509262
41: -41.8874130, 27.6813660, -41.7423172, 27.6370583, -69.5244751, 69.4236755
42: -29.5101433, 24.2251816, -29.4068413, 24.0917339, -53.6018753, 53.6320229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=45, inp2_unstable=46, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1654

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.5579133, upper bound: 51.2641326
time: 57.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.5579133, upper bound: 51.2964314
time: 97.99 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -62.1729279, 22.0060177, -62.1070328, 21.9833565, -84.1562805, 84.1130447
1: -34.6991196, 24.8451900, -34.6662979, 24.8186951, -59.5178146, 59.5114822
2: -27.8339081, 22.9875889, -27.7785301, 22.9606152, -50.7945213, 50.7661133
3: -32.7198982, 29.1347408, -32.6510124, 29.0885868, -61.8084831, 61.7857513
4: -34.6792297, 27.2558651, -34.6227417, 27.2235603, -61.9027901, 61.8786087
5: -30.8402157, 32.4133911, -30.7918472, 32.3773613, -63.2175751, 63.2052383
6: -37.9365845, 29.4785309, -37.8502655, 29.4479408, -67.3845062, 67.3287964
7: -40.0671425, 31.7029343, -40.0233078, 31.6742630, -71.7413940, 71.7262421
8: -38.6956558, 33.1294632, -38.6173477, 33.0929794, -71.7886353, 71.7468109
9: -29.9493179, 31.6235123, -29.8726425, 31.5027466, -61.4520645, 61.4961548
10: -43.7500763, 47.9498024, -43.5759315, 47.6310349, -91.3811111, 91.5257339
11: -44.8556442, 26.0496864, -44.7699699, 25.8949356, -70.7505798, 70.8196564
12: -42.2223473, 34.2676086, -42.1255112, 34.1251450, -76.3474884, 76.3931198
13: -45.5842133, 40.1136436, -45.5353470, 40.0579681, -85.6421814, 85.6489868
14: -77.7639694, 23.1423550, -77.6467590, 22.9862137, -100.7501831, 100.7891159
15: -37.0022316, 28.0768013, -36.9665375, 28.0050201, -65.0072479, 65.0433350
16: -48.1647987, 35.7548904, -48.0604515, 35.5888557, -83.7536545, 83.8153229
17: -77.0803680, 35.4127922, -76.9879608, 35.2535477, -112.3339081, 112.4007492
18: -40.5964317, 28.6446819, -40.5327377, 28.5885620, -69.1849976, 69.1774216
19: -30.9161053, 16.3297386, -30.8587399, 16.2833633, -47.1994705, 47.1884766
20: -31.8554802, 19.5254250, -31.8143826, 19.5083199, -51.3638000, 51.3398056
21: -43.6975784, 18.6492004, -43.6339951, 18.5928669, -62.2904434, 62.2831955
22: -51.3876877, 17.5421219, -51.2916603, 17.4958439, -68.8835297, 68.8337784
23: -31.9547729, 23.1906681, -31.9159794, 23.1453133, -55.1000862, 55.1066475
24: -44.3211746, 22.8830948, -44.2565536, 22.8487549, -67.1699295, 67.1396484
25: -34.0308380, 25.6104488, -33.9878006, 25.5784149, -59.6092529, 59.5982475
26: -49.8900337, 33.9330025, -49.8321838, 33.8416481, -83.7316818, 83.7651825
27: -47.8222427, 21.3803520, -47.6966820, 21.3191109, -69.1413498, 69.0770264
28: -33.9148636, 23.8843269, -33.8420258, 23.8444061, -57.7592697, 57.7263527
29: -57.2709541, 16.4546089, -57.2040901, 16.4110661, -73.6820221, 73.6586990
30: -42.2295647, 25.2916260, -42.1894760, 25.2499809, -67.4795456, 67.4811020
31: -39.6752319, 26.0899200, -39.6135521, 26.0302029, -65.7054367, 65.7034683
32: -44.9197998, 22.9436760, -44.8454666, 22.9142437, -67.8340454, 67.7891388
33: -59.5978127, 38.0865669, -59.4613762, 38.0028038, -97.6006165, 97.5479431
34: -52.2743645, 25.4464645, -52.1432571, 25.3771553, -77.6515198, 77.5897217
35: -53.5729523, 29.2983494, -53.4172249, 29.2149124, -82.7878571, 82.7155762
36: -53.2624359, 28.9230442, -53.1078835, 28.8537216, -82.1161499, 82.0309296
37: -71.7106552, 29.5364799, -71.5810089, 29.4765282, -101.1871796, 101.1174927
38: -62.8326988, 33.8117027, -62.6427002, 33.7295647, -96.5622559, 96.4543991
39: -72.9697342, 33.8963966, -72.8277740, 33.8352280, -106.8049622, 106.7241669
40: -59.3143234, 29.4848824, -59.1870232, 29.4345798, -88.7489014, 88.6719055
41: -41.8874130, 27.6813660, -41.7987480, 27.6430702, -69.5304871, 69.4801178
42: -29.5101433, 24.2251816, -29.4429340, 24.1459618, -53.6560974, 53.6681137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=45, inp2_unstable=46, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1654

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.5579133, upper bound: 51.2654848
time: 78.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.9127056, upper bound: 51.2964558
time: 85.38 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 166.26 seconds
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 166.26
Output dim: 10, lower bound: -50.5579133, upper bound: 51.2627223
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 166.26
Output dim: 10, lower bound: -50.5579133, upper bound: 51.2964302
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 166.26
Output dim: 10, lower bound: -50.5579133, upper bound: 51.2641326
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 166.26
Output dim: 10, lower bound: -50.5579133, upper bound: 51.2964314
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 166.26
Output dim: 10, lower bound: -50.5579133, upper bound: 51.2654848
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 166.26
Output dim: 10, lower bound: -50.9127056, upper bound: 51.2964558

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -62.1622810, 21.9989758, -61.9647446, 21.8829536, -84.0452347, 83.9637222
1: -34.6928101, 24.8378410, -34.5866089, 24.7040710, -59.3968811, 59.4244499
2: -27.8270340, 22.9819698, -27.6894722, 22.8819656, -50.7089996, 50.6714401
3: -32.7005386, 29.1286011, -32.4749603, 28.9436874, -61.6442184, 61.6035614
4: -34.6706924, 27.2507706, -34.4910507, 27.1471252, -61.8178177, 61.7418213
5: -30.8298225, 32.4072533, -30.6677341, 32.2209244, -63.0507431, 63.0749893
6: -37.9183807, 29.4750786, -37.6187057, 29.3936462, -67.3120270, 67.0937805
7: -40.0605240, 31.6979771, -39.9238281, 31.5136528, -71.5741653, 71.6217957
8: -38.6861038, 33.1221161, -38.4688950, 32.9803314, -71.6664352, 71.5910034
9: -29.9420185, 31.6130733, -29.7255707, 31.2415771, -61.1835938, 61.3386459
10: -43.7408295, 47.8899765, -43.3230743, 46.7408257, -90.4816513, 91.2130508
11: -44.8489532, 26.0168381, -44.6157837, 25.4297638, -70.2787094, 70.6326141
12: -42.2159004, 34.2437401, -42.0046692, 33.8714523, -76.0873566, 76.2484055
13: -45.5698433, 40.1052170, -45.3771057, 39.9303665, -85.5002060, 85.4823227
14: -77.7534180, 23.1037788, -77.4569092, 22.4992008, -100.2526169, 100.5606842
15: -36.9912567, 28.0696049, -36.8440018, 27.8567104, -64.8479691, 64.9136047
16: -48.1522903, 35.7285423, -47.8383636, 35.1267967, -83.2790833, 83.5669098
17: -77.0740280, 35.3744583, -76.7937164, 34.6965332, -111.7705536, 112.1681747
18: -40.5891762, 28.6186142, -40.4007149, 28.3437119, -68.9328842, 69.0193253
19: -30.9089279, 16.3172379, -30.7343998, 16.1397018, -47.0486298, 47.0516357
20: -31.8500500, 19.5196724, -31.7323570, 19.4359074, -51.2859497, 51.2520294
21: -43.6903000, 18.6328678, -43.4935379, 18.4084740, -62.0987701, 62.1264038
22: -51.3741760, 17.5350628, -51.0488319, 17.4206886, -68.7948608, 68.5838928
23: -31.9487724, 23.1754608, -31.8202267, 22.9627934, -54.9115677, 54.9956894
24: -44.3145256, 22.8727169, -44.1184998, 22.7365265, -67.0510559, 66.9912186
25: -34.0251961, 25.5985966, -33.8986969, 25.4487705, -59.4739685, 59.4972916
26: -49.8827477, 33.9107094, -49.6742744, 33.6197891, -83.5025330, 83.5849838
27: -47.8029900, 21.3761673, -47.3758698, 21.2242947, -69.0272827, 68.7520370
28: -33.8996048, 23.8803596, -33.6554108, 23.7897758, -57.6893768, 57.5357704
29: -57.2595177, 16.4460678, -57.0430794, 16.3271599, -73.5866776, 73.4891510
30: -42.2233238, 25.2773819, -42.1004601, 25.0747356, -67.2980576, 67.3778381
31: -39.6658974, 26.0688744, -39.4786682, 25.8251877, -65.4910889, 65.5475464
32: -44.8977509, 22.9404278, -44.5687103, 22.8518372, -67.7495880, 67.5091400
33: -59.5737877, 38.0814934, -59.0376053, 37.8675461, -97.4413300, 97.1190948
34: -52.2467117, 25.4412441, -51.7730064, 25.2692261, -77.5159302, 77.2142487
35: -53.5408554, 29.2942963, -52.9438934, 29.0939617, -82.6348190, 82.2381897
36: -53.2262917, 28.9198513, -52.5970497, 28.7452354, -81.9715118, 81.5168991
37: -71.6930161, 29.5318069, -71.1335602, 29.3413010, -101.0343170, 100.6653671
38: -62.8011703, 33.8060379, -62.1184998, 33.5851021, -96.3862534, 95.9245377
39: -72.9452667, 33.8925400, -72.3324585, 33.7080002, -106.6532669, 106.2249908
40: -59.2908554, 29.4820595, -58.7551689, 29.3301716, -88.6210251, 88.2372284
41: -41.8671417, 27.6775398, -41.4938049, 27.5721054, -69.4392471, 69.1713409
42: -29.5046692, 24.2166176, -29.3053207, 23.9464397, -53.4511070, 53.5219383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=45, inp2_unstable=45, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.5024184, upper bound: 51.0347023
time: 67.25 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.5024184, upper bound: 51.0347023
time: 74.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -62.1729279, 22.0060177, -62.0382042, 21.9366226, -84.1095428, 84.0442200
1: -34.6991196, 24.8451900, -34.6332970, 24.7575302, -59.4566460, 59.4784851
2: -27.8339081, 22.9875889, -27.7390900, 22.9238739, -50.7577820, 50.7266731
3: -32.7198982, 29.1347408, -32.6178398, 28.9803257, -61.7002258, 61.7525749
4: -34.6792297, 27.2558651, -34.5537262, 27.1759109, -61.8551407, 61.8095894
5: -30.8402157, 32.4133911, -30.7419243, 32.2606125, -63.1008301, 63.1553154
6: -37.9365845, 29.4785309, -37.7563019, 29.4155350, -67.3521194, 67.2348328
7: -40.0671425, 31.7029343, -39.9717674, 31.5501404, -71.6172791, 71.6746979
8: -38.6956558, 33.1294632, -38.5394516, 33.0350952, -71.7307510, 71.6689148
9: -29.9493179, 31.6235123, -29.7749977, 31.3222809, -61.2715988, 61.3985100
10: -43.7500763, 47.9498024, -43.3851242, 47.2195511, -90.9696274, 91.3349228
11: -44.8556442, 26.0496864, -44.6580086, 25.6921349, -70.5477753, 70.7076950
12: -42.2223473, 34.2676086, -42.0497551, 34.0583725, -76.2807159, 76.3173676
13: -45.5842133, 40.1136436, -45.4853668, 39.9780045, -85.5622177, 85.5990067
14: -77.7639694, 23.1423550, -77.5313721, 22.8079014, -100.5718689, 100.6737289
15: -37.0022316, 28.0768013, -36.9221001, 27.9081955, -64.9104309, 64.9989014
16: -48.1647987, 35.7548904, -47.9159927, 35.3353348, -83.5001297, 83.6708832
17: -77.0803680, 35.4127922, -76.8365402, 35.0012970, -112.0816498, 112.2493286
18: -40.5964317, 28.6446819, -40.4308853, 28.5307121, -69.1271439, 69.0755692
19: -30.9161053, 16.3297386, -30.7823086, 16.2358284, -47.1519279, 47.1120453
20: -31.8554802, 19.5254250, -31.7688770, 19.4787693, -51.3342438, 51.2943039
21: -43.6975784, 18.6492004, -43.5433693, 18.5354424, -62.2330208, 62.1925659
22: -51.3876877, 17.5421219, -51.1477165, 17.4604378, -68.8481293, 68.6898346
23: -31.9547729, 23.1906681, -31.8584042, 23.0790997, -55.0338745, 55.0490723
24: -44.3211746, 22.8830948, -44.1456146, 22.8105984, -67.1317673, 67.0287094
25: -34.0308380, 25.6104488, -33.9306412, 25.5378571, -59.5686951, 59.5410919
26: -49.8900337, 33.9330025, -49.7225380, 33.7885895, -83.6786194, 83.6555328
27: -47.8222427, 21.3803520, -47.5263138, 21.2529259, -69.0751572, 68.9066620
28: -33.9148636, 23.8843269, -33.7728958, 23.8060532, -57.7209167, 57.6572151
29: -57.2709541, 16.4546089, -57.1263123, 16.3788338, -73.6497803, 73.5809174
30: -42.2295647, 25.2916260, -42.1413879, 25.1794014, -67.4089584, 67.4330139
31: -39.6752319, 26.0899200, -39.5374107, 25.9859486, -65.6611786, 65.6273270
32: -44.9197998, 22.9436760, -44.7373314, 22.8704586, -67.7902527, 67.6809998
33: -59.5978127, 38.0865669, -59.2254448, 37.8993645, -97.4971771, 97.3120117
34: -52.2743645, 25.4464645, -51.9907875, 25.3059616, -77.5803223, 77.4372559
35: -53.5729523, 29.2983494, -53.1986961, 29.1207466, -82.6936951, 82.4970398
36: -53.2624359, 28.9230442, -52.8849869, 28.7648354, -82.0272675, 81.8080292
37: -71.7106552, 29.5364799, -71.2660522, 29.3671494, -101.0778046, 100.8025284
38: -62.8326988, 33.8117027, -62.3660316, 33.6241646, -96.4568634, 96.1777344
39: -72.9697342, 33.8963966, -72.5186768, 33.7302742, -106.6999969, 106.4150696
40: -59.3143234, 29.4848824, -58.9361801, 29.3477516, -88.6620789, 88.4210587
41: -41.8874130, 27.6813660, -41.6510849, 27.5977364, -69.4851532, 69.3324432
42: -29.5101433, 24.2251816, -29.3386536, 24.0112915, -53.5214272, 53.5638351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=45, inp2_unstable=45, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.5024184, upper bound: 51.0628205
time: 77.22 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.5024184, upper bound: 51.2914842
time: 79.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -62.1622810, 21.9989758, -62.0308800, 21.9068737, -84.0691528, 84.0298538
1: -34.6928101, 24.8378410, -34.6214790, 24.7309666, -59.4237747, 59.4593201
2: -27.8270340, 22.9819698, -27.7444382, 22.9113388, -50.7383728, 50.7264061
3: -32.7005386, 29.1286011, -32.5492439, 28.9920330, -61.6925659, 61.6778412
4: -34.6706924, 27.2507706, -34.5463257, 27.1781769, -61.8488693, 61.7970963
5: -30.8298225, 32.4072533, -30.7163219, 32.2605591, -63.0903816, 63.1235733
6: -37.9183807, 29.4750786, -37.7062149, 29.4245949, -67.3429718, 67.1812897
7: -40.0605240, 31.6979771, -39.9695740, 31.5429802, -71.6035004, 71.6675491
8: -38.6861038, 33.1221161, -38.5449524, 33.0186005, -71.7047043, 71.6670685
9: -29.9420185, 31.6130733, -29.8035889, 31.3636208, -61.3056297, 61.4166603
10: -43.7408295, 47.8899765, -43.4987335, 47.0650635, -90.8058929, 91.3887100
11: -44.8489532, 26.0168381, -44.7024612, 25.5856037, -70.4345551, 70.7192993
12: -42.2159004, 34.2437401, -42.1018524, 34.0138664, -76.2297668, 76.3455963
13: -45.5698433, 40.1052170, -45.4268990, 39.9861221, -85.5559692, 85.5321121
14: -77.7534180, 23.1037788, -77.5757599, 22.6583271, -100.4117432, 100.6795349
15: -36.9912567, 28.0696049, -36.8803520, 27.9283180, -64.9195709, 64.9499512
16: -48.1522903, 35.7285423, -47.9450569, 35.2950211, -83.4472961, 83.6735992
17: -77.0740280, 35.3744583, -76.8854828, 34.8581810, -111.9322052, 112.2599335
18: -40.5891762, 28.6186142, -40.4654350, 28.4037476, -68.9929199, 69.0840454
19: -30.9089279, 16.3172379, -30.7931881, 16.1904335, -47.0993614, 47.1104202
20: -31.8500500, 19.5196724, -31.7740993, 19.4536457, -51.3036957, 51.2937698
21: -43.6903000, 18.6328678, -43.5582581, 18.4704475, -62.1607475, 62.1911240
22: -51.3741760, 17.5350628, -51.1426277, 17.4684067, -68.8425827, 68.6776886
23: -31.9487724, 23.1754608, -31.8595524, 23.0083122, -54.9570847, 55.0350113
24: -44.3145256, 22.8727169, -44.1787262, 22.7707443, -67.0852661, 67.0514450
25: -34.0251961, 25.5985966, -33.9397163, 25.4847298, -59.5099220, 59.5383148
26: -49.8827477, 33.9107094, -49.7344398, 33.7200241, -83.6027679, 83.6451492
27: -47.8029900, 21.3761673, -47.5030403, 21.2873802, -69.0903702, 68.8792114
28: -33.8996048, 23.8803596, -33.7301407, 23.8311386, -57.7307396, 57.6105003
29: -57.2595177, 16.4460678, -57.1071205, 16.3718834, -73.6313934, 73.5531921
30: -42.2233238, 25.2773819, -42.1412277, 25.1180458, -67.3413696, 67.4186096
31: -39.6658974, 26.0688744, -39.5426979, 25.8880043, -65.5538940, 65.6115723
32: -44.8977509, 22.9404278, -44.6409073, 22.8822594, -67.7800140, 67.5813293
33: -59.5737877, 38.0814934, -59.1760979, 37.9522552, -97.5260468, 97.2575836
34: -52.2467117, 25.4412441, -51.9066772, 25.3396301, -77.5863419, 77.3479156
35: -53.5408554, 29.2942963, -53.1022873, 29.1780872, -82.7189407, 82.3965759
36: -53.2262917, 28.9198513, -52.7545090, 28.8156738, -82.0419617, 81.6743622
37: -71.6930161, 29.5318069, -71.2647705, 29.4027710, -101.0957870, 100.7965775
38: -62.8011703, 33.8060379, -62.3117752, 33.6684761, -96.4696503, 96.1178131
39: -72.9452667, 33.8925400, -72.4750061, 33.7704582, -106.7157135, 106.3675308
40: -59.2908554, 29.4820595, -58.8839340, 29.3795967, -88.6704483, 88.3659897
41: -41.8671417, 27.6775398, -41.5840836, 27.6115456, -69.4786835, 69.2616272
42: -29.5046692, 24.2166176, -29.3738098, 24.0258160, -53.5304871, 53.5904274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=45, inp2_unstable=45, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.5039438, upper bound: 51.0370435
time: 74.33 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.5039437, upper bound: 51.2578690
time: 68.36 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -62.1729279, 22.0060177, -62.1048088, 21.9605293, -84.1334534, 84.1108246
1: -34.6991196, 24.8451900, -34.6681824, 24.7847347, -59.4838524, 59.5133705
2: -27.8339081, 22.9875889, -27.7952156, 22.9525146, -50.7864227, 50.7827988
3: -32.7198982, 29.1347408, -32.6933022, 29.0283222, -61.7482147, 61.8280411
4: -34.6792297, 27.2558651, -34.6099739, 27.2062073, -61.8854370, 61.8658371
5: -30.8402157, 32.4133911, -30.7925320, 32.2999344, -63.1401520, 63.2059174
6: -37.9365845, 29.4785309, -37.8442917, 29.4462128, -67.3827896, 67.3228149
7: -40.0671425, 31.7029343, -40.0178680, 31.5793896, -71.6465302, 71.7208023
8: -38.6956558, 33.1294632, -38.6166420, 33.0731659, -71.7688217, 71.7461090
9: -29.9493179, 31.6235123, -29.8522263, 31.4448738, -61.3941803, 61.4757385
10: -43.7500763, 47.9498024, -43.5604248, 47.5443077, -91.2943878, 91.5102234
11: -44.8556442, 26.0496864, -44.7440414, 25.8483219, -70.7039642, 70.7937317
12: -42.2223473, 34.2676086, -42.1466599, 34.2016411, -76.4239883, 76.4142685
13: -45.5842133, 40.1136436, -45.5349426, 40.0341759, -85.6183929, 85.6485901
14: -77.7639694, 23.1423550, -77.6498260, 22.9672737, -100.7312317, 100.7921753
15: -37.0022316, 28.0768013, -36.9587288, 27.9804516, -64.9826736, 65.0355301
16: -48.1647987, 35.7548904, -48.0218430, 35.5043335, -83.6691284, 83.7767258
17: -77.0803680, 35.4127922, -76.9283066, 35.1634865, -112.2438507, 112.3410950
18: -40.5964317, 28.6446819, -40.4952087, 28.5912418, -69.1876755, 69.1398926
19: -30.9161053, 16.3297386, -30.8408508, 16.2863579, -47.2024612, 47.1705894
20: -31.8554802, 19.5254250, -31.8107433, 19.4964371, -51.3519173, 51.3361664
21: -43.6975784, 18.6492004, -43.6079254, 18.5970268, -62.2945976, 62.2571259
22: -51.3876877, 17.5421219, -51.2420731, 17.5069199, -68.8946075, 68.7841873
23: -31.9547729, 23.1906681, -31.8971634, 23.1244774, -55.0792503, 55.0878296
24: -44.3211746, 22.8830948, -44.2059250, 22.8444481, -67.1656189, 67.0890121
25: -34.0308380, 25.6104488, -33.9722214, 25.5735683, -59.6044083, 59.5826721
26: -49.8900337, 33.9330025, -49.7820663, 33.8902550, -83.7802887, 83.7150726
27: -47.8222427, 21.3803520, -47.6535759, 21.3150902, -69.1373291, 69.0339203
28: -33.9148636, 23.8843269, -33.8477936, 23.8466644, -57.7615280, 57.7321167
29: -57.2709541, 16.4546089, -57.1907043, 16.4240112, -73.6949615, 73.6453094
30: -42.2295647, 25.2916260, -42.1820145, 25.2236710, -67.4532242, 67.4736404
31: -39.6752319, 26.0899200, -39.6007347, 26.0493965, -65.7246246, 65.6906586
32: -44.9197998, 22.9436760, -44.8097076, 22.9005547, -67.8203430, 67.7533875
33: -59.5978127, 38.0865669, -59.3641357, 37.9837608, -97.5815735, 97.4506989
34: -52.2743645, 25.4464645, -52.1248398, 25.3759422, -77.6503067, 77.5713043
35: -53.5729523, 29.2983494, -53.3573036, 29.2046776, -82.7776337, 82.6556549
36: -53.2624359, 28.9230442, -53.0425110, 28.8348083, -82.0972443, 81.9655533
37: -71.7106552, 29.5364799, -71.3973694, 29.4279938, -101.1386490, 100.9338531
38: -62.8326988, 33.8117027, -62.5604324, 33.7070236, -96.5397186, 96.3721313
39: -72.9697342, 33.8963966, -72.6624527, 33.7923279, -106.7620621, 106.5588531
40: -59.3143234, 29.4848824, -59.0650063, 29.3970203, -88.7113342, 88.5498886
41: -41.8874130, 27.6813660, -41.7414742, 27.6367950, -69.5242081, 69.4228363
42: -29.5101433, 24.2251816, -29.4065056, 24.0911770, -53.6013184, 53.6316833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=45, inp2_unstable=45, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.5039446, upper bound: 51.0370444
time: 116.07 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.5039437, upper bound: 51.0370444
time: 72.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -62.1622810, 21.9989758, -62.0327263, 21.9283829, -84.0906525, 84.0317001
1: -34.6928101, 24.8378410, -34.6189270, 24.7643375, -59.4571457, 59.4567680
2: -27.8270340, 22.9819698, -27.7284946, 22.9180107, -50.7450447, 50.7104645
3: -32.7005386, 29.1286011, -32.5066109, 29.0510769, -61.7516174, 61.6352119
4: -34.6706924, 27.2507706, -34.5586014, 27.1947517, -61.8654442, 61.8093681
5: -30.8298225, 32.4072533, -30.7167664, 32.3365402, -63.1663628, 63.1240158
6: -37.9183807, 29.4750786, -37.7117882, 29.4258308, -67.3442078, 67.1868668
7: -40.0605240, 31.6979771, -39.9747276, 31.6371517, -71.6976776, 71.6727066
8: -38.6861038, 33.1221161, -38.5462227, 33.0375443, -71.7236481, 71.6683273
9: -29.9420185, 31.6130733, -29.8233643, 31.4217415, -61.3637581, 61.4364319
10: -43.7408295, 47.8899765, -43.5141449, 47.1505814, -90.8914108, 91.4041214
11: -44.8489532, 26.0168381, -44.7279320, 25.6316509, -70.4806061, 70.7447662
12: -42.2159004, 34.2437401, -42.0798187, 33.9372139, -76.1531143, 76.3235474
13: -45.5698433, 40.1052170, -45.4256020, 40.0098076, -85.5796356, 85.5308151
14: -77.7534180, 23.1037788, -77.5719223, 22.6761761, -100.4295959, 100.6757050
15: -36.9912567, 28.0696049, -36.8868484, 27.9524040, -64.9436646, 64.9564438
16: -48.1522903, 35.7285423, -47.9828033, 35.3796082, -83.5318985, 83.7113495
17: -77.0740280, 35.3744583, -76.9448090, 34.9476471, -112.0216751, 112.3192673
18: -40.5891762, 28.6186142, -40.5020752, 28.4004860, -68.9896622, 69.1206894
19: -30.9089279, 16.3172379, -30.8106594, 16.1868649, -47.0957870, 47.1278954
20: -31.8500500, 19.5196724, -31.7776661, 19.4648170, -51.3148651, 51.2973404
21: -43.6903000, 18.6328678, -43.5840492, 18.4654675, -62.1557655, 62.2169037
22: -51.3741760, 17.5350628, -51.1906242, 17.4558105, -68.8299866, 68.7256851
23: -31.9487724, 23.1754608, -31.8778172, 23.0281410, -54.9769058, 55.0532761
24: -44.3145256, 22.8727169, -44.2285156, 22.7729874, -67.0875092, 67.1012344
25: -34.0251961, 25.5985966, -33.9557037, 25.4882545, -59.5134506, 59.5542984
26: -49.8827477, 33.9107094, -49.7824707, 33.6716919, -83.5544357, 83.6931763
27: -47.8029900, 21.3761673, -47.5453796, 21.2906761, -69.0936661, 68.9215469
28: -33.8996048, 23.8803596, -33.7230644, 23.8275738, -57.7271690, 57.6034241
29: -57.2595177, 16.4460678, -57.1190605, 16.3589020, -73.6184235, 73.5651245
30: -42.2233238, 25.2773819, -42.1485176, 25.1443577, -67.3676834, 67.4259033
31: -39.6658974, 26.0688744, -39.5549164, 25.8684120, -65.5343094, 65.6237946
32: -44.8977509, 22.9404278, -44.6757126, 22.8952694, -67.7930222, 67.6161346
33: -59.5737877, 38.0814934, -59.2725449, 37.9708099, -97.5446014, 97.3540344
34: -52.2467117, 25.4412441, -51.9243965, 25.3404922, -77.5872040, 77.3656387
35: -53.5408554, 29.2942963, -53.1612701, 29.1882744, -82.7291260, 82.4555664
36: -53.2262917, 28.9198513, -52.8188286, 28.8344364, -82.0607224, 81.7386780
37: -71.6930161, 29.5318069, -71.4472961, 29.4506168, -101.1436310, 100.9791031
38: -62.8011703, 33.8060379, -62.3944397, 33.6907425, -96.4919052, 96.2004776
39: -72.9452667, 33.8925400, -72.6401062, 33.8128853, -106.7581482, 106.5326385
40: -59.2908554, 29.4820595, -59.0048294, 29.4168682, -88.7077255, 88.4868851
41: -41.8671417, 27.6775398, -41.6404266, 27.6171741, -69.4843140, 69.3179626
42: -29.5046692, 24.2166176, -29.4095554, 24.0804710, -53.5851364, 53.6261749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=45, inp2_unstable=45, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 630
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.5024184, upper bound: 51.0381817
time: 75.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.5024184, upper bound: 51.2592297
time: 68.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -62.1729279, 22.0060177, -62.1061134, 21.9823761, -84.1553040, 84.1121292
1: -34.6991196, 24.8451900, -34.6656532, 24.8182106, -59.5173302, 59.5108414
2: -27.8339081, 22.9875889, -27.7779560, 22.9601669, -50.7940674, 50.7655449
3: -32.7198982, 29.1347408, -32.6497002, 29.0882149, -61.8081055, 61.7844315
4: -34.6792297, 27.2558651, -34.6219864, 27.2232590, -61.9024887, 61.8778534
5: -30.8402157, 32.4133911, -30.7912540, 32.3770218, -63.2172394, 63.2046432
6: -37.9365845, 29.4785309, -37.8494720, 29.4477215, -67.3843079, 67.3280029
7: -40.0671425, 31.7029343, -40.0227928, 31.6739807, -71.7411194, 71.7257233
8: -38.6956558, 33.1294632, -38.6167526, 33.0924454, -71.7881012, 71.7462158
9: -29.9493179, 31.6235123, -29.8722000, 31.5022507, -61.4515610, 61.4957123
10: -43.7500763, 47.9498024, -43.5754051, 47.6289444, -91.3790131, 91.5252075
11: -44.8556442, 26.0496864, -44.7694321, 25.8937092, -70.7493515, 70.8191223
12: -42.2223473, 34.2676086, -42.1251831, 34.1241684, -76.3465118, 76.3927917
13: -45.5842133, 40.1136436, -45.5334511, 40.0574265, -85.6416321, 85.6470947
14: -77.7639694, 23.1423550, -77.6461792, 22.9848747, -100.7488403, 100.7885361
15: -37.0022316, 28.0768013, -36.9651031, 28.0045929, -65.0068207, 65.0419006
16: -48.1647987, 35.7548904, -48.0596886, 35.5876732, -83.7524719, 83.8145676
17: -77.0803680, 35.4127922, -76.9875107, 35.2521896, -112.3325577, 112.4002991
18: -40.5964317, 28.6446819, -40.5322418, 28.5875359, -69.1839676, 69.1769257
19: -30.9161053, 16.3297386, -30.8583183, 16.2825737, -47.1986732, 47.1880569
20: -31.8554802, 19.5254250, -31.8140850, 19.5077057, -51.3631821, 51.3395081
21: -43.6975784, 18.6492004, -43.6335831, 18.5920811, -62.2896500, 62.2827797
22: -51.3876877, 17.5421219, -51.2903442, 17.4953537, -68.8830414, 68.8324585
23: -31.9547729, 23.1906681, -31.9156380, 23.1443939, -55.0991669, 55.1063080
24: -44.3211746, 22.8830948, -44.2560463, 22.8471375, -67.1683121, 67.1391373
25: -34.0308380, 25.6104488, -33.9873886, 25.5773811, -59.6082191, 59.5978355
26: -49.8900337, 33.9330025, -49.8317108, 33.8403702, -83.7304001, 83.7647095
27: -47.8222427, 21.3803520, -47.6958733, 21.3187904, -69.1410370, 69.0762177
28: -33.9148636, 23.8843269, -33.8407249, 23.8441315, -57.7589951, 57.7250443
29: -57.2709541, 16.4546089, -57.2026062, 16.4105663, -73.6815186, 73.6572113
30: -42.2295647, 25.2916260, -42.1890182, 25.2492599, -67.4788208, 67.4806442
31: -39.6752319, 26.0899200, -39.6130219, 26.0289764, -65.7042084, 65.7029419
32: -44.9197998, 22.9436760, -44.8444519, 22.9138222, -67.8336182, 67.7881241
33: -59.5978127, 38.0865669, -59.4603386, 38.0024605, -97.6002731, 97.5469055
34: -52.2743645, 25.4464645, -52.1421432, 25.3768234, -77.6511841, 77.5886078
35: -53.5729523, 29.2983494, -53.4159775, 29.2146511, -82.7876053, 82.7143250
36: -53.2624359, 28.9230442, -53.1065750, 28.8534641, -82.1158905, 82.0296173
37: -71.7106552, 29.5364799, -71.5800171, 29.4762573, -101.1869125, 101.1165009
38: -62.8326988, 33.8117027, -62.6414337, 33.7292709, -96.5619659, 96.4531403
39: -72.9697342, 33.8963966, -72.8266373, 33.8349953, -106.8047333, 106.7230301
40: -59.3143234, 29.4848824, -59.1859665, 29.4344215, -88.7487488, 88.6708527
41: -41.8874130, 27.6813660, -41.7978973, 27.6428146, -69.5302277, 69.4792633
42: -29.5101433, 24.2251816, -29.4425888, 24.1454029, -53.6555481, 53.6677704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=45, inp2_unstable=45, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1688
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1672
type: A, layer: 1, pos: 1672
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1673
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.5024184, upper bound: 50.8141055
time: 78.10 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.5024184, upper bound: 51.0561396
time: 81.16 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 161.42 seconds
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5024184, upper bound: 51.0347023
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5024184, upper bound: 51.0347023
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5024184, upper bound: 51.0628205
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5024184, upper bound: 51.2914842
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5039438, upper bound: 51.0370435
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5039437, upper bound: 51.2578690
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5039446, upper bound: 51.0370444
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5039437, upper bound: 51.0370444
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5024184, upper bound: 51.0381817
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5024184, upper bound: 51.2592297
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5024184, upper bound: 50.8141055
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 161.42
Output dim: 10, lower bound: -50.5024184, upper bound: 51.0561396

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -62.1647263, 22.0037746, -62.0382042, 21.9366226, -84.1013489, 84.0419769
1: -34.6937294, 24.8430080, -34.6332970, 24.7575302, -59.4512596, 59.4763031
2: -27.8276043, 22.9852829, -27.7390900, 22.9238739, -50.7514801, 50.7243652
3: -32.7113037, 29.1316910, -32.6178398, 28.9803257, -61.6916275, 61.7495308
4: -34.6726379, 27.2534580, -34.5537262, 27.1759109, -61.8485489, 61.8071823
5: -30.8331432, 32.4106750, -30.7419243, 32.2606125, -63.0937500, 63.1525993
6: -37.9325714, 29.4640579, -37.7563019, 29.4155350, -67.3481064, 67.2203598
7: -40.0613327, 31.7006149, -39.9717674, 31.5501404, -71.6114655, 71.6723785
8: -38.6888428, 33.1263847, -38.5394516, 33.0350952, -71.7239380, 71.6658325
9: -29.9386826, 31.6187592, -29.7749977, 31.3222809, -61.2609634, 61.3937569
10: -43.7462692, 47.9356422, -43.3851242, 47.2195511, -90.9658203, 91.3207626
11: -44.8513107, 26.0377312, -44.6580086, 25.6921349, -70.5434418, 70.6957397
12: -42.2192154, 34.2564125, -42.0497551, 34.0583725, -76.2775879, 76.3061676
13: -45.5662193, 40.1092987, -45.4853668, 39.9780045, -85.5442200, 85.5946579
14: -77.7573013, 23.1325722, -77.5313721, 22.8079014, -100.5652008, 100.6639404
15: -36.9803848, 28.0736465, -36.9221001, 27.9081955, -64.8885803, 64.9957428
16: -48.1590233, 35.7439613, -47.9159927, 35.3353348, -83.4943542, 83.6599579
17: -77.0745773, 35.4001160, -76.8365402, 35.0012970, -112.0758743, 112.2366562
18: -40.5912018, 28.6310062, -40.4308853, 28.5307121, -69.1219101, 69.0618896
19: -30.9131813, 16.3239746, -30.7823086, 16.2358284, -47.1490097, 47.1062851
20: -31.8529053, 19.5201283, -31.7688770, 19.4787693, -51.3316650, 51.2890015
21: -43.6943054, 18.6411629, -43.5433693, 18.5354424, -62.2297478, 62.1845322
22: -51.3808479, 17.5356846, -51.1477165, 17.4604378, -68.8412781, 68.6833954
23: -31.9521046, 23.1842003, -31.8584042, 23.0790997, -55.0312042, 55.0426025
24: -44.3168869, 22.8694115, -44.1456146, 22.8105984, -67.1274872, 67.0150299
25: -34.0273476, 25.6047382, -33.9306412, 25.5378571, -59.5652046, 59.5353775
26: -49.8859406, 33.9218826, -49.7225380, 33.7885895, -83.6745300, 83.6444244
27: -47.8178787, 21.3656883, -47.5263138, 21.2529259, -69.0708008, 68.8919983
28: -33.9085999, 23.8795109, -33.7728958, 23.8060532, -57.7146492, 57.6524048
29: -57.2632713, 16.4470406, -57.1263123, 16.3788338, -73.6421051, 73.5733490
30: -42.2254562, 25.2839279, -42.1413879, 25.1794014, -67.4048538, 67.4253082
31: -39.6712608, 26.0825806, -39.5374107, 25.9859486, -65.6572113, 65.6199951
32: -44.9102020, 22.9389095, -44.7373314, 22.8704586, -67.7806549, 67.6762314
33: -59.5896606, 38.0836716, -59.2254448, 37.8993645, -97.4890289, 97.3091125
34: -52.2676010, 25.4439430, -51.9907875, 25.3059616, -77.5735474, 77.4347305
35: -53.5645065, 29.2965851, -53.1986961, 29.1207466, -82.6852570, 82.4952774
36: -53.2560959, 28.9215050, -52.8849869, 28.7648354, -82.0209198, 81.8064880
37: -71.7048187, 29.5253181, -71.2660522, 29.3671494, -101.0719604, 100.7913666
38: -62.8250580, 33.8058472, -62.3660316, 33.6241646, -96.4492188, 96.1718750
39: -72.9616852, 33.8942680, -72.5186768, 33.7302742, -106.6919556, 106.4129410
40: -59.3081284, 29.4773922, -58.9361801, 29.3477516, -88.6558838, 88.4135666
41: -41.8830261, 27.6734600, -41.6510849, 27.5977364, -69.4807587, 69.3245392
42: -29.5078239, 24.2173080, -29.3386536, 24.0112915, -53.5191154, 53.5559616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=44, inp2_unstable=45, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 646
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1639

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.3551453, upper bound: 51.2441424
time: 75.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.3551453, upper bound: 51.2564373
time: 85.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -62.1540031, 21.9967327, -62.0308800, 21.9068737, -84.0608749, 84.0276108
1: -34.6873894, 24.8356647, -34.6214790, 24.7309666, -59.4183540, 59.4571457
2: -27.8207664, 22.9796543, -27.7444382, 22.9113388, -50.7321053, 50.7240906
3: -32.6919518, 29.1255455, -32.5492439, 28.9920330, -61.6839828, 61.6747894
4: -34.6640739, 27.2483826, -34.5463257, 27.1781769, -61.8422470, 61.7947083
5: -30.8227386, 32.4045410, -30.7163219, 32.2605591, -63.0832977, 63.1208611
6: -37.9143600, 29.4605656, -37.7062149, 29.4245949, -67.3389587, 67.1667786
7: -40.0547180, 31.6956444, -39.9695740, 31.5429802, -71.5977020, 71.6652222
8: -38.6792755, 33.1189957, -38.5449524, 33.0186005, -71.6978760, 71.6639481
9: -29.9313622, 31.6083031, -29.8035889, 31.3636208, -61.2949791, 61.4118919
10: -43.7370110, 47.8758011, -43.4987335, 47.0650635, -90.8020782, 91.3745346
11: -44.8446083, 26.0048885, -44.7024612, 25.5856037, -70.4302063, 70.7073517
12: -42.2127609, 34.2325287, -42.1018524, 34.0138664, -76.2266235, 76.3343811
13: -45.5517464, 40.1008301, -45.4268990, 39.9861221, -85.5378723, 85.5277252
14: -77.7467194, 23.0939865, -77.5757599, 22.6583271, -100.4050369, 100.6697464
15: -36.9693069, 28.0664787, -36.8803520, 27.9283180, -64.8976288, 64.9468307
16: -48.1465454, 35.7176018, -47.9450569, 35.2950211, -83.4415588, 83.6626587
17: -77.0682602, 35.3617401, -76.8854828, 34.8581810, -111.9264374, 112.2472153
18: -40.5838699, 28.6049118, -40.4654350, 28.4037476, -68.9876099, 69.0703430
19: -30.9060059, 16.3114662, -30.7931881, 16.1904335, -47.0964355, 47.1046524
20: -31.8474751, 19.5143490, -31.7740993, 19.4536457, -51.3011208, 51.2884483
21: -43.6870499, 18.6248322, -43.5582581, 18.4704475, -62.1574974, 62.1830902
22: -51.3673592, 17.5286217, -51.1426277, 17.4684067, -68.8357697, 68.6712494
23: -31.9461098, 23.1689491, -31.8595524, 23.0083122, -54.9544220, 55.0285034
24: -44.3102341, 22.8590279, -44.1787262, 22.7707443, -67.0809784, 67.0377502
25: -34.0216904, 25.5928745, -33.9397163, 25.4847298, -59.5064201, 59.5325928
26: -49.8786469, 33.8995895, -49.7344398, 33.7200241, -83.5986710, 83.6340256
27: -47.7986145, 21.3614845, -47.5030403, 21.2873802, -69.0859985, 68.8645172
28: -33.8933640, 23.8754749, -33.7301407, 23.8311386, -57.7245026, 57.6056137
29: -57.2518539, 16.4385147, -57.1071205, 16.3718834, -73.6237335, 73.5456390
30: -42.2192116, 25.2698116, -42.1412277, 25.1180458, -67.3372574, 67.4110413
31: -39.6618881, 26.0615463, -39.5426979, 25.8880043, -65.5498886, 65.6042480
32: -44.8881454, 22.9356689, -44.6409073, 22.8822594, -67.7703934, 67.5765762
33: -59.5656357, 38.0786018, -59.1760979, 37.9522552, -97.5178909, 97.2546921
34: -52.2399368, 25.4387627, -51.9066772, 25.3396301, -77.5795670, 77.3454361
35: -53.5324097, 29.2925491, -53.1022873, 29.1780872, -82.7104950, 82.3948364
36: -53.2199173, 28.9182949, -52.7545090, 28.8156738, -82.0355835, 81.6728058
37: -71.6871872, 29.5206032, -71.2647705, 29.4027710, -101.0899506, 100.7853699
38: -62.7935066, 33.8002167, -62.3117752, 33.6684761, -96.4619751, 96.1119919
39: -72.9372101, 33.8903961, -72.4750061, 33.7704582, -106.7076721, 106.3653946
40: -59.2846107, 29.4745502, -58.8839340, 29.3795967, -88.6642075, 88.3584824
41: -41.8627625, 27.6696663, -41.5840836, 27.6115456, -69.4743042, 69.2537537
42: -29.5023670, 24.2087288, -29.3738098, 24.0258160, -53.5281830, 53.5825386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=44, inp2_unstable=45, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1688
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 1658
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 693
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1639

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.3566617, upper bound: 51.2134571
time: 81.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.3551453, upper bound: 51.2134562
time: 80.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -62.1540031, 21.9967327, -62.0327263, 21.9283829, -84.0823822, 84.0294571
1: -34.6873894, 24.8356647, -34.6189270, 24.7643375, -59.4517212, 59.4545898
2: -27.8207664, 22.9796543, -27.7284946, 22.9180107, -50.7387733, 50.7081490
3: -32.6919518, 29.1255455, -32.5066109, 29.0510769, -61.7430267, 61.6321564
4: -34.6640739, 27.2483826, -34.5586014, 27.1947517, -61.8588257, 61.8069763
5: -30.8227386, 32.4045410, -30.7167664, 32.3365402, -63.1592751, 63.1213074
6: -37.9143600, 29.4605656, -37.7117882, 29.4258308, -67.3401871, 67.1723480
7: -40.0547180, 31.6956444, -39.9747276, 31.6371517, -71.6918716, 71.6703720
8: -38.6792755, 33.1189957, -38.5462227, 33.0375443, -71.7168121, 71.6652069
9: -29.9313622, 31.6083031, -29.8233643, 31.4217415, -61.3530998, 61.4316673
10: -43.7370110, 47.8758011, -43.5141449, 47.1505814, -90.8875885, 91.3899460
11: -44.8446083, 26.0048885, -44.7279320, 25.6316509, -70.4762573, 70.7328186
12: -42.2127609, 34.2325287, -42.0798187, 33.9372139, -76.1499786, 76.3123474
13: -45.5517464, 40.1008301, -45.4256020, 40.0098076, -85.5615387, 85.5264282
14: -77.7467194, 23.0939865, -77.5719223, 22.6761761, -100.4228973, 100.6659088
15: -36.9693069, 28.0664787, -36.8868484, 27.9524040, -64.9217072, 64.9533234
16: -48.1465454, 35.7176018, -47.9828033, 35.3796082, -83.5261536, 83.7004089
17: -77.0682602, 35.3617401, -76.9448090, 34.9476471, -112.0159073, 112.3065491
18: -40.5838699, 28.6049118, -40.5020752, 28.4004860, -68.9843597, 69.1069870
19: -30.9060059, 16.3114662, -30.8106594, 16.1868649, -47.0928726, 47.1221237
20: -31.8474751, 19.5143490, -31.7776661, 19.4648170, -51.3122940, 51.2920151
21: -43.6870499, 18.6248322, -43.5840492, 18.4654675, -62.1525192, 62.2088776
22: -51.3673592, 17.5286217, -51.1906242, 17.4558105, -68.8231659, 68.7192459
23: -31.9461098, 23.1689491, -31.8778172, 23.0281410, -54.9742508, 55.0467606
24: -44.3102341, 22.8590279, -44.2285156, 22.7729874, -67.0832214, 67.0875397
25: -34.0216904, 25.5928745, -33.9557037, 25.4882545, -59.5099449, 59.5485764
26: -49.8786469, 33.8995895, -49.7824707, 33.6716919, -83.5503387, 83.6820602
27: -47.7986145, 21.3614845, -47.5453796, 21.2906761, -69.0892944, 68.9068604
28: -33.8933640, 23.8754749, -33.7230644, 23.8275738, -57.7209282, 57.5985413
29: -57.2518539, 16.4385147, -57.1190605, 16.3589020, -73.6107559, 73.5575714
30: -42.2192116, 25.2698116, -42.1485176, 25.1443577, -67.3635712, 67.4183273
31: -39.6618881, 26.0615463, -39.5549164, 25.8684120, -65.5302963, 65.6164627
32: -44.8881454, 22.9356689, -44.6757126, 22.8952694, -67.7834091, 67.6113815
33: -59.5656357, 38.0786018, -59.2725449, 37.9708099, -97.5364456, 97.3511505
34: -52.2399368, 25.4387627, -51.9243965, 25.3404922, -77.5804291, 77.3631592
35: -53.5324097, 29.2925491, -53.1612701, 29.1882744, -82.7206802, 82.4538193
36: -53.2199173, 28.9182949, -52.8188286, 28.8344364, -82.0543442, 81.7371216
37: -71.6871872, 29.5206032, -71.4472961, 29.4506168, -101.1378021, 100.9678955
38: -62.7935066, 33.8002167, -62.3944397, 33.6907425, -96.4842453, 96.1946564
39: -72.9372101, 33.8903961, -72.6401062, 33.8128853, -106.7500916, 106.5305023
40: -59.2846107, 29.4745502, -59.0048294, 29.4168682, -88.7014771, 88.4793777
41: -41.8627625, 27.6696663, -41.6404266, 27.6171741, -69.4799347, 69.3100891
42: -29.5023670, 24.2087288, -29.4095554, 24.0804710, -53.5828323, 53.6182861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=44, inp2_unstable=45, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=603, inp2_unstable=603, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=28, inp2_unstable=28, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1655
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 1641
type: B, layer: 1, pos: 1641
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 631
type: B, layer: 1, pos: 694
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 630
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 631
type: A, layer: 1, pos: 1658
type: B, layer: 1, pos: 1658
type: A, layer: 1, pos: 646
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 1638
type: A, layer: 1, pos: 1638
type: B, layer: 1, pos: 1721
type: A, layer: 1, pos: 1721
type: B, layer: 1, pos: 677
type: A, layer: 1, pos: 677
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 693
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1719
type: B, layer: 1, pos: 1719
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1702
type: B, layer: 1, pos: 1702
type: A, layer: 1, pos: 1673
type: B, layer: 1, pos: 1785
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1785
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1718
type: A, layer: 1, pos: 1718
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1769
type: B, layer: 1, pos: 1769
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 712
type: A, layer: 1, pos: 712
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1767
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1770
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1639

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1655

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 10, lower bound: -50.3551453, upper bound: 51.2148586
time: 82.53 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 10, lower bound: -50.3551453, upper bound: 51.2592308
time: 100.85 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 185.60 seconds
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 185.60
Output dim: 10, lower bound: -50.3551453, upper bound: 51.2441424
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 185.60
Output dim: 10, lower bound: -50.3551453, upper bound: 51.2564373
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 185.60
Output dim: 10, lower bound: -50.3566617, upper bound: 51.2134571
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 185.60
Output dim: 10, lower bound: -50.3551453, upper bound: 51.2134562
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 185.60
Output dim: 10, lower bound: -50.3551453, upper bound: 51.2148586
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 185.60
Output dim: 10, lower bound: -50.3551453, upper bound: 51.2592308

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 91.61 + 3604.93 = 3696.54 seconds
