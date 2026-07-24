## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 5)
Time budget: 1800 seconds
Split limit: 100
Threshold: 9.9371055474


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=128, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-22.4528351, 9.3018913, -22.4528351, 9.3018913, -31.6889954, 31.6889954)
1: (-12.0579910, 7.8092408, -12.0579910, 7.8092408, -19.8672314, 19.8672314)
2: (-11.6616526, 9.6392078, -11.6616526, 9.6392078, -18.7814941, 18.7814903)
3: (-17.6266747, 7.4750128, -17.6266747, 7.4750128, -23.5188141, 23.5188141)
4: (-19.6373272, 5.1794066, -19.6373272, 5.1794066, -22.4152756, 22.4152756)
5: (-15.5945940, 9.7024384, -15.5945940, 9.7024384, -24.1424713, 24.1424713)
6: (-31.9586792, -7.3889952, -31.9586792, -7.3889952, -19.8810463, 19.8810425)
7: (-21.6431904, 6.0399432, -21.6431904, 6.0399432, -26.3111877, 26.3111877)
8: (-23.6548958, 7.6122541, -23.6548958, 7.6122541, -29.5465698, 29.5465622)
9: (-13.7933216, 10.0569639, -13.7933216, 10.0569639, -20.7479706, 20.7479706)
10: (-13.9747210, 14.1570501, -13.9747210, 14.1570501, -27.6405106, 27.6405106)
11: (-10.2519779, 11.3968925, -10.2519779, 11.3968925, -17.6082954, 17.6082916)
12: (-23.3104210, 13.2517042, -23.3104210, 13.2517042, -34.4471130, 34.4471130)
13: (-25.3844223, 6.1506929, -25.3844223, 6.1506929, -31.0297546, 31.0297470)
14: (-26.3112030, 14.9268684, -26.3112030, 14.9268684, -39.5937500, 39.5937576)
15: (-10.0700121, 13.0112495, -10.0700121, 13.0112495, -21.6990509, 21.6990547)
16: (-20.9371681, 4.5060515, -20.9371681, 4.5060515, -25.2432404, 25.2432404)
17: (-23.0856533, 11.2652283, -23.0856533, 11.2652283, -34.3508835, 34.3508835)
18: (-11.2293196, 16.5958691, -11.2293196, 16.5958691, -27.0257187, 27.0257111)
19: (-7.2696838, 8.3759174, -7.2696838, 8.3759174, -14.7189674, 14.7189693)
20: (-6.5881424, 10.0585117, -6.5881424, 10.0585117, -15.4445305, 15.4445305)
21: (-7.6137123, 11.8018913, -7.6137123, 11.8018913, -18.4414215, 18.4414215)
22: (-5.0874300, 15.3768673, -5.0874300, 15.3768673, -18.3680840, 18.3680878)
23: (-2.9946470, 15.0638771, -2.9946470, 15.0638771, -15.8121185, 15.8121147)
24: (-5.3951068, 13.2942801, -5.3951068, 13.2942801, -14.4689064, 14.4689102)
25: (-0.9779487, 19.6314106, -0.9779487, 19.6314106, -15.4996719, 15.4996719)
26: (-12.1344585, 19.6839962, -12.1344585, 19.6839962, -31.8184547, 31.8184547)
27: (-9.4839783, 10.9603243, -9.4839783, 10.9603243, -19.7664948, 19.7664948)
28: (-4.2052813, 15.1499662, -4.2052813, 15.1499662, -17.6528015, 17.6528015)
29: (-3.9171309, 15.9205132, -3.9171309, 15.9205132, -16.3217545, 16.3217525)
30: (-10.8963623, 10.4176655, -10.8963623, 10.4176655, -17.7909660, 17.7909622)
31: (-6.8471394, 12.5883255, -6.8471394, 12.5883255, -18.8262520, 18.8262482)
32: (-26.5178833, -1.8192225, -26.5178833, -1.8192225, -22.6703110, 22.6703110)
33: (-43.5889168, -7.7874060, -43.5889168, -7.7874060, -28.9142456, 28.9142456)
34: (-36.2045822, -6.0280871, -36.2045822, -6.0280871, -23.0099030, 23.0099068)
35: (-26.7947922, 1.2281170, -26.7947922, 1.2281170, -24.9686661, 24.9686661)
36: (-27.0775299, 4.8327208, -27.0775299, 4.8327208, -31.4258728, 31.4258652)
37: (-44.2352524, -9.2084599, -44.2352524, -9.2084599, -28.6157761, 28.6157684)
38: (-31.5190010, 3.1028409, -31.5190010, 3.1028409, -34.6218414, 34.6218414)
39: (-48.4663544, -10.6560192, -48.4663544, -10.6560192, -33.8900146, 33.8900070)
40: (-44.5805206, -17.6986408, -44.5805206, -17.6986408, -19.9788589, 19.9788513)
41: (-30.4815140, -4.1211324, -30.4815140, -4.1211324, -21.3767014, 21.3767014)
42: (-19.9963417, -0.2377048, -19.9963417, -0.2377048, -15.4059563, 15.4059582)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.33 + 40.37 = 42.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 25, lower bound: -9.9470526, upper bound: 9.9470526

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1723
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 1781
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1723

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9311805, upper bound: 9.9445473
time: 23.07 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9445471, upper bound: 9.9445472
time: 22.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 45.66 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 45.66
Output dim: 25, lower bound: -9.9311805, upper bound: 9.9445473
IS_B2, status: Status.UNKNOWN, split count: 1, time: 45.66
Output dim: 25, lower bound: -9.9445471, upper bound: 9.9445472

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -22.4490681, 9.2882423, -22.4459877, 9.2773228, -31.6597290, 31.6683578
1: -12.0567589, 7.8040566, -12.0557423, 7.7999167, -19.8566761, 19.8597984
2: -11.6598425, 9.6255322, -11.6584082, 9.6144276, -18.7520065, 18.7629814
3: -17.6251621, 7.4640112, -17.6239262, 7.4551916, -23.4943619, 23.5032654
4: -19.6325092, 5.1736031, -19.6286049, 5.1689606, -22.3969498, 22.3999405
5: -15.5928745, 9.6836052, -15.5915108, 9.6680384, -24.1063080, 24.1204147
6: -31.9353371, -7.3911686, -31.9168739, -7.3928671, -19.8490868, 19.8356400
7: -21.6403332, 6.0287933, -21.6379986, 6.0195446, -26.2755280, 26.2877960
8: -23.6535110, 7.6057253, -23.6524544, 7.6004629, -29.5284729, 29.5342560
9: -13.7891750, 10.0517035, -13.7857733, 10.0475721, -20.7294312, 20.7335892
10: -13.9699163, 14.1465149, -13.9660912, 14.1379719, -27.6104050, 27.6178207
11: -10.2454338, 11.3955746, -10.2402315, 11.3944969, -17.5915852, 17.5903778
12: -23.2809868, 13.2487593, -23.2565651, 13.2462778, -34.4121552, 34.3899841
13: -25.3704510, 6.1445079, -25.3591595, 6.1396146, -31.0051880, 30.9991837
14: -26.3039780, 14.9041672, -26.2980957, 14.8869057, -39.5426788, 39.5540771
15: -10.0666399, 13.0032425, -10.0639458, 12.9968920, -21.6789856, 21.6782303
16: -20.9315319, 4.5009637, -20.9270153, 4.4968324, -25.2087479, 25.2164154
17: -23.0779476, 11.2594252, -23.0716820, 11.2547541, -34.3327026, 34.3311081
18: -11.2214680, 16.5939236, -11.2153893, 16.5922966, -27.0082855, 27.0007782
19: -7.2621737, 8.3666000, -7.2561531, 8.3590736, -14.6936817, 14.6952515
20: -6.5817451, 10.0465050, -6.5765753, 10.0370216, -15.4144516, 15.4191418
21: -7.6061449, 11.7924500, -7.6000452, 11.7847843, -18.4159431, 18.4174271
22: -5.0807152, 15.3690376, -5.0752640, 15.3628139, -18.3464165, 18.3388100
23: -2.9878278, 15.0506439, -2.9823551, 15.0398998, -15.7806473, 15.7865143
24: -5.3912821, 13.2838936, -5.3881788, 13.2752399, -14.4434547, 14.4489136
25: -0.9730802, 19.6097031, -0.9691415, 19.5921898, -15.4526863, 15.4653931
26: -12.1126995, 19.6812496, -12.0949173, 19.6789742, -31.7916737, 31.7761669
27: -9.4770489, 10.9509954, -9.4714470, 10.9433413, -19.7415085, 19.7409172
28: -4.1959400, 15.1349583, -4.1883402, 15.1224680, -17.6152191, 17.6201782
29: -3.9102488, 15.9114113, -3.9046164, 15.9038982, -16.3002434, 16.2985802
30: -10.8903179, 10.4107056, -10.8854847, 10.4050856, -17.7688179, 17.7711029
31: -6.8400612, 12.5698929, -6.8343925, 12.5545740, -18.7829628, 18.7922211
32: -26.4942856, -1.8219080, -26.4748745, -1.8240519, -22.6413727, 22.6253662
33: -43.5616989, -7.7919760, -43.5400963, -7.7956514, -28.8762589, 28.8569717
34: -36.1764297, -6.0305276, -36.1534882, -6.0324883, -22.9775467, 22.9566956
35: -26.7723827, 1.2264175, -26.7543240, 1.2250686, -24.9421921, 24.9248123
36: -27.0510540, 4.8310857, -27.0291615, 4.8298149, -31.3959503, 31.3752670
37: -44.1890869, -9.2107477, -44.1510811, -9.2125607, -28.5652542, 28.5287552
38: -31.4955750, 3.1002226, -31.4763279, 3.0980997, -34.5936737, 34.5765495
39: -48.4318237, -10.6610107, -48.4037094, -10.6651001, -33.8458557, 33.8212814
40: -44.5412903, -17.7009277, -44.5096321, -17.7027779, -19.9356308, 19.9067307
41: -30.4459114, -4.1238899, -30.4173946, -4.1260853, -21.3360519, 21.3101959
42: -19.9771080, -0.2417526, -19.9624043, -0.2450414, -15.3781967, 15.3672810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1725

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9307216, upper bound: 9.9276224
time: 21.29 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9307246, upper bound: 9.9440915
time: 23.45 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -22.4513435, 9.2991657, -22.4917431, 9.3028898, -31.6862793, 31.7271805
1: -12.0572376, 7.8077135, -12.0752878, 7.8126216, -19.8698597, 19.8830013
2: -11.6608400, 9.6367445, -11.6847019, 9.6406384, -18.7790146, 18.8073769
3: -17.6260719, 7.4729891, -17.6577454, 7.4796214, -23.5161743, 23.5526276
4: -19.6353035, 5.1777301, -19.6392765, 5.1867046, -22.4156952, 22.4141083
5: -15.5937424, 9.6992493, -15.6414185, 9.7036867, -24.1407928, 24.1856613
6: -31.9534702, -7.3897119, -31.9569798, -7.3476639, -19.8998032, 19.8661270
7: -21.6415653, 6.0368676, -21.6738014, 6.0426598, -26.2981110, 26.3602066
8: -23.6538811, 7.6095858, -23.6664715, 7.6169176, -29.5445404, 29.5653076
9: -13.7918873, 10.0554667, -13.8009605, 10.0632811, -20.7416763, 20.7586975
10: -13.9725752, 14.1535034, -13.9861698, 14.1627121, -27.6402740, 27.6511459
11: -10.2479973, 11.3964701, -10.2714024, 11.4040976, -17.6000214, 17.6342125
12: -23.3061848, 13.2504406, -23.3148842, 13.3354063, -34.5258942, 34.4428864
13: -25.3825722, 6.1475658, -25.3937664, 6.1684523, -31.0439301, 31.0356979
14: -26.3094864, 14.9216061, -26.3800850, 14.9234905, -39.5832062, 39.6536331
15: -10.0689564, 13.0044689, -10.0869932, 13.0098314, -21.7180481, 21.6871529
16: -20.9332199, 4.5048981, -20.9514503, 4.5130258, -25.2108383, 25.2966080
17: -23.0838261, 11.2603188, -23.1309242, 11.2661734, -34.3499985, 34.3912430
18: -11.2252111, 16.5953712, -11.2341022, 16.6084156, -27.0447540, 27.0216751
19: -7.2677479, 8.3751211, -7.3004827, 8.3759556, -14.7140846, 14.7468338
20: -6.5859289, 10.0567179, -6.6158805, 10.0608616, -15.4407654, 15.4706116
21: -7.6115341, 11.8009453, -7.6387854, 11.8042831, -18.4400063, 18.4646339
22: -5.0858002, 15.3686686, -5.1110716, 15.3695869, -18.3855057, 18.3458481
23: -2.9925203, 15.0625896, -3.0309706, 15.0639200, -15.8050232, 15.8465805
24: -5.3939028, 13.2923317, -5.4153743, 13.2966480, -14.4639397, 14.4836082
25: -0.9765520, 19.6277199, -1.0314975, 19.6314621, -15.4842815, 15.5452271
26: -12.1301384, 19.6833496, -12.1434116, 19.7397079, -31.8698463, 31.8267612
27: -9.4813232, 10.9585543, -9.4953604, 10.9612751, -19.7621078, 19.7669640
28: -4.2028160, 15.1476479, -4.2382154, 15.1492300, -17.6454468, 17.6852379
29: -3.9156656, 15.9166536, -3.9574366, 15.9170456, -16.3179550, 16.3416328
30: -10.8941612, 10.4160347, -10.9070063, 10.4226818, -17.7916183, 17.8004036
31: -6.8451881, 12.5863504, -6.9070482, 12.5886621, -18.8166847, 18.8799744
32: -26.5143967, -1.8201613, -26.5188656, -1.7798195, -22.7000198, 22.6590118
33: -43.5834389, -7.7894402, -43.5869331, -7.7190008, -28.9859619, 28.8952484
34: -36.2015076, -6.0288877, -36.2061615, -5.9537959, -23.0796280, 22.9953537
35: -26.7921619, 1.2267537, -26.7990074, 1.2849684, -25.0231171, 24.9603958
36: -27.0734959, 4.8319941, -27.0805645, 4.8968482, -31.4846191, 31.4242401
37: -44.2283478, -9.2092476, -44.2361755, -9.0773287, -28.7414856, 28.5947189
38: -31.5147076, 3.1021209, -31.5211697, 3.1633391, -34.6780472, 34.6232910
39: -48.4626541, -10.6583128, -48.4669647, -10.5571270, -33.9857788, 33.8715363
40: -44.5747452, -17.6993446, -44.5770187, -17.5896397, -20.0845337, 19.9521828
41: -30.4751358, -4.1219616, -30.4787903, -4.0458221, -21.4477119, 21.3585548
42: -19.9932728, -0.2387424, -19.9970131, -0.2057533, -15.4307632, 15.3980122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=128, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=150, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1449

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1725

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9440884, upper bound: 9.9276224
time: 20.45 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9440914, upper bound: 9.9440915
time: 22.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 44.96 seconds
IS_B1_A1, status: Status.VERIFIED, split count: 2, time: 44.96
Output dim: 25, lower bound: -9.9307216, upper bound: 9.9276224
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 44.96
Output dim: 25, lower bound: -9.9307246, upper bound: 9.9440915
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 44.96
Output dim: 25, lower bound: -9.9440884, upper bound: 9.9276224
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 44.96
Output dim: 25, lower bound: -9.9440914, upper bound: 9.9440915

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -22.4484406, 9.2879429, -22.4456844, 9.2771587, -31.6579437, 31.6723633
1: -12.0565033, 7.8036785, -12.0556145, 7.7997370, -19.8562393, 19.8592930
2: -11.6584482, 9.6251965, -11.6577015, 9.6142273, -18.7388306, 18.7620850
3: -17.6248245, 7.4635229, -17.6237354, 7.4549203, -23.4733810, 23.5023346
4: -19.6300468, 5.1731911, -19.6272545, 5.1687307, -22.3676605, 22.3992004
5: -15.5913525, 9.6832256, -15.5907288, 9.6678295, -24.1043243, 24.1205215
6: -31.9336491, -7.3913794, -31.9159927, -7.3930020, -19.8708572, 19.8255844
7: -21.6377983, 6.0284190, -21.6367378, 6.0193882, -26.2577438, 26.2859802
8: -23.6525135, 7.6053314, -23.6516571, 7.6002679, -29.5006714, 29.5335999
9: -13.7886562, 10.0513248, -13.7854919, 10.0473404, -20.7095337, 20.7320709
10: -13.9694500, 14.1458273, -13.9657841, 14.1376228, -27.6080017, 27.6159058
11: -10.2449274, 11.3944969, -10.2399855, 11.3939533, -17.5907993, 17.5755615
12: -23.2803154, 13.2483435, -23.2562294, 13.2460842, -34.4103088, 34.3872833
13: -25.3679199, 6.1441245, -25.3578854, 6.1394229, -30.9824982, 30.9974365
14: -26.3029251, 14.9022284, -26.2975655, 14.8859491, -39.5390930, 39.5521240
15: -10.0651512, 13.0027609, -10.0631809, 12.9966030, -21.6747055, 21.6869812
16: -20.9303665, 4.5005274, -20.9263115, 4.4965959, -25.2286148, 25.2058868
17: -23.0764351, 11.2591457, -23.0709114, 11.2546406, -34.3310776, 34.3300552
18: -11.2209492, 16.5922432, -11.2151089, 16.5914688, -27.0076065, 26.9680328
19: -7.2614489, 8.3652534, -7.2557755, 8.3584013, -14.6924610, 14.6840324
20: -6.5811281, 10.0449600, -6.5762372, 10.0362492, -15.4126129, 15.4049606
21: -7.6053143, 11.7911224, -7.5996122, 11.7840967, -18.4141159, 18.4024887
22: -5.0801744, 15.3684750, -5.0749750, 15.3625097, -18.3441086, 18.3152161
23: -2.9872847, 15.0488157, -2.9820843, 15.0389805, -15.7791977, 15.7539520
24: -5.3909473, 13.2823334, -5.3879881, 13.2744589, -14.4429665, 14.4118729
25: -0.9725323, 19.6082058, -0.9688621, 19.5913467, -15.4518871, 15.4098244
26: -12.1119671, 19.6801529, -12.0945072, 19.6782742, -31.7902412, 31.7746601
27: -9.4765911, 10.9496021, -9.4712029, 10.9426622, -19.7403603, 19.7183838
28: -4.1952782, 15.1332579, -4.1879978, 15.1216259, -17.6137428, 17.5972710
29: -3.9097528, 15.9111309, -3.9043770, 15.9037476, -16.2996025, 16.2553978
30: -10.8898048, 10.4094505, -10.8851795, 10.4044323, -17.7673264, 17.7545166
31: -6.8392348, 12.5683842, -6.8339882, 12.5538073, -18.7813225, 18.7764664
32: -26.4926434, -1.8221879, -26.4739819, -1.8242273, -22.6581497, 22.6160736
33: -43.5590973, -7.7923584, -43.5388031, -7.7958994, -28.8168030, 28.8536072
34: -36.1732101, -6.0309343, -36.1518631, -6.0326910, -22.9499969, 22.9531937
35: -26.7698040, 1.2262211, -26.7529373, 1.2249370, -24.9158478, 24.9221115
36: -27.0480099, 4.8308568, -27.0275879, 4.8296561, -31.3840027, 31.3732910
37: -44.1859894, -9.2109404, -44.1495438, -9.2126999, -28.5186996, 28.5256729
38: -31.4925938, 3.0999217, -31.4748249, 3.0979195, -34.5905151, 34.5747452
39: -48.4281502, -10.6614094, -48.4018974, -10.6652946, -33.7686920, 33.8186035
40: -44.5385132, -17.7011185, -44.5082092, -17.7028732, -19.9270020, 19.8961639
41: -30.4444275, -4.1242480, -30.4166222, -4.1262889, -21.3332901, 21.3052673
42: -19.9749584, -0.2422438, -19.9612751, -0.2452931, -15.3841705, 15.3635292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1716

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9201147, upper bound: 9.9431566
time: 25.97 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9297889, upper bound: 9.9431566
time: 29.39 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -22.3977699, 9.2696686, -22.4617462, 9.2991886, -31.6293335, 31.6637039
1: -12.0295944, 7.7875209, -12.0604134, 7.8097687, -19.8393631, 19.8479347
2: -11.6351261, 9.6234150, -11.6705894, 9.6376772, -18.7495270, 18.7783051
3: -17.5881939, 7.4461756, -17.6359005, 7.4753437, -23.4766006, 23.5084457
4: -19.5806999, 5.1507387, -19.6082649, 5.1830754, -22.3530502, 22.3465195
5: -15.5722771, 9.6822243, -15.6294184, 9.6985970, -24.1127472, 24.1534805
6: -31.9313202, -7.4047661, -31.9454880, -7.3522272, -19.8704834, 19.8566856
7: -21.6068707, 6.0197039, -21.6549339, 6.0391445, -26.2589111, 26.3216171
8: -23.6002045, 7.5783997, -23.6363735, 7.6134820, -29.4861298, 29.5010834
9: -13.7608089, 10.0301609, -13.7831612, 10.0592260, -20.7081146, 20.7203827
10: -13.9593391, 14.1380482, -13.9813862, 14.1561089, -27.6183167, 27.6296463
11: -10.2205486, 11.3629971, -10.2672892, 11.3854227, -17.5519466, 17.5948029
12: -23.2912140, 13.2370405, -23.3083801, 13.3284588, -34.4995422, 34.4206848
13: -25.3099442, 6.1043243, -25.3532562, 6.1632023, -30.9638824, 30.9496078
14: -26.2844086, 14.8970861, -26.3714294, 14.9106007, -39.5522919, 39.6192474
15: -10.0519133, 12.9885864, -10.0795097, 13.0057716, -21.6946411, 21.6533661
16: -20.9100952, 4.4863787, -20.9394760, 4.5093307, -25.1847000, 25.2885971
17: -23.0562820, 11.2479982, -23.1209030, 11.2631512, -34.3194351, 34.3689003
18: -11.1854362, 16.5438519, -11.2280741, 16.5792122, -26.9686737, 26.9615326
19: -7.2412262, 8.3491573, -7.2950516, 8.3614578, -14.6706696, 14.7124138
20: -6.5609374, 10.0282316, -6.6107717, 10.0449095, -15.3992348, 15.4361382
21: -7.5800533, 11.7599945, -7.6328526, 11.7817593, -18.3866196, 18.4185944
22: -5.0600142, 15.3417521, -5.1060486, 15.3537617, -18.3427849, 18.3195190
23: -2.9582448, 15.0068817, -3.0260448, 15.0326910, -15.7366104, 15.7854004
24: -5.3630347, 13.2475214, -5.4108815, 13.2709904, -14.4016228, 14.4328651
25: -0.9497309, 19.5769539, -1.0269194, 19.6020107, -15.4128494, 15.4845772
26: -12.0854263, 19.6444168, -12.1345510, 19.7177372, -31.8031635, 31.7789688
27: -9.4443951, 10.9151020, -9.4899960, 10.9375992, -19.6973686, 19.7182617
28: -4.1672020, 15.0976467, -4.2321267, 15.1213865, -17.5805359, 17.6291008
29: -3.8896093, 15.8768978, -3.9528990, 15.8935089, -16.2610474, 16.2933483
30: -10.8669577, 10.3756809, -10.9031372, 10.4008808, -17.7432404, 17.7555466
31: -6.8115511, 12.5383329, -6.9013300, 12.5626717, -18.7548370, 18.8261604
32: -26.4802990, -1.8409925, -26.5005760, -1.7837095, -22.6691589, 22.6462555
33: -43.5031586, -7.8279505, -43.5427475, -7.7230873, -28.9163513, 28.8182602
34: -36.1509552, -6.0451279, -36.1792526, -5.9555092, -23.0358124, 22.9576721
35: -26.7456322, 1.2035303, -26.7740402, 1.2826972, -24.9787827, 24.9173660
36: -27.0317822, 4.8132696, -27.0594101, 4.8946996, -31.4427185, 31.3871841
37: -44.1408119, -9.2415895, -44.1878014, -9.0802660, -28.6600342, 28.5264816
38: -31.4645710, 3.0829697, -31.4986649, 3.1613693, -34.6259384, 34.5816345
39: -48.3515930, -10.7033682, -48.4047661, -10.5598097, -33.8779602, 33.7661133
40: -44.4952049, -17.7315903, -44.5317116, -17.5919704, -20.0295410, 19.9155540
41: -30.4313126, -4.1427407, -30.4550056, -4.0490880, -21.4108810, 21.3412857
42: -19.9769077, -0.2495193, -19.9890347, -0.2094369, -15.4093895, 15.3881912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=150, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1449

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1716

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9334788, upper bound: 9.9266852
time: 24.76 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9431528, upper bound: 9.9266852
time: 23.62 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -22.4506397, 9.2988625, -22.4914513, 9.3027401, -31.6845703, 31.7312164
1: -12.0570211, 7.8073030, -12.0751534, 7.8123999, -19.8694210, 19.8824558
2: -11.6594496, 9.6364002, -11.6840124, 9.6404562, -18.7658157, 18.8064728
3: -17.6257324, 7.4724360, -17.6575356, 7.4793491, -23.4951172, 23.5517273
4: -19.6328602, 5.1773167, -19.6379395, 5.1864738, -22.3864517, 22.4133530
5: -15.5922289, 9.6988716, -15.6406555, 9.7034473, -24.1387634, 24.1857758
6: -31.9517326, -7.3899465, -31.9561138, -7.3478251, -19.9215317, 19.8560715
7: -21.6390762, 6.0365171, -21.6725349, 6.0424833, -26.2803192, 26.3584595
8: -23.6528683, 7.6092095, -23.6657028, 7.6167021, -29.5166931, 29.5646362
9: -13.7913713, 10.0550575, -13.8006973, 10.0630665, -20.7217636, 20.7571793
10: -13.9721146, 14.1528263, -13.9858885, 14.1623516, -27.6377869, 27.6492233
11: -10.2475252, 11.3953753, -10.2711391, 11.4035654, -17.5992489, 17.6193848
12: -23.3055305, 13.2499876, -23.3144989, 13.3351908, -34.5241089, 34.4401779
13: -25.3800125, 6.1471868, -25.3924465, 6.1682887, -31.0212708, 31.0339355
14: -26.3083916, 14.9196568, -26.3795090, 14.9224510, -39.5796967, 39.6514893
15: -10.0674191, 13.0040073, -10.0862379, 13.0095911, -21.7137680, 21.6958656
16: -20.9320946, 4.5044575, -20.9507866, 4.5127916, -25.2306213, 25.2860641
17: -23.0822983, 11.2600565, -23.1301346, 11.2660437, -34.3483429, 34.3901901
18: -11.2246695, 16.5936699, -11.2337894, 16.6075974, -27.0440598, 26.9889603
19: -7.2670202, 8.3737583, -7.3001022, 8.3752699, -14.7128639, 14.7356091
20: -6.5852766, 10.0551844, -6.6155486, 10.0600901, -15.4389076, 15.4564133
21: -7.6106853, 11.7996159, -7.6383495, 11.8036098, -18.4381676, 18.4496880
22: -5.0852494, 15.3681469, -5.1107693, 15.3692446, -18.3831825, 18.3221931
23: -2.9919724, 15.0607624, -3.0306635, 15.0630283, -15.8035583, 15.8140182
24: -5.3935480, 13.2907763, -5.4151821, 13.2958641, -14.4634666, 14.4465904
25: -0.9760060, 19.6262131, -1.0311980, 19.6306019, -15.4834690, 15.4896393
26: -12.1293793, 19.6822414, -12.1429577, 19.7390327, -31.8684120, 31.8251991
27: -9.4808464, 10.9571753, -9.4951077, 10.9605589, -19.7609406, 19.7444115
28: -4.2021351, 15.1459694, -4.2378535, 15.1483879, -17.6439743, 17.6623154
29: -3.9151688, 15.9163876, -3.9571638, 15.9168968, -16.3173180, 16.2984467
30: -10.8935900, 10.4147730, -10.9066782, 10.4220524, -17.7901192, 17.7837982
31: -6.8443518, 12.5848398, -6.9066000, 12.5879011, -18.8150482, 18.8641891
32: -26.5127659, -1.8204441, -26.5179367, -1.7799778, -22.7167130, 22.6496658
33: -43.5808716, -7.7898355, -43.5855980, -7.7192254, -28.9264984, 28.8918304
34: -36.1982803, -6.0293069, -36.2044983, -5.9540019, -23.0520630, 22.9918289
35: -26.7895775, 1.2265291, -26.7975864, 1.2848258, -24.9966812, 24.9577103
36: -27.0704632, 4.8317513, -27.0790577, 4.8966722, -31.4727631, 31.4221878
37: -44.2251816, -9.2094679, -44.2346115, -9.0774584, -28.6948624, 28.5916214
38: -31.5117092, 3.1017618, -31.5196552, 3.1631885, -34.6748962, 34.6214180
39: -48.4590263, -10.6587391, -48.4651260, -10.5573292, -33.9086151, 33.8687592
40: -44.5720291, -17.6995373, -44.5756073, -17.5897255, -20.0755615, 19.9416122
41: -30.4736481, -4.1223521, -30.4780273, -4.0459800, -21.4448929, 21.3535919
42: -19.9911156, -0.2392702, -19.9959259, -0.2060175, -15.4366951, 15.3942585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=150, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1449

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1716

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9334827, upper bound: 9.9431566
time: 22.96 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9431565, upper bound: 9.9431566
time: 19.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 45.00 seconds
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 45.00
Output dim: 25, lower bound: -9.9201147, upper bound: 9.9431566
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 45.00
Output dim: 25, lower bound: -9.9297889, upper bound: 9.9431566
IS_B2_A1_B1, status: Status.VERIFIED, split count: 3, time: 45.00
Output dim: 25, lower bound: -9.9334788, upper bound: 9.9266852
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 45.00
Output dim: 25, lower bound: -9.9431528, upper bound: 9.9266852
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 45.00
Output dim: 25, lower bound: -9.9334827, upper bound: 9.9431566
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 45.00
Output dim: 25, lower bound: -9.9431565, upper bound: 9.9431566

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -22.4451962, 9.2846603, -22.4340382, 9.2704115, -31.6485138, 31.6576920
1: -12.0554152, 7.8009911, -12.0480719, 7.7940445, -19.8494606, 19.8490639
2: -11.6519909, 9.6231632, -11.6451244, 9.6036482, -18.7285690, 18.7477798
3: -17.6227417, 7.4591885, -17.6159210, 7.4468956, -23.4623718, 23.4852829
4: -19.6177425, 5.1698399, -19.6068039, 5.1465979, -22.3444290, 22.3839722
5: -15.5904741, 9.6746922, -15.5819578, 9.6526108, -24.0876694, 24.1023254
6: -31.9315586, -7.3925295, -31.9100628, -7.3988338, -19.8623657, 19.8164597
7: -21.6343269, 6.0272970, -21.6245842, 6.0158949, -26.2524261, 26.2646713
8: -23.6452332, 7.6028309, -23.6391888, 7.5870252, -29.4872894, 29.5226517
9: -13.7863579, 10.0452824, -13.7782612, 10.0358381, -20.6939087, 20.7104340
10: -13.9667845, 14.1332302, -13.9494143, 14.1107349, -27.5789871, 27.5873489
11: -10.2404003, 11.3934946, -10.2154827, 11.3920460, -17.5839500, 17.5488052
12: -23.2727547, 13.2458115, -23.2396812, 13.2280407, -34.3703003, 34.3656387
13: -25.3525791, 6.1416197, -25.3301601, 6.1154590, -30.9434509, 30.9680634
14: -26.2984581, 14.8969841, -26.2756958, 14.8760052, -39.5256348, 39.5286636
15: -10.0645018, 12.9944935, -10.0560980, 12.9809237, -21.6615372, 21.6772537
16: -20.9247665, 4.4976187, -20.9062080, 4.4895258, -25.2145691, 25.1630325
17: -23.0688667, 11.2582073, -23.0389366, 11.2438297, -34.3126984, 34.2971420
18: -11.2185574, 16.5879421, -11.1983128, 16.5825977, -26.9967499, 26.9485474
19: -7.2558699, 8.3630924, -7.2403688, 8.3544512, -14.6815224, 14.6628876
20: -6.5764618, 10.0338821, -6.5555162, 10.0164146, -15.3886070, 15.3738327
21: -7.5994520, 11.7834253, -7.5724955, 11.7708626, -18.3949394, 18.3677330
22: -5.0766134, 15.3600788, -5.0494328, 15.3481064, -18.3297615, 18.2851753
23: -2.9823508, 15.0398417, -2.9602981, 15.0238209, -15.7571487, 15.7187424
24: -5.3884840, 13.2760143, -5.3733621, 13.2632141, -14.4302711, 14.3892746
25: -0.9684792, 19.5898113, -0.9388757, 19.5609283, -15.4213333, 15.3586998
26: -12.1075401, 19.6761780, -12.0788546, 19.6686783, -31.7762184, 31.7550316
27: -9.4745750, 10.9478951, -9.4555836, 10.9391212, -19.7340622, 19.6967621
28: -4.1898651, 15.1198645, -4.1559772, 15.0994310, -17.5855217, 17.5514984
29: -3.9062996, 15.8990688, -3.8691149, 15.8837385, -16.2848434, 16.2120953
30: -10.8854942, 10.3947105, -10.8478136, 10.3786449, -17.7398300, 17.7056084
31: -6.8341393, 12.5650587, -6.8215961, 12.5481501, -18.7695312, 18.7594528
32: -26.4828148, -1.8236175, -26.4549599, -1.8484364, -22.6291885, 22.5994492
33: -43.5337524, -7.7945704, -43.4940300, -7.8292518, -28.7517166, 28.8036499
34: -36.1704788, -6.0327287, -36.1427841, -6.0448956, -22.9340668, 22.9407845
35: -26.7598476, 1.2251320, -26.7325687, 1.2067924, -24.8869705, 24.9004822
36: -27.0377274, 4.8297777, -27.0064335, 4.8107700, -31.3550415, 31.3522949
37: -44.1549339, -9.2123890, -44.0957832, -9.2469845, -28.4528198, 28.4703903
38: -31.4843845, 3.0972948, -31.4581566, 3.0747118, -34.5590973, 34.5554504
39: -48.3853302, -10.6641474, -48.3295822, -10.7262831, -33.6647339, 33.7414093
40: -44.5118637, -17.7022686, -44.4637909, -17.7446423, -19.8571587, 19.8593025
41: -30.4302273, -4.1262488, -30.3900356, -4.1510158, -21.2964325, 21.2806625
42: -19.9715843, -0.2439585, -19.9531364, -0.2530484, -15.3711014, 15.3522797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=126, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9135884, upper bound: 9.9407126
time: 16.87 seconds

## Relational analysis of IS_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9176711, upper bound: 9.9407126
time: 26.81 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -22.4481354, 9.2874146, -22.4452133, 9.2762814, -31.6563721, 31.6712646
1: -12.0563974, 7.8033915, -12.0554447, 7.7991996, -19.8555965, 19.8588371
2: -11.6572189, 9.6248913, -11.6555986, 9.6137867, -18.7319031, 18.7631989
3: -17.6246338, 7.4623098, -17.6234150, 7.4528723, -23.4690247, 23.5020523
4: -19.6292210, 5.1727457, -19.6260834, 5.1679912, -22.3634872, 22.3856430
5: -15.5912580, 9.6816559, -15.5905542, 9.6651525, -24.0987091, 24.1190109
6: -31.9333363, -7.3915739, -31.9154873, -7.3933082, -19.8691902, 19.8229980
7: -21.6359386, 6.0282941, -21.6335411, 6.0191946, -26.2485657, 26.2922363
8: -23.6504364, 7.6050148, -23.6481457, 7.5997295, -29.4951477, 29.5299683
9: -13.7882805, 10.0502377, -13.7848911, 10.0455532, -20.7058640, 20.7356644
10: -13.9687347, 14.1452208, -13.9646111, 14.1366158, -27.5990143, 27.6135254
11: -10.2442589, 11.3943386, -10.2388496, 11.3937130, -17.5892448, 17.5745697
12: -23.2784557, 13.2480869, -23.2532310, 13.2456522, -34.4150543, 34.3808899
13: -25.3661518, 6.1438484, -25.3549156, 6.1389933, -30.9803467, 30.9907837
14: -26.3023911, 14.9014730, -26.2966957, 14.8847332, -39.5326996, 39.5480347
15: -10.0650463, 13.0010986, -10.0630054, 12.9939594, -21.6698837, 21.6837006
16: -20.9295387, 4.5003352, -20.9248829, 4.4962392, -25.2179871, 25.2200699
17: -23.0752964, 11.2589941, -23.0690155, 11.2543526, -34.3296509, 34.3280106
18: -11.2206059, 16.5912247, -11.2145996, 16.5897312, -27.0026398, 26.9663467
19: -7.2607851, 8.3647480, -7.2546778, 8.3575249, -14.6900826, 14.6845016
20: -6.5807142, 10.0439091, -6.5755520, 10.0345898, -15.4038315, 15.4031715
21: -7.6046562, 11.7902012, -7.5985031, 11.7825356, -18.4039841, 18.4004097
22: -5.0799122, 15.3668318, -5.0745077, 15.3599739, -18.3302116, 18.3111725
23: -2.9868231, 15.0474529, -2.9812732, 15.0369101, -15.7647095, 15.7528458
24: -5.3907189, 13.2814264, -5.3876371, 13.2729321, -14.4313660, 14.4105301
25: -0.9721417, 19.6070824, -0.9681773, 19.5894089, -15.4163094, 15.4056396
26: -12.1115799, 19.6785774, -12.0938053, 19.6756554, -31.7872353, 31.7723827
27: -9.4763556, 10.9493256, -9.4708099, 10.9421959, -19.7376747, 19.7195854
28: -4.1948566, 15.1324358, -4.1873150, 15.1202402, -17.5995026, 17.5957870
29: -3.9094787, 15.9104118, -3.9038815, 15.9026041, -16.2795715, 16.2508392
30: -10.8893204, 10.4084835, -10.8844414, 10.4028206, -17.7465858, 17.7528152
31: -6.8383198, 12.5677061, -6.8324366, 12.5526600, -18.7792702, 18.7753220
32: -26.4918919, -1.8223505, -26.4726524, -1.8244867, -22.6526260, 22.6003876
33: -43.5568161, -7.7925644, -43.5353050, -7.7962337, -28.8148041, 28.8190460
34: -36.1729126, -6.0312214, -36.1513481, -6.0331745, -22.9488220, 22.9515266
35: -26.7681389, 1.2259808, -26.7500534, 1.2245622, -24.9140701, 24.9148331
36: -27.0469093, 4.8307271, -27.0258522, 4.8294053, -31.3827515, 31.3672333
37: -44.1838646, -9.2111063, -44.1459045, -9.2129517, -28.5170059, 28.4885101
38: -31.4912167, 3.0995913, -31.4726677, 3.0973620, -34.5885773, 34.5722580
39: -48.4254227, -10.6615839, -48.3972549, -10.6655836, -33.7655792, 33.7606506
40: -44.5367661, -17.7011986, -44.5052719, -17.7030468, -19.9233437, 19.8390541
41: -30.4434319, -4.1244555, -30.4149132, -4.1266494, -21.3309326, 21.2794189
42: -19.9746113, -0.2424169, -19.9607353, -0.2455592, -15.3826313, 15.3619652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=126, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9232624, upper bound: 9.9407126
time: 19.42 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9273449, upper bound: 9.9407126
time: 24.82 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -22.3974724, 9.2691422, -22.4612713, 9.2982979, -31.6277924, 31.6626129
1: -12.0294809, 7.7872052, -12.0602322, 7.8092546, -19.8387356, 19.8474369
2: -11.6339159, 9.6230850, -11.6684761, 9.6371632, -18.7426300, 18.7794342
3: -17.5880013, 7.4449749, -17.6356068, 7.4732766, -23.4722443, 23.5081406
4: -19.5798492, 5.1502542, -19.6070938, 5.1823902, -22.3489075, 22.3330154
5: -15.5721626, 9.6806545, -15.6291981, 9.6958570, -24.1071930, 24.1519623
6: -31.9310074, -7.4049382, -31.9449844, -7.3524966, -19.8688393, 19.8541107
7: -21.6050282, 6.0195570, -21.6517639, 6.0389071, -26.2497711, 26.3279114
8: -23.5981426, 7.5780978, -23.6328754, 7.6130219, -29.4805603, 29.4973755
9: -13.7604084, 10.0291386, -13.7825317, 10.0574427, -20.7044296, 20.7239838
10: -13.9586105, 14.1374454, -13.9801664, 14.1551056, -27.6092987, 27.6272278
11: -10.2198830, 11.3628654, -10.2661657, 11.3851633, -17.5503693, 17.5938110
12: -23.2892876, 13.2367840, -23.3053875, 13.3280287, -34.5042572, 34.4142609
13: -25.3082123, 6.1040702, -25.3503532, 6.1628222, -30.9617462, 30.9429550
14: -26.2838497, 14.8964243, -26.3705215, 14.9093704, -39.5459290, 39.6152420
15: -10.0518074, 12.9868851, -10.0793428, 13.0030823, -21.6898041, 21.6501122
16: -20.9092045, 4.4862061, -20.9380379, 4.5089970, -25.1740417, 25.3028069
17: -23.0551128, 11.2477999, -23.1190414, 11.2628498, -34.3179626, 34.3668404
18: -11.1851158, 16.5428181, -11.2275686, 16.5774651, -26.9636993, 26.9598083
19: -7.2405748, 8.3486443, -7.2939520, 8.3605766, -14.6682854, 14.7128944
20: -6.5605412, 10.0271854, -6.6100998, 10.0432730, -15.3904610, 15.4343567
21: -7.5793972, 11.7590675, -7.6317234, 11.7802162, -18.3765068, 18.4165306
22: -5.0597382, 15.3400869, -5.1055484, 15.3512230, -18.3289032, 18.3154869
23: -2.9577813, 15.0055113, -3.0252662, 15.0306168, -15.7221489, 15.7842827
24: -5.3628025, 13.2466030, -5.4104757, 13.2694454, -14.3900108, 14.4315071
25: -0.9493513, 19.5758247, -1.0262599, 19.6000805, -15.3772888, 15.4804077
26: -12.0850163, 19.6428566, -12.1338825, 19.7150841, -31.8001003, 31.7767391
27: -9.4441757, 10.9148312, -9.4895935, 10.9371395, -19.6947327, 19.7194824
28: -4.1668215, 15.0968313, -4.2314606, 15.1200027, -17.5662727, 17.6276207
29: -3.8893175, 15.8761940, -3.9523959, 15.8923349, -16.2410126, 16.2887917
30: -10.8664970, 10.3747349, -10.9023762, 10.3992939, -17.7224922, 17.7538490
31: -6.8106508, 12.5376453, -6.8998041, 12.5615368, -18.7528152, 18.8250160
32: -26.4795303, -1.8411322, -26.4992695, -1.7840214, -22.6636429, 22.6306152
33: -43.5008850, -7.8281431, -43.5393448, -7.7234621, -28.9143219, 28.7837143
34: -36.1507072, -6.0454016, -36.1787529, -5.9559698, -23.0346146, 22.9560165
35: -26.7439575, 1.2033100, -26.7711487, 1.2822881, -24.9770508, 24.9100952
36: -27.0307026, 4.8131094, -27.0576878, 4.8944359, -31.4414215, 31.3811340
37: -44.1387329, -9.2417717, -44.1842308, -9.0805511, -28.6583405, 28.4893723
38: -31.4631996, 3.0826364, -31.4965172, 3.1608462, -34.6240463, 34.5791550
39: -48.3488617, -10.7035189, -48.4001465, -10.5600777, -33.8748627, 33.7081299
40: -44.4934921, -17.7317047, -44.5287857, -17.5921116, -20.0258636, 19.8584442
41: -30.4302826, -4.1429214, -30.4532757, -4.0494184, -21.4085083, 21.3154373
42: -19.9765587, -0.2496834, -19.9884758, -0.2097096, -15.4078617, 15.3866444

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=126, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=150, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1449

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 713

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9407088, upper bound: 9.9201576
time: 22.72 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9407088, upper bound: 9.9242410
time: 19.84 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -22.4474220, 9.2955961, -22.4797859, 9.2959423, -31.6750793, 31.7165451
1: -12.0559044, 7.8046379, -12.0676022, 7.8067698, -19.8626747, 19.8722401
2: -11.6529751, 9.6344194, -11.6714563, 9.6298447, -18.7555542, 18.7921677
3: -17.6236458, 7.4681034, -17.6497498, 7.4713240, -23.4841232, 23.5346603
4: -19.6205750, 5.1739793, -19.6174965, 5.1643534, -22.3631973, 22.3981934
5: -15.5913391, 9.6903419, -15.6318703, 9.6881857, -24.1220856, 24.1675186
6: -31.9496727, -7.3910809, -31.9501762, -7.3536406, -19.9130707, 19.8469162
7: -21.6355476, 6.0353775, -21.6604252, 6.0389352, -26.2749863, 26.3371201
8: -23.6455536, 7.6067100, -23.6532173, 7.6034575, -29.5032806, 29.5536423
9: -13.7890434, 10.0490103, -13.7934542, 10.0515404, -20.7061157, 20.7355118
10: -13.9694586, 14.1402378, -13.9694853, 14.1355057, -27.6087418, 27.6206360
11: -10.2429657, 11.3943729, -10.2466259, 11.4016361, -17.5924034, 17.5926056
12: -23.2979431, 13.2474623, -23.2979717, 13.3171482, -34.4840393, 34.4185104
13: -25.3646774, 6.1447072, -25.3647060, 6.1443458, -30.9822388, 31.0045319
14: -26.3038883, 14.9144764, -26.3576317, 14.9125347, -39.5662231, 39.6279831
15: -10.0667877, 12.9957294, -10.0790987, 12.9938707, -21.7005539, 21.6861458
16: -20.9264984, 4.5015764, -20.9306221, 4.5057206, -25.2165527, 25.2431870
17: -23.0746994, 11.2591124, -23.0981789, 11.2552223, -34.3299217, 34.3572922
18: -11.2222662, 16.5894051, -11.2169933, 16.5987091, -27.0332260, 26.9693985
19: -7.2614384, 8.3715992, -7.2847090, 8.3713264, -14.7019272, 14.7145004
20: -6.5806341, 10.0441084, -6.5948391, 10.0402679, -15.4149208, 15.4253254
21: -7.6048222, 11.7919140, -7.6112347, 11.7903595, -18.4190102, 18.4149551
22: -5.0816793, 15.3597155, -5.0852323, 15.3548698, -18.3688583, 18.2921257
23: -2.9870071, 15.0517721, -3.0089083, 15.0478573, -15.7815208, 15.7788200
24: -5.3910866, 13.2844353, -5.4005384, 13.2846088, -14.4507599, 14.4239883
25: -0.9719772, 19.6078377, -1.0012193, 19.6001759, -15.4529305, 15.4385338
26: -12.1249628, 19.6782379, -12.1273041, 19.7294407, -31.8544044, 31.8055420
27: -9.4788132, 10.9554863, -9.4794950, 10.9570465, -19.7546387, 19.7228088
28: -4.1967459, 15.1325569, -4.2058644, 15.1261892, -17.6157417, 17.6166153
29: -3.9117031, 15.9043589, -3.9219017, 15.8968887, -16.3025284, 16.2551041
30: -10.8893089, 10.4000187, -10.8693027, 10.3962669, -17.7626305, 17.7348824
31: -6.8392754, 12.5815201, -6.8942633, 12.5822487, -18.8032494, 18.8471909
32: -26.5028744, -1.8219261, -26.4989452, -1.8042188, -22.6877670, 22.6330338
33: -43.5554886, -7.7920694, -43.5409012, -7.7525654, -28.8614197, 28.8418808
34: -36.1955833, -6.0311275, -36.1953735, -5.9661942, -23.0360870, 22.9794769
35: -26.7796307, 1.2254372, -26.7772541, 1.2666783, -24.9678421, 24.9361191
36: -27.0601654, 4.8307238, -27.0578728, 4.8777618, -31.4437408, 31.4011688
37: -44.1942101, -9.2108870, -44.1809006, -9.1117496, -28.6290131, 28.5363617
38: -31.5035267, 3.0992150, -31.5030403, 3.1400118, -34.6435394, 34.6022568
39: -48.4162369, -10.6614208, -48.3928680, -10.6182461, -33.8047180, 33.7916870
40: -44.5453644, -17.7006798, -44.5311928, -17.6315117, -20.0056801, 19.9047852
41: -30.4594669, -4.1243267, -30.4514427, -4.0707264, -21.4080162, 21.3289680
42: -19.9877548, -0.2409544, -19.9877453, -0.2137475, -15.4236069, 15.3830013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=126, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=150, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1449

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 713

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9310391, upper bound: 9.9366317
time: 25.04 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9310391, upper bound: 9.9407126
time: 19.87 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -22.4503880, 9.2983665, -22.4909782, 9.3018370, -31.6829681, 31.7301178
1: -12.0568857, 7.8070250, -12.0749636, 7.8119121, -19.8687973, 19.8819885
2: -11.6582327, 9.6361217, -11.6819172, 9.6399498, -18.7589111, 18.8076172
3: -17.6255493, 7.4712133, -17.6572418, 7.4772925, -23.4907837, 23.5514603
4: -19.6320229, 5.1769013, -19.6367588, 5.1857605, -22.3822327, 22.3998260
5: -15.5921307, 9.6972866, -15.6404896, 9.7007389, -24.1331711, 24.1842728
6: -31.9514351, -7.3901124, -31.9555798, -7.3480930, -19.9198761, 19.8534927
7: -21.6372032, 6.0363817, -21.6693726, 6.0422525, -26.2711792, 26.3647003
8: -23.6508045, 7.6089253, -23.6621895, 7.6161981, -29.5111084, 29.5609894
9: -13.7909698, 10.0539780, -13.8000469, 10.0612564, -20.7180634, 20.7608032
10: -13.9713850, 14.1522417, -13.9847021, 14.1613483, -27.6287613, 27.6468124
11: -10.2468615, 11.3952255, -10.2700195, 11.4033070, -17.5977058, 17.6184044
12: -23.3036842, 13.2497749, -23.3114777, 13.3347778, -34.5288544, 34.4336395
13: -25.3782673, 6.1469436, -25.3894768, 6.1678309, -31.0191040, 31.0273056
14: -26.3078918, 14.9189510, -26.3786354, 14.9212246, -39.5732880, 39.6474533
15: -10.0673170, 13.0023403, -10.0860367, 13.0069256, -21.7089386, 21.6926117
16: -20.9312153, 4.5042706, -20.9493008, 4.5124540, -25.2200012, 25.3002052
17: -23.0811443, 11.2598724, -23.1282387, 11.2657585, -34.3469009, 34.3881111
18: -11.2243633, 16.5925999, -11.2332592, 16.6058235, -27.0391159, 26.9872284
19: -7.2663593, 8.3732491, -7.2990065, 8.3743896, -14.7105007, 14.7360897
20: -6.5848866, 10.0541353, -6.6148777, 10.0584497, -15.4301147, 15.4546585
21: -7.6100359, 11.7986937, -7.6372461, 11.8020496, -18.4280510, 18.4476242
22: -5.0849886, 15.3664751, -5.1103001, 15.3667269, -18.3693275, 18.3181572
23: -2.9915171, 15.0594044, -3.0298719, 15.0609474, -15.7890854, 15.8129158
24: -5.3933334, 13.2898998, -5.4148006, 13.2943316, -14.4518661, 14.4452362
25: -0.9756269, 19.6250668, -1.0305486, 19.6286736, -15.4479008, 15.4854774
26: -12.1289854, 19.6806831, -12.1422958, 19.7364101, -31.8653946, 31.8229790
27: -9.4806061, 10.9568996, -9.4947147, 10.9601192, -19.7582741, 19.7456322
28: -4.2017465, 15.1451464, -4.2371826, 15.1470022, -17.6297035, 17.6608543
29: -3.9148779, 15.9156761, -3.9567027, 15.9157505, -16.2972832, 16.2938995
30: -10.8931217, 10.4138184, -10.9059401, 10.4204464, -17.7693710, 17.7821121
31: -6.8434310, 12.5841475, -6.9050937, 12.5867577, -18.8130226, 18.8630409
32: -26.5119495, -1.8206410, -26.5166416, -1.7802458, -22.7111893, 22.6340256
33: -43.5785675, -7.7900448, -43.5821533, -7.7195883, -28.9245071, 28.8572922
34: -36.1979904, -6.0295954, -36.2040024, -5.9545016, -23.0508423, 22.9901695
35: -26.7879486, 1.2262998, -26.7947273, 1.2844582, -24.9949493, 24.9504471
36: -27.0693970, 4.8316197, -27.0773067, 4.8964357, -31.4714203, 31.4161224
37: -44.2230682, -9.2096386, -44.2310448, -9.0777283, -28.6932144, 28.5544815
38: -31.5103836, 3.1014590, -31.5174942, 3.1626892, -34.6730728, 34.6189537
39: -48.4563751, -10.6589031, -48.4605484, -10.5576324, -33.9054871, 33.8108673
40: -44.5703049, -17.6996117, -44.5726585, -17.5898876, -20.0718994, 19.8845139
41: -30.4726505, -4.1225290, -30.4763451, -4.0463281, -21.4425278, 21.3277550
42: -19.9907780, -0.2393985, -19.9953556, -0.2062697, -15.4351387, 15.3926811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=126, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=150, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 713
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1449

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 713

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9407126, upper bound: 9.9366317
time: 13.85 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9407126, upper bound: 9.9407126
time: 26.89 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 42.96 seconds
IS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 42.96
Output dim: 25, lower bound: -9.9135884, upper bound: 9.9407126
IS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 42.96
Output dim: 25, lower bound: -9.9176711, upper bound: 9.9407126
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 42.96
Output dim: 25, lower bound: -9.9232624, upper bound: 9.9407126
IS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 42.96
Output dim: 25, lower bound: -9.9273449, upper bound: 9.9407126
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 42.96
Output dim: 25, lower bound: -9.9407088, upper bound: 9.9201576
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 42.96
Output dim: 25, lower bound: -9.9407088, upper bound: 9.9242410
IS_B2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 42.96
Output dim: 25, lower bound: -9.9310391, upper bound: 9.9366317
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 42.96
Output dim: 25, lower bound: -9.9310391, upper bound: 9.9407126
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 42.96
Output dim: 25, lower bound: -9.9407126, upper bound: 9.9366317
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 42.96
Output dim: 25, lower bound: -9.9407126, upper bound: 9.9407126

## BFS IS instance: IS_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -22.4150486, 9.2827587, -22.3802166, 9.2376337, -31.5787659, 31.5989075
1: -12.0350819, 7.7985625, -12.0130081, 7.7711802, -19.8062630, 19.8115711
2: -11.6287575, 9.6212454, -11.6051083, 9.5716476, -18.6694031, 18.7028923
3: -17.5994873, 7.4570761, -17.5760460, 7.4205008, -23.4133453, 23.4441986
4: -19.6039581, 5.1670442, -19.5822659, 5.1200819, -22.3024139, 22.3559341
5: -15.5675192, 9.6729298, -15.5423737, 9.6210880, -24.0328217, 24.0611267
6: -31.9293919, -7.3987041, -31.9021912, -7.4072499, -19.8590393, 19.7996788
7: -21.5997200, 6.0236998, -21.5662231, 5.9802094, -26.1740494, 26.1972504
8: -23.6158981, 7.6001124, -23.5897675, 7.5487289, -29.4180908, 29.4672623
9: -13.7821531, 10.0432854, -13.7702675, 10.0283127, -20.6754837, 20.7028770
10: -13.9624310, 14.1299629, -13.9272518, 14.0989761, -27.5532379, 27.5596161
11: -10.2294960, 11.3895082, -10.1915283, 11.3782225, -17.5601864, 17.5216217
12: -23.2700233, 13.2154751, -23.2046070, 13.1743526, -34.3162079, 34.3007050
13: -25.3484306, 6.1376491, -25.3271465, 6.1017065, -30.9230957, 30.9669342
14: -26.2781582, 14.8950138, -26.2362480, 14.8506556, -39.4758911, 39.4853668
15: -10.0536766, 12.9788160, -10.0256834, 12.9487839, -21.6178284, 21.6300926
16: -20.9100590, 4.4938293, -20.8751030, 4.4734554, -25.1797867, 25.1236000
17: -23.0579796, 11.2545834, -23.0165081, 11.2307739, -34.2887535, 34.2710915
18: -11.2154980, 16.5820370, -11.1868973, 16.5764065, -26.9861832, 26.9230347
19: -7.2524157, 8.3624077, -7.2274380, 8.3484745, -14.6697845, 14.6470909
20: -6.5694313, 10.0323782, -6.5402279, 10.0009022, -15.3663483, 15.3536949
21: -7.5958557, 11.7815266, -7.5617342, 11.7609501, -18.3785591, 18.3480835
22: -5.0706501, 15.3353882, -5.0244503, 15.3045883, -18.2793350, 18.2292519
23: -2.9783015, 15.0381632, -2.9470701, 15.0161676, -15.7390594, 15.7019997
24: -5.3860512, 13.2720528, -5.3592749, 13.2552595, -14.4187584, 14.3711853
25: -0.9637861, 19.5736637, -0.9196200, 19.5307884, -15.3932056, 15.3243866
26: -12.1010561, 19.6484985, -12.0427694, 19.6201839, -31.7212410, 31.6912689
27: -9.4714565, 10.9457197, -9.4458427, 10.9274979, -19.7315979, 19.6761131
28: -4.1860132, 15.1180992, -4.1373954, 15.0927801, -17.5681419, 17.5294495
29: -3.8995318, 15.8834076, -3.8442202, 15.8563156, -16.2495651, 16.1667404
30: -10.8832493, 10.3864059, -10.8295088, 10.3607025, -17.7203369, 17.6793137
31: -6.8267641, 12.5626316, -6.8020716, 12.5315657, -18.7496796, 18.7364120
32: -26.4809303, -1.8277645, -26.4487343, -1.8602238, -22.6167068, 22.5810318
33: -43.5311165, -7.8167629, -43.4543266, -7.8690004, -28.7196884, 28.7527237
34: -36.1677094, -6.0514421, -36.1108704, -6.0768728, -22.9110947, 22.8981590
35: -26.7566910, 1.2114630, -26.7029991, 1.1843271, -24.8662186, 24.8632965
36: -27.0341148, 4.8128152, -26.9825573, 4.7799459, -31.3204956, 31.3123016
37: -44.1512985, -9.2428627, -44.0488892, -9.2995377, -28.3969421, 28.3932419
38: -31.4790955, 3.0944614, -31.4408989, 3.0641208, -34.5432167, 34.5353622
39: -48.3831558, -10.6773386, -48.3064613, -10.7509909, -33.6434631, 33.7117920
40: -44.5095520, -17.7122173, -44.4509506, -17.7638893, -19.8360214, 19.8346519
41: -30.4281826, -4.1332932, -30.3805351, -4.1593394, -21.2853546, 21.2623177
42: -19.9698200, -0.2489738, -19.9454975, -0.2597551, -15.3603783, 15.3390064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=125, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_B1_A2_B1_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9080613, upper bound: 9.9351053
time: 22.31 seconds

## Relational analysis of IS_B1_A2_B1_B1_B2

### Relational analysis result of IS_B1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9080613, upper bound: 9.9351904
time: 20.62 seconds

## BFS IS instance: IS_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -22.4417686, 9.2835388, -22.4285698, 9.2684946, -31.6466980, 31.6496277
1: -12.0514326, 7.7998080, -12.0417786, 7.7920876, -19.8435211, 19.8415871
2: -11.6487474, 9.6219177, -11.6400890, 9.6015339, -18.7235641, 18.7254677
3: -17.6200638, 7.4583387, -17.6116982, 7.4455242, -23.4581604, 23.4734039
4: -19.6168690, 5.1682281, -19.6052780, 5.1439166, -22.3406448, 22.3682632
5: -15.5878830, 9.6740761, -15.5778599, 9.6515551, -24.0837402, 24.0781326
6: -31.9230804, -7.3938761, -31.8961201, -7.4011016, -19.8516312, 19.8238220
7: -21.6266232, 6.0259261, -21.6125584, 6.0136185, -26.2436600, 26.2320175
8: -23.6417198, 7.6013789, -23.6335220, 7.5846143, -29.4812317, 29.4993134
9: -13.7855549, 10.0424461, -13.7768745, 10.0310564, -20.6988068, 20.7031021
10: -13.9657726, 14.1316853, -13.9477015, 14.1082268, -27.5662079, 27.5765076
11: -10.2333879, 11.3925800, -10.2039137, 11.3905249, -17.5743618, 17.5327759
12: -23.2713432, 13.2418442, -23.2373199, 13.2216282, -34.3494720, 34.3594437
13: -25.3512363, 6.1303124, -25.3279076, 6.0964942, -30.9397125, 30.9526291
14: -26.2955017, 14.8957596, -26.2708397, 14.8739424, -39.5222015, 39.5213089
15: -10.0619698, 12.9845219, -10.0518551, 12.9651480, -21.6358643, 21.6613045
16: -20.9169159, 4.4955511, -20.8937168, 4.4860697, -25.2048264, 25.1491432
17: -23.0668602, 11.2565317, -23.0356102, 11.2409859, -34.3078461, 34.2921410
18: -11.2164927, 16.5866165, -11.1948662, 16.5803795, -26.9887390, 26.9523544
19: -7.2548857, 8.3627338, -7.2387133, 8.3538647, -14.6785145, 14.6588345
20: -6.5748978, 10.0334425, -6.5529213, 10.0156679, -15.3813972, 15.3624001
21: -7.5977688, 11.7829046, -7.5697212, 11.7699814, -18.3889694, 18.3607750
22: -5.0743246, 15.3518867, -5.0456548, 15.3343706, -18.3060913, 18.2735672
23: -2.9811335, 15.0394096, -2.9583879, 15.0231018, -15.7540207, 15.7133408
24: -5.3879151, 13.2749996, -5.3724365, 13.2615833, -14.4259605, 14.3859673
25: -0.9664679, 19.5841599, -0.9355593, 19.5513992, -15.3994179, 15.3472443
26: -12.1056690, 19.6707191, -12.0757885, 19.6598949, -31.7655640, 31.7465076
27: -9.4688187, 10.9470100, -9.4461498, 10.9376373, -19.7220268, 19.7067490
28: -4.1888714, 15.1192522, -4.1543241, 15.0983963, -17.5794563, 17.5455246
29: -3.9043293, 15.8930893, -3.8658237, 15.8739338, -16.2667084, 16.2032166
30: -10.8840141, 10.3934565, -10.8454142, 10.3765678, -17.7281685, 17.7011108
31: -6.8316708, 12.5641184, -6.8174744, 12.5466003, -18.7630806, 18.7516708
32: -26.4793472, -1.8244953, -26.4492226, -1.8498759, -22.6222000, 22.5993881
33: -43.5322609, -7.7968159, -43.4915962, -7.8326263, -28.7222824, 28.7942123
34: -36.1689682, -6.0343313, -36.1402893, -6.0474944, -22.9057617, 22.9326363
35: -26.7583351, 1.2213283, -26.7300797, 1.2008457, -24.8623886, 24.8922272
36: -27.0362701, 4.8251753, -27.0040150, 4.8032207, -31.3438263, 31.3439331
37: -44.1534119, -9.2142410, -44.0932388, -9.2501125, -28.4187546, 28.4643555
38: -31.4823360, 3.0967188, -31.4547997, 3.0737867, -34.5561218, 34.5515175
39: -48.3842735, -10.6679459, -48.3278313, -10.7326469, -33.6486359, 33.7323303
40: -44.5101471, -17.7042522, -44.4609299, -17.7479248, -19.8507538, 19.8545227
41: -30.4269371, -4.1276627, -30.3845348, -4.1533031, -21.2907028, 21.2765884
42: -19.9696312, -0.2451735, -19.9499016, -0.2550864, -15.3666115, 15.3470612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=125, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 663

## Relational analysis of IS_B1_A2_B1_B2_B1

### Relational analysis result of IS_B1_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9121429, upper bound: 9.9351053
time: 16.31 seconds

## Relational analysis of IS_B1_A2_B1_B2_B2

### Relational analysis result of IS_B1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9121429, upper bound: 9.9351904
time: 21.40 seconds

## BFS IS instance: IS_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -22.4180489, 9.2855434, -22.3913746, 9.2434921, -31.5866699, 31.6124878
1: -12.0360641, 7.8009162, -12.0203838, 7.7763257, -19.8123894, 19.8213005
2: -11.6339951, 9.6229572, -11.6155949, 9.5817814, -18.6727371, 18.7183037
3: -17.6013393, 7.4601574, -17.5835629, 7.4265151, -23.4199982, 23.4609680
4: -19.6154175, 5.1698885, -19.6015587, 5.1415110, -22.3214493, 22.3575897
5: -15.5683422, 9.6798477, -15.5509968, 9.6336613, -24.0438995, 24.0778122
6: -31.9311600, -7.3977394, -31.9075642, -7.4016924, -19.8658638, 19.8062286
7: -21.6013947, 6.0247564, -21.5751534, 5.9834681, -26.1702194, 26.2248459
8: -23.6211510, 7.6023240, -23.5987206, 7.5614591, -29.4258270, 29.4745255
9: -13.7840672, 10.0482578, -13.7768698, 10.0380459, -20.6874313, 20.7281189
10: -13.9643364, 14.1419945, -13.9424677, 14.1248531, -27.5732498, 27.5857849
11: -10.2333803, 11.3903294, -10.2148972, 11.3799000, -17.5654793, 17.5474014
12: -23.2757530, 13.2177496, -23.2181969, 13.1919765, -34.3609161, 34.3159943
13: -25.3620491, 6.1398458, -25.3519077, 6.1251965, -30.9600372, 30.9896851
14: -26.2820911, 14.8994894, -26.2572327, 14.8593912, -39.4829102, 39.5048141
15: -10.0542364, 12.9853773, -10.0326252, 12.9618101, -21.6261902, 21.6365318
16: -20.9147663, 4.4965405, -20.8938065, 4.4801855, -25.1832123, 25.1806717
17: -23.0644379, 11.2553530, -23.0465164, 11.2412243, -34.3056641, 34.3018684
18: -11.2176065, 16.5852947, -11.2032051, 16.5835114, -26.9920502, 26.9408493
19: -7.2573214, 8.3640690, -7.2417431, 8.3515320, -14.6783600, 14.6687031
20: -6.5736661, 10.0424156, -6.5602837, 10.0190821, -15.3815613, 15.3830566
21: -7.6010513, 11.7883081, -7.5877724, 11.7726269, -18.3876190, 18.3807678
22: -5.0739384, 15.3421535, -5.0495071, 15.3164539, -18.2798233, 18.2552376
23: -2.9828091, 15.0457792, -2.9680300, 15.0292330, -15.7466125, 15.7360992
24: -5.3883038, 13.2774754, -5.3735809, 13.2649689, -14.4198494, 14.3924599
25: -0.9674463, 19.5909119, -0.9489512, 19.5592728, -15.3881741, 15.3713341
26: -12.1050491, 19.6509132, -12.0577393, 19.6271706, -31.7322197, 31.7086525
27: -9.4732370, 10.9471626, -9.4610519, 10.9305792, -19.7351799, 19.6989403
28: -4.1910257, 15.1306915, -4.1687374, 15.1135798, -17.5821075, 17.5737038
29: -3.9026937, 15.8947487, -3.8789825, 15.8751688, -16.2443008, 16.2055054
30: -10.8870754, 10.4001961, -10.8661518, 10.3848829, -17.7270699, 17.7265091
31: -6.8309617, 12.5652599, -6.8129206, 12.5360661, -18.7594147, 18.7522888
32: -26.4899883, -1.8264666, -26.4664078, -1.8362370, -22.6400909, 22.5820312
33: -43.5541763, -7.8147445, -43.4956436, -7.8359466, -28.7827454, 28.7681732
34: -36.1701202, -6.0498877, -36.1194305, -6.0651460, -22.9258652, 22.9088745
35: -26.7649708, 1.2123303, -26.7203979, 1.2020907, -24.8932495, 24.8775940
36: -27.0433273, 4.8136868, -27.0020027, 4.7986579, -31.3482208, 31.3272400
37: -44.1802101, -9.2416096, -44.0990295, -9.2655382, -28.4611130, 28.4113693
38: -31.4859390, 3.0967970, -31.4553318, 3.0868025, -34.5727425, 34.5521278
39: -48.4232941, -10.6747885, -48.3740959, -10.6902065, -33.7443390, 33.7310333
40: -44.5344620, -17.7111492, -44.4923935, -17.7222404, -19.9022102, 19.8143997
41: -30.4413128, -4.1314692, -30.4054375, -4.1349463, -21.3198624, 21.2611046
42: -19.9728775, -0.2474284, -19.9530964, -0.2522750, -15.3719139, 15.3486938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=125, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_B1_A2_B2_B1_A1

### Relational analysis result of IS_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9176495, upper bound: 9.9351904
time: 28.34 seconds

## Relational analysis of IS_B1_A2_B2_B1_A2

### Relational analysis result of IS_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9177344, upper bound: 9.9351904
time: 25.14 seconds

## BFS IS instance: IS_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -22.4446850, 9.2863064, -22.4397392, 9.2743950, -31.6545715, 31.6632385
1: -12.0524178, 7.8021989, -12.0491486, 7.7972488, -19.8496666, 19.8513470
2: -11.6539860, 9.6236458, -11.6505594, 9.6116524, -18.7268982, 18.7408943
3: -17.6219387, 7.4614592, -17.6191654, 7.4515009, -23.4647827, 23.4901886
4: -19.6283150, 5.1711626, -19.6245480, 5.1653175, -22.3596420, 22.3698883
5: -15.5886717, 9.6810532, -15.5864811, 9.6641130, -24.0948486, 24.0948410
6: -31.9248581, -7.3929067, -31.9015617, -7.3955493, -19.8584824, 19.8303375
7: -21.6282787, 6.0269523, -21.6215172, 6.0169420, -26.2398148, 26.2595749
8: -23.6469097, 7.6035843, -23.6424866, 7.5972986, -29.4889679, 29.5066071
9: -13.7874680, 10.0473881, -13.7835236, 10.0407925, -20.7108078, 20.7283401
10: -13.9677401, 14.1437111, -13.9629164, 14.1340866, -27.5862350, 27.6026382
11: -10.2372541, 11.3934345, -10.2272892, 11.3921785, -17.5796528, 17.5585365
12: -23.2770557, 13.2441292, -23.2508698, 13.2392197, -34.3941956, 34.3747101
13: -25.3648300, 6.1325154, -25.3527069, 6.1199970, -30.9766083, 30.9753876
14: -26.2994614, 14.9002628, -26.2918129, 14.8826094, -39.5292206, 39.5407562
15: -10.0624847, 12.9911366, -10.0588188, 12.9781580, -21.6441956, 21.6677475
16: -20.9216347, 4.4982557, -20.9124031, 4.4927864, -25.2082291, 25.2062531
17: -23.0732441, 11.2572975, -23.0656357, 11.2514935, -34.3247375, 34.3229332
18: -11.2185545, 16.5898495, -11.2111187, 16.5875034, -26.9946365, 26.9701691
19: -7.2597799, 8.3643913, -7.2530117, 8.3569384, -14.6870956, 14.6804390
20: -6.5791430, 10.0434647, -6.5729790, 10.0338516, -15.3965950, 15.3917637
21: -7.6029892, 11.7896786, -7.5957460, 11.7816753, -18.3980103, 18.3934784
22: -5.0775909, 15.3586550, -5.0707173, 15.3462400, -18.3065529, 18.2995796
23: -2.9856687, 15.0470390, -2.9793372, 15.0361843, -15.7615776, 15.7474442
24: -5.3901501, 13.2804413, -5.3867269, 13.2712984, -14.4270477, 14.4072304
25: -0.9701071, 19.6014099, -0.9649119, 19.5798893, -15.3943939, 15.3942070
26: -12.1097021, 19.6731300, -12.0907679, 19.6668739, -31.7765770, 31.7638969
27: -9.4706163, 10.9484310, -9.4613552, 10.9407082, -19.7256241, 19.7295532
28: -4.1938610, 15.1318359, -4.1856499, 15.1192017, -17.5934448, 17.5897903
29: -3.9074907, 15.9044199, -3.9005842, 15.8927946, -16.2614365, 16.2419510
30: -10.8878765, 10.4072485, -10.8820419, 10.4007473, -17.7349129, 17.7483406
31: -6.8358397, 12.5667553, -6.8283291, 12.5511131, -18.7728462, 18.7675247
32: -26.4884167, -1.8232031, -26.4669304, -1.8258991, -22.6456299, 22.6003647
33: -43.5553131, -7.7947946, -43.5328369, -7.7996445, -28.7853775, 28.8096542
34: -36.1713982, -6.0327458, -36.1489029, -6.0357523, -22.9205322, 22.9433289
35: -26.7666149, 1.2222328, -26.7475166, 1.2186060, -24.8894577, 24.9065475
36: -27.0454788, 4.8260527, -27.0234394, 4.8219066, -31.3715668, 31.3589706
37: -44.1823044, -9.2129622, -44.1433411, -9.2160625, -28.4829178, 28.4824753
38: -31.4892216, 3.0990005, -31.4692535, 3.0964451, -34.5856667, 34.5682526
39: -48.4243546, -10.6653709, -48.3954964, -10.6719284, -33.7494812, 33.7515259
40: -44.5350800, -17.7031822, -44.5023537, -17.7063503, -19.9169235, 19.8342972
41: -30.4401245, -4.1258302, -30.4094181, -4.1289177, -21.3251915, 21.2753525
42: -19.9726639, -0.2436299, -19.9574966, -0.2475991, -15.3781395, 15.3567448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=125, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1783
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 713
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1343
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1709
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 939
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_B1_A2_B2_B2_A1

### Relational analysis result of IS_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9217288, upper bound: 9.9351904
time: 29.12 seconds

## Relational analysis of IS_B1_A2_B2_B2_A2

### Relational analysis result of IS_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9218141, upper bound: 9.9351904
time: 24.81 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -22.3436966, 9.2363663, -22.4311104, 9.2964172, -31.5689850, 31.5928268
1: -11.9944172, 7.7643437, -12.0398064, 7.8068023, -19.8012199, 19.8041496
2: -11.5938969, 9.5911121, -11.6452217, 9.6352415, -18.6977615, 18.7202377
3: -17.5481319, 7.4186068, -17.6123009, 7.4711609, -23.4311295, 23.4591064
4: -19.5553780, 5.1238070, -19.5933609, 5.1794906, -22.3208847, 22.2909698
5: -15.5326424, 9.6491652, -15.6062727, 9.6940432, -24.0660019, 24.0971222
6: -31.9230957, -7.4133253, -31.9428329, -7.3586969, -19.8519974, 19.8507881
7: -21.5466156, 5.9838667, -21.6172390, 6.0353494, -26.1824341, 26.2494965
8: -23.5487003, 7.5398455, -23.6034336, 7.6102958, -29.4251404, 29.4279938
9: -13.7524195, 10.0216255, -13.7783804, 10.0554390, -20.6969147, 20.7054825
10: -13.9364767, 14.1256714, -13.9758339, 14.1518612, -27.5815887, 27.6014709
11: -10.1959171, 11.3490143, -10.2552500, 11.3812218, -17.5232658, 17.5700836
12: -23.2543068, 13.1831455, -23.3026505, 13.2977343, -34.4393616, 34.3601837
13: -25.3051586, 6.0902672, -25.3461647, 6.1587944, -30.9607391, 30.9227524
14: -26.2444305, 14.8710632, -26.3501358, 14.9073362, -39.5026398, 39.5652542
15: -10.0213614, 12.9547520, -10.0685167, 12.9873695, -21.6425934, 21.6064568
16: -20.8781528, 4.4701128, -20.9232521, 4.5051851, -25.1347580, 25.2679596
17: -23.0326405, 11.2347727, -23.1080894, 11.2592030, -34.2918434, 34.3428612
18: -11.1736965, 16.5366096, -11.2245483, 16.5716419, -26.9382095, 26.9492722
19: -7.2276459, 8.3426723, -7.2904644, 8.3599110, -14.6524620, 14.7011337
20: -6.5452738, 10.0116711, -6.6030293, 10.0417709, -15.3703079, 15.4120903
21: -7.5686722, 11.7491379, -7.6280746, 11.7783384, -18.3568878, 18.4001312
22: -5.0347185, 15.2965794, -5.0995383, 15.3265600, -18.2729683, 18.2650032
23: -2.9445257, 14.9978514, -3.0212145, 15.0289488, -15.7054176, 15.7661591
24: -5.3487329, 13.2386475, -5.4080620, 13.2654972, -14.3719482, 14.4199562
25: -0.9300714, 19.5456924, -1.0215402, 19.5839291, -15.3428841, 15.4521904
26: -12.0489464, 19.5943642, -12.1274033, 19.6874313, -31.7363777, 31.7217674
27: -9.4344273, 10.9032230, -9.4864750, 10.9349747, -19.6741333, 19.7170258
28: -4.1481667, 15.0901537, -4.2275763, 15.1182318, -17.5441742, 17.6102142
29: -3.8643627, 15.8487320, -3.9455414, 15.8766689, -16.1956215, 16.2533836
30: -10.8482552, 10.3567829, -10.9001236, 10.3910027, -17.6961708, 17.7343292
31: -6.7911530, 12.5210457, -6.8923969, 12.5591211, -18.7297401, 18.8051147
32: -26.4732838, -1.8529706, -26.4974289, -1.7881374, -22.6450424, 22.6180725
33: -43.4612083, -7.8678446, -43.5367241, -7.7456837, -28.8634262, 28.7516327
34: -36.1187553, -6.0774059, -36.1759834, -5.9747434, -22.9919510, 22.9330750
35: -26.7142773, 1.1808681, -26.7679634, 1.2686563, -24.9397583, 24.8893433
36: -27.0068607, 4.7823286, -27.0541077, 4.8774371, -31.4014130, 31.3466873
37: -44.0918465, -9.2943058, -44.1806030, -9.1110563, -28.5811844, 28.4334717
38: -31.4459476, 3.0719876, -31.4912052, 3.1580477, -34.6039963, 34.5631943
39: -48.3257294, -10.7281609, -48.3980026, -10.5733509, -33.8451233, 33.6869049
40: -44.4806519, -17.7508602, -44.5264969, -17.6021194, -20.0011749, 19.8373032
41: -30.4207764, -4.1512232, -30.4512329, -4.0563912, -21.3902359, 21.3044014
42: -19.9689274, -0.2563887, -19.9867115, -0.2147198, -15.3945236, 15.3758812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=126, inp2_unstable=126, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=150, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1656
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1449

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_B2_A1_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9351016, upper bound: 9.9146411
time: 27.73 seconds

## Relational analysis of IS_B2_A1_B2_A1_A2

### Relational analysis result of IS_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9351878, upper bound: 9.9146411
time: 24.61 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.3920383, 9.2672710, -22.4578514, 9.2971458, -31.6197510, 31.6607437
1: -12.0231972, 7.7852597, -12.0562487, 7.8080602, -19.8312569, 19.8415089
2: -11.6288624, 9.6210070, -11.6652470, 9.6359501, -18.7202988, 18.7744370
3: -17.5837307, 7.4435625, -17.6329288, 7.4724379, -23.4603882, 23.5038757
4: -19.5783501, 5.1475835, -19.6062202, 5.1807580, -22.3331757, 22.3291702
5: -15.5680847, 9.6796227, -15.6266441, 9.6952372, -24.0830612, 24.1481018
6: -31.9171009, -7.4071994, -31.9364796, -7.3538566, -19.8764038, 19.8433647
7: -21.5930138, 6.0173492, -21.6440868, 6.0375500, -26.2171478, 26.3191757
8: -23.5924892, 7.5756869, -23.6293201, 7.6115665, -29.4572144, 29.4912186
9: -13.7590599, 10.0243959, -13.7817383, 10.0545998, -20.6971283, 20.7289009
10: -13.9569302, 14.1349277, -13.9791813, 14.1535883, -27.5984421, 27.6144867
11: -10.2082996, 11.3613434, -10.2592125, 11.3842812, -17.5344734, 17.5842590
12: -23.2870007, 13.2303619, -23.3039513, 13.3241405, -34.4981384, 34.3933334
13: -25.3059616, 6.0850835, -25.3489876, 6.1515312, -30.9463654, 30.9393387
14: -26.2790108, 14.8943129, -26.3676128, 14.9081583, -39.5386200, 39.6118240
15: -10.0475845, 12.9710712, -10.0767994, 12.9931889, -21.6738434, 21.6244545
16: -20.8967037, 4.4827285, -20.9302120, 4.5069103, -25.1602936, 25.2930222
17: -23.0517273, 11.2450447, -23.1169891, 11.2611942, -34.3129196, 34.3620338
18: -11.1817131, 16.5405865, -11.2255249, 16.5761356, -26.9675140, 26.9518204
19: -7.2389221, 8.3480759, -7.2929401, 8.3602333, -14.6642265, 14.7098885
20: -6.5579319, 10.0264406, -6.6085386, 10.0428219, -15.3790588, 15.4271545
21: -7.5766420, 11.7581892, -7.6300745, 11.7796965, -18.3695869, 18.4105644
22: -5.0559158, 15.3263569, -5.1032658, 15.3430281, -18.3173485, 18.2918205
23: -2.9558573, 15.0047913, -3.0240898, 15.0301847, -15.7167244, 15.7811584
24: -5.3618889, 13.2449598, -5.4099293, 13.2684336, -14.3867378, 14.4271774
25: -0.9460444, 19.5662880, -1.0242491, 19.5943871, -15.3658180, 15.4584656
26: -12.0819330, 19.6340370, -12.1320410, 19.7096443, -31.7915764, 31.7660789
27: -9.4347124, 10.9133282, -9.4838562, 10.9362497, -19.7047882, 19.7074585
28: -4.1651449, 15.0958004, -4.2304440, 15.1194019, -17.5603027, 17.6216202
29: -3.8860097, 15.8663836, -3.9504004, 15.8863335, -16.2321396, 16.2706642
30: -10.8641443, 10.3726549, -10.9009495, 10.3980408, -17.7180328, 17.7421646
31: -6.8065476, 12.5360775, -6.8973236, 12.5606089, -18.7450180, 18.8185844
32: -26.4738026, -1.8425574, -26.4958153, -1.7849064, -22.6637115, 22.6235886
33: -43.4984283, -7.8315587, -43.5379028, -7.7256889, -28.9049377, 28.7542572
34: -36.1482086, -6.0480185, -36.1772346, -5.9575500, -23.0264893, 22.9277611
35: -26.7413845, 1.1973734, -26.7696362, 1.2786088, -24.9687424, 24.8855362
36: -27.0283012, 4.8056078, -27.0562401, 4.8898554, -31.4331207, 31.3700180
37: -44.1361580, -9.2448759, -44.1827354, -9.0824223, -28.6523056, 28.4552765
38: -31.4598389, 3.0816822, -31.4945068, 3.1602831, -34.6201210, 34.5761871
39: -48.3470840, -10.7098579, -48.3991127, -10.5638828, -33.8657227, 33.6920929
40: -44.4906311, -17.7349663, -44.5270386, -17.5940990, -20.0211334, 19.8520012
41: -30.4247780, -4.1452155, -30.4499931, -4.0508199, -21.4044647, 21.3097115
42: -19.9733238, -0.2517047, -19.9865303, -0.2109218, -15.4026451, 15.3821411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=126, inp2_unstable=126, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=150, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1723
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1720
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1337
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1433
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1315
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 970

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_B2_A1_B2_A2_A1

### Relational analysis result of IS_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9351016, upper bound: 9.9187228
time: 22.26 seconds

## Relational analysis of IS_B2_A1_B2_A2_A2

### Relational analysis result of IS_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9351878, upper bound: 9.9187228
time: 17.25 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.4420090, 9.2937078, -22.4763489, 9.2948675, -31.6670380, 31.7147522
1: -12.0495815, 7.8026562, -12.0636301, 7.8055820, -19.8551636, 19.8662872
2: -11.6479502, 9.6322956, -11.6682100, 9.6285982, -18.7332153, 18.7871475
3: -17.6194134, 7.4667253, -17.6470795, 7.4704809, -23.4722595, 23.5304260
4: -19.6190777, 5.1712985, -19.6165829, 5.1627784, -22.3474503, 22.3943329
5: -15.5872307, 9.6893120, -15.6292934, 9.6875944, -24.0979004, 24.1636963
6: -31.9357414, -7.3933496, -31.9416847, -7.3549843, -19.9206314, 19.8361893
7: -21.6235237, 6.0331354, -21.6527367, 6.0375853, -26.2423172, 26.3283691
8: -23.6399708, 7.6042867, -23.6497421, 7.6019988, -29.4799500, 29.5475159
9: -13.7876930, 10.0442619, -13.7926302, 10.0487051, -20.6987991, 20.7404404
10: -13.9677753, 14.1377010, -13.9684591, 14.1339779, -27.5979004, 27.6078491
11: -10.2313986, 11.3928623, -10.2396660, 11.4007235, -17.5764904, 17.5830536
12: -23.2956009, 13.2410469, -23.2965012, 13.3132668, -34.4778748, 34.3976288
13: -25.3624802, 6.1257243, -25.3633842, 6.1330438, -30.9668121, 31.0008774
14: -26.2990723, 14.9123478, -26.3547020, 14.9113092, -39.5589294, 39.6245041
15: -10.0625544, 12.9799147, -10.0765715, 12.9839897, -21.6846237, 21.6604691
16: -20.9139996, 4.4981060, -20.9228115, 4.5036287, -25.2027588, 25.2333755
17: -23.0713158, 11.2562370, -23.0961628, 11.2535839, -34.3248978, 34.3524017
18: -11.2188520, 16.5871429, -11.2149162, 16.5973587, -27.0370712, 26.9613724
19: -7.2597980, 8.3710194, -7.2837048, 8.3709717, -14.6978531, 14.7114887
20: -6.5780420, 10.0433683, -6.5932631, 10.0398254, -15.4035225, 15.4181175
21: -7.6020346, 11.7910500, -7.6095705, 11.7898350, -18.4120598, 18.4089851
22: -5.0778637, 15.3459568, -5.0829406, 15.3466835, -18.3572922, 18.2684708
23: -2.9850521, 15.0510521, -3.0077600, 15.0474243, -15.7761154, 15.7756958
24: -5.3901634, 13.2827930, -5.3999672, 13.2836113, -14.4474716, 14.4196777
25: -0.9686427, 19.5982895, -0.9991932, 19.5945034, -15.4414539, 15.4165840
26: -12.1218729, 19.6694546, -12.1254673, 19.7239876, -31.8458595, 31.7949219
27: -9.4693594, 10.9539595, -9.4737415, 10.9561310, -19.7647209, 19.7107887
28: -4.1950884, 15.1315269, -4.2048688, 15.1255627, -17.6097641, 17.6105881
29: -3.9083934, 15.8945484, -3.9198980, 15.8908634, -16.2936325, 16.2369652
30: -10.8869133, 10.3979816, -10.8678551, 10.3950586, -17.7581596, 17.7232056
31: -6.8351574, 12.5799608, -6.8917723, 12.5813093, -18.7954826, 18.8407516
32: -26.4971104, -1.8233418, -26.4955101, -1.8050718, -22.6878281, 22.6259995
33: -43.5530472, -7.7953973, -43.5394325, -7.7548456, -28.8520432, 28.8124466
34: -36.1930923, -6.0337162, -36.1939125, -5.9678173, -23.0279694, 22.9512062
35: -26.7771454, 1.2194781, -26.7757454, 1.2629519, -24.9595261, 24.9115753
36: -27.0577583, 4.8232188, -27.0564442, 4.8731561, -31.4353790, 31.3900375
37: -44.1916122, -9.2139854, -44.1793480, -9.1135979, -28.6230240, 28.5022659
38: -31.5001202, 3.0982604, -31.5009441, 3.1394548, -34.6395760, 34.5992050
39: -48.4144440, -10.6677694, -48.3917389, -10.6220684, -33.7955322, 33.7756042
40: -44.5424576, -17.7039967, -44.5294838, -17.6334667, -20.0009689, 19.8983154
41: -30.4539223, -4.1266527, -30.4481506, -4.0721087, -21.4039574, 21.3231888
42: -19.9845028, -0.2429891, -19.9858189, -0.2149582, -15.4183922, 15.3785095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=126, inp2_unstable=126, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=150, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1623
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1364
type: A, layer: 1, pos: 1022
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1710
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1581
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 970

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_B2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9254277, upper bound: 9.9351903
time: 22.08 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9255189, upper bound: 9.9351903
time: 21.62 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.3965683, 9.2655964, -22.4607315, 9.2999125, -31.6242523, 31.6603622
1: -12.0218048, 7.7841434, -12.0545521, 7.8094668, -19.8312721, 19.8386955
2: -11.6182194, 9.6041203, -11.6586981, 9.6380234, -18.7140121, 18.7483749
3: -17.5856743, 7.4448805, -17.6339931, 7.4751620, -23.4497223, 23.5023880
4: -19.6075401, 5.1504316, -19.6229897, 5.1829128, -22.3542480, 22.3577805
5: -15.5525494, 9.6658201, -15.6175375, 9.6989489, -24.0919952, 24.1294174
6: -31.9435215, -7.3985820, -31.9534225, -7.3542743, -19.9030380, 19.8501663
7: -21.5788078, 6.0006819, -21.6348610, 6.0386643, -26.2037964, 26.2863083
8: -23.6013641, 7.5706105, -23.6327667, 7.6134915, -29.4557190, 29.4915543
9: -13.7829876, 10.0464344, -13.7958994, 10.0592957, -20.7105484, 20.7422829
10: -13.9492369, 14.1404667, -13.9803391, 14.1581097, -27.6010361, 27.6210632
11: -10.2228909, 11.3813934, -10.2590942, 11.3993769, -17.5705719, 17.5946655
12: -23.2686939, 13.1960478, -23.3088341, 13.3044147, -34.4638519, 34.3796463
13: -25.3752232, 6.1331615, -25.3853397, 6.1638412, -31.0181427, 31.0071259
14: -26.2684612, 14.8936033, -26.3581657, 14.9192038, -39.5300293, 39.5975494
15: -10.0368891, 12.9702063, -10.0752306, 12.9912004, -21.6617203, 21.6489525
16: -20.9001312, 4.4881907, -20.9345646, 4.5086174, -25.1806946, 25.2654190
17: -23.0586395, 11.2467861, -23.1173248, 11.2621317, -34.3207703, 34.3641129
18: -11.2129622, 16.5864429, -11.2302618, 16.5999908, -27.0136337, 26.9766922
19: -7.2534237, 8.3672714, -7.2955284, 8.3737125, -14.6946545, 14.7243366
20: -6.5696144, 10.0386333, -6.6078033, 10.0569324, -15.4099808, 15.4323673
21: -7.5992918, 11.7887726, -7.6336040, 11.8001623, -18.4084282, 18.4312286
22: -5.0599284, 15.3229427, -5.1043153, 15.3420658, -18.3133392, 18.2677002
23: -2.9782581, 15.0517178, -3.0258188, 15.0592728, -15.7723579, 15.7947922
24: -5.3792629, 13.2819424, -5.4123492, 13.2903852, -14.4337883, 14.4336929
25: -0.9563437, 19.5949631, -1.0258269, 19.6125259, -15.4135208, 15.4572372
26: -12.0928917, 19.6322002, -12.1358051, 19.7087593, -31.8016510, 31.7680054
27: -9.4708824, 10.9452858, -9.4915924, 10.9579430, -19.7376556, 19.7431793
28: -4.1830959, 15.1384754, -4.2332993, 15.1452284, -17.6076088, 17.6434517
29: -3.8899174, 15.8882198, -3.9498272, 15.9000626, -16.2518616, 16.2584896
30: -10.8748417, 10.3958931, -10.9036865, 10.4121742, -17.7430534, 17.7625809
31: -6.8239284, 12.5675478, -6.8977032, 12.5843191, -18.7899628, 18.8431396
32: -26.5057220, -1.8324499, -26.5147400, -1.7844009, -22.6926117, 22.6214600
33: -43.5388794, -7.8297472, -43.5795288, -7.7418046, -28.8735580, 28.8252182
34: -36.1660271, -6.0615530, -36.2012482, -5.9732442, -23.0082016, 22.9672241
35: -26.7582836, 1.2038379, -26.7915134, 1.2707796, -24.9576111, 24.9297333
36: -27.0455284, 4.8008184, -27.0737190, 4.8793731, -31.4313812, 31.3815994
37: -44.1761932, -9.2621670, -44.2274246, -9.1082735, -28.6160660, 28.4986115
38: -31.4930744, 3.0907855, -31.5121994, 3.1598310, -34.6529045, 34.6029854
39: -48.4331551, -10.6835518, -48.4583855, -10.5708523, -33.8758240, 33.7896423
40: -44.5574226, -17.7188549, -44.5703621, -17.5998974, -20.0472183, 19.8633041
41: -30.4631329, -4.1308432, -30.4742641, -4.0532937, -21.4242554, 21.3166542
42: -19.9831238, -0.2461209, -19.9936047, -0.2112889, -15.4218292, 15.3819141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=126, inp2_unstable=126, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=150, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1656
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 986
type: A, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1434
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 924
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 779
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1302
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1434
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1417
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1395
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1433
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1310
type: A, layer: 1, pos: 1315
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 970
type: B, layer: 1, pos: 1449

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_B2_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9351053, upper bound: 9.9311094
time: 21.44 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9351903, upper bound: 9.9311094
time: 27.06 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.4449100, 9.2964973, -22.4875183, 9.3006554, -31.6748962, 31.7283554
1: -12.0505810, 7.8050518, -12.0710173, 7.8107467, -19.8613281, 19.8760681
2: -11.6531792, 9.6340237, -11.6787004, 9.6387043, -18.7365799, 18.8026047
3: -17.6212883, 7.4698172, -17.6545887, 7.4764652, -23.4788818, 23.5471725
4: -19.6305428, 5.1742220, -19.6358566, 5.1841402, -22.3664932, 22.3959885
5: -15.5880070, 9.6962566, -15.6379423, 9.7001247, -24.1090317, 24.1803589
6: -31.9375114, -7.3923845, -31.9471054, -7.3494473, -19.9274597, 19.8427544
7: -21.6251717, 6.0341482, -21.6616783, 6.0409112, -26.2385025, 26.3559265
8: -23.6451473, 7.6064720, -23.6586781, 7.6147728, -29.4877625, 29.5548630
9: -13.7896147, 10.0492296, -13.7992554, 10.0584126, -20.7107544, 20.7657089
10: -13.9697151, 14.1497231, -13.9836836, 14.1598492, -27.6179123, 27.6340103
11: -10.2352734, 11.3936834, -10.2630520, 11.4023943, -17.5817738, 17.6088333
12: -23.3013420, 13.2433300, -23.3101177, 13.3308773, -34.5226288, 34.4129028
13: -25.3760414, 6.1279149, -25.3881683, 6.1565437, -31.0037842, 31.0237045
14: -26.3030624, 14.9168520, -26.3757172, 14.9199791, -39.5659790, 39.6439667
15: -10.0631094, 12.9865313, -10.0835085, 12.9970121, -21.6930161, 21.6669502
16: -20.9186916, 4.5008144, -20.9414902, 4.5103521, -25.2061615, 25.2904625
17: -23.0777550, 11.2570534, -23.1262245, 11.2640553, -34.3418121, 34.3832779
18: -11.2209167, 16.5903969, -11.2312202, 16.6045151, -27.0429840, 26.9792404
19: -7.2647114, 8.3726692, -7.2980003, 8.3740368, -14.7064171, 14.7330894
20: -6.5822949, 10.0533962, -6.6133170, 10.0579958, -15.4187355, 15.4474564
21: -7.6072540, 11.7978277, -7.6355839, 11.8015347, -18.4211235, 18.4416542
22: -5.0811577, 15.3527489, -5.1080279, 15.3585300, -18.3577499, 18.2945061
23: -2.9895773, 15.0586653, -3.0287066, 15.0605049, -15.7836800, 15.8098183
24: -5.3924198, 13.2882366, -5.4142065, 13.2933121, -14.4485817, 14.4409180
25: -0.9722805, 19.6155586, -1.0285416, 19.6229782, -15.4364319, 15.4635086
26: -12.1259317, 19.6718826, -12.1404867, 19.7309608, -31.8568916, 31.8123703
27: -9.4711685, 10.9554005, -9.4889832, 10.9592056, -19.7683487, 19.7336235
28: -4.2000828, 15.1441231, -4.2361803, 15.1463957, -17.6237259, 17.6548195
29: -3.9115663, 15.9058495, -3.9546809, 15.9097252, -16.2883911, 16.2757664
30: -10.8907576, 10.4117565, -10.9045296, 10.4192009, -17.7649193, 17.7704277
31: -6.8393364, 12.5826120, -6.9026165, 12.5858068, -18.8052483, 18.8566132
32: -26.5062370, -1.8220558, -26.5131664, -1.7811227, -22.7112503, 22.6269760
33: -43.5761032, -7.7934518, -43.5807190, -7.7218351, -28.9151230, 28.8278351
34: -36.1954918, -6.0321593, -36.2024918, -5.9560547, -23.0427246, 22.9619370
35: -26.7854061, 1.2203526, -26.7932205, 1.2806845, -24.9865646, 24.9258804
36: -27.0670013, 4.8240957, -27.0758362, 4.8917909, -31.4631042, 31.4049683
37: -44.2205772, -9.2127094, -44.2294579, -9.0796051, -28.6871948, 28.5203934
38: -31.5070076, 3.1005445, -31.5154228, 3.1620955, -34.6691017, 34.6159668
39: -48.4545441, -10.6652117, -48.4594727, -10.5614090, -33.8963776, 33.7947998
40: -44.5673904, -17.7029495, -44.5709229, -17.5918541, -20.0671692, 19.8780746
41: -30.4670830, -4.1248527, -30.4730148, -4.0477223, -21.4384804, 21.3220024
42: -19.9875393, -0.2414622, -19.9934120, -0.2074800, -15.4299183, 15.3881912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=126, inp2_unstable=126, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=150, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: B, layer: 1, pos: 663
type: A, layer: 1, pos: 1694
type: B, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1623
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1783
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 647
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1609
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 1693
type: B, layer: 1, pos: 1693
type: A, layer: 1, pos: 1621
type: B, layer: 1, pos: 1621
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1579
type: B, layer: 1, pos: 1579
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 1365
type: B, layer: 1, pos: 1365
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 988
type: A, layer: 1, pos: 988
type: B, layer: 1, pos: 1781
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1720
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 985
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1337
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1582
type: A, layer: 1, pos: 1461
type: B, layer: 1, pos: 1657
type: A, layer: 1, pos: 1582
type: B, layer: 1, pos: 1461
type: A, layer: 1, pos: 1657
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1782
type: B, layer: 1, pos: 1782
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1659
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 1659
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1343
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1640
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1709
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1640
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1347
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1364
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 924
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 779
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1547
type: B, layer: 1, pos: 1022
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 1547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1302
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 939
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1417
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1318
type: A, layer: 1, pos: 1318
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1321
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1321
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1455
type: A, layer: 1, pos: 1455
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 970
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 1395
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1449
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1297
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1710
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1581
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1310
type: B, layer: 1, pos: 1449
type: A, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_B2_A2_B2_A2_A1

### Relational analysis result of IS_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9351053, upper bound: 9.9351903
time: 26.86 seconds

## Relational analysis of IS_B2_A2_B2_A2_A2

### Relational analysis result of IS_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9351903, upper bound: 9.9351903
time: 22.67 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 51.78 seconds
IS_B1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9080613, upper bound: 9.9351053
IS_B1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9080613, upper bound: 9.9351904
IS_B1_A2_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9121429, upper bound: 9.9351053
IS_B1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9121429, upper bound: 9.9351904
IS_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9176495, upper bound: 9.9351904
IS_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9177344, upper bound: 9.9351904
IS_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9217288, upper bound: 9.9351904
IS_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9218141, upper bound: 9.9351904
IS_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9351016, upper bound: 9.9146411
IS_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9351878, upper bound: 9.9146411
IS_B2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9351016, upper bound: 9.9187228
IS_B2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9351878, upper bound: 9.9187228
IS_B2_A2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9254277, upper bound: 9.9351903
IS_B2_A2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9255189, upper bound: 9.9351903
IS_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9351053, upper bound: 9.9311094
IS_B2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9351903, upper bound: 9.9311094
IS_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9351053, upper bound: 9.9351903
IS_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 51.78
Output dim: 25, lower bound: -9.9351903, upper bound: 9.9351903

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 42.70 + 959.81 = 1002.51 seconds
