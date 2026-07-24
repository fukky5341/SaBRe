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
execution time: IAR + RelationalAnalysis = 2.36 + 40.56 = 42.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 25, lower bound: -9.9470526, upper bound: 9.9470526

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1299

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1723

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9445473, upper bound: 9.9311804
time: 23.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9445473, upper bound: 9.9445471
time: 23.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 47.74 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 47.74
Output dim: 25, lower bound: -9.9445473, upper bound: 9.9311804
IS_A2, status: Status.UNKNOWN, split count: 1, time: 47.74
Output dim: 25, lower bound: -9.9445473, upper bound: 9.9445471

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -22.4459877, 9.2773228, -22.4490681, 9.2882423, -31.6683655, 31.6597290
1: -12.0557423, 7.7999167, -12.0567589, 7.8040566, -19.8597984, 19.8566761
2: -11.6584082, 9.6144276, -11.6598425, 9.6255322, -18.7629852, 18.7520065
3: -17.6239262, 7.4551916, -17.6251621, 7.4640112, -23.5032654, 23.4943542
4: -19.6286049, 5.1689606, -19.6325092, 5.1736031, -22.3999405, 22.3969498
5: -15.5915108, 9.6680384, -15.5928745, 9.6836052, -24.1204071, 24.1063080
6: -31.9168739, -7.3928671, -31.9353371, -7.3911686, -19.8356438, 19.8490868
7: -21.6379986, 6.0195446, -21.6403332, 6.0287933, -26.2877960, 26.2755356
8: -23.6524544, 7.6004629, -23.6535110, 7.6057253, -29.5342560, 29.5284653
9: -13.7857733, 10.0475721, -13.7891750, 10.0517035, -20.7335892, 20.7294312
10: -13.9660912, 14.1379719, -13.9699163, 14.1465149, -27.6178207, 27.6104050
11: -10.2402315, 11.3944969, -10.2454338, 11.3955746, -17.5903778, 17.5915871
12: -23.2565651, 13.2462778, -23.2809868, 13.2487593, -34.3899841, 34.4121628
13: -25.3591595, 6.1396146, -25.3704510, 6.1445079, -30.9991760, 31.0051804
14: -26.2980957, 14.8869057, -26.3039780, 14.9041672, -39.5540771, 39.5426788
15: -10.0639458, 12.9968920, -10.0666399, 13.0032425, -21.6782303, 21.6789932
16: -20.9270153, 4.4968324, -20.9315319, 4.5009637, -25.2164154, 25.2087479
17: -23.0716820, 11.2547541, -23.0779476, 11.2594252, -34.3311081, 34.3327026
18: -11.2153893, 16.5922966, -11.2214680, 16.5939236, -27.0007782, 27.0082855
19: -7.2561531, 8.3590736, -7.2621737, 8.3666000, -14.6952515, 14.6936817
20: -6.5765753, 10.0370216, -6.5817451, 10.0465050, -15.4191399, 15.4144516
21: -7.6000452, 11.7847843, -7.6061449, 11.7924500, -18.4174309, 18.4159431
22: -5.0752640, 15.3628139, -5.0807152, 15.3690376, -18.3388100, 18.3464203
23: -2.9823551, 15.0398998, -2.9878278, 15.0506439, -15.7865143, 15.7806435
24: -5.3881788, 13.2752399, -5.3912821, 13.2838936, -14.4489136, 14.4434547
25: -0.9691415, 19.5921898, -0.9730802, 19.6097031, -15.4653912, 15.4526863
26: -12.0949173, 19.6789742, -12.1126995, 19.6812496, -31.7761669, 31.7916737
27: -9.4714470, 10.9433413, -9.4770489, 10.9509954, -19.7409210, 19.7415161
28: -4.1883402, 15.1224680, -4.1959400, 15.1349583, -17.6201782, 17.6152191
29: -3.9046164, 15.9038982, -3.9102488, 15.9114113, -16.2985802, 16.3002415
30: -10.8854847, 10.4050856, -10.8903179, 10.4107056, -17.7711029, 17.7688141
31: -6.8343925, 12.5545740, -6.8400612, 12.5698929, -18.7922173, 18.7829590
32: -26.4748745, -1.8240519, -26.4942856, -1.8219080, -22.6253662, 22.6413727
33: -43.5400963, -7.7956514, -43.5616989, -7.7919760, -28.8569794, 28.8762512
34: -36.1534882, -6.0324883, -36.1764297, -6.0305276, -22.9566956, 22.9775429
35: -26.7543240, 1.2250686, -26.7723827, 1.2264175, -24.9248123, 24.9421921
36: -27.0291615, 4.8298149, -27.0510540, 4.8310857, -31.3752594, 31.3959427
37: -44.1510811, -9.2125607, -44.1890869, -9.2107477, -28.5287552, 28.5652542
38: -31.4763279, 3.0980997, -31.4955750, 3.1002226, -34.5765495, 34.5936737
39: -48.4037094, -10.6651001, -48.4318237, -10.6610107, -33.8212891, 33.8458557
40: -44.5096321, -17.7027779, -44.5412903, -17.7009277, -19.9067307, 19.9356308
41: -30.4173946, -4.1260853, -30.4459114, -4.1238899, -21.3101997, 21.3360481
42: -19.9624043, -0.2450414, -19.9771080, -0.2417526, -15.3672791, 15.3781948

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=128, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1299

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1725

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9276224, upper bound: 9.9307216
time: 24.04 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9440915, upper bound: 9.9307246
time: 19.44 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.4917431, 9.3028898, -22.4513435, 9.2991657, -31.7271729, 31.6862717
1: -12.0752878, 7.8126216, -12.0572376, 7.8077135, -19.8830013, 19.8698597
2: -11.6847019, 9.6406384, -11.6608400, 9.6367445, -18.8073730, 18.7790146
3: -17.6577454, 7.4796214, -17.6260719, 7.4729891, -23.5526276, 23.5161667
4: -19.6392765, 5.1867046, -19.6353035, 5.1777301, -22.4141083, 22.4156952
5: -15.6414185, 9.7036867, -15.5937424, 9.6992493, -24.1856537, 24.1407852
6: -31.9569798, -7.3476639, -31.9534702, -7.3897119, -19.8661346, 19.8998032
7: -21.6738014, 6.0426598, -21.6415653, 6.0368676, -26.3602142, 26.2981033
8: -23.6664715, 7.6169176, -23.6538811, 7.6095858, -29.5653076, 29.5445404
9: -13.8009605, 10.0632811, -13.7918873, 10.0554667, -20.7586975, 20.7416725
10: -13.9861698, 14.1627121, -13.9725752, 14.1535034, -27.6511459, 27.6402740
11: -10.2714024, 11.4040976, -10.2479973, 11.3964701, -17.6342106, 17.6000214
12: -23.3148842, 13.3354063, -23.3061848, 13.2504406, -34.4428864, 34.5258942
13: -25.3937664, 6.1684523, -25.3825722, 6.1475658, -31.0356903, 31.0439301
14: -26.3800850, 14.9234905, -26.3094864, 14.9216061, -39.6536255, 39.5832062
15: -10.0869932, 13.0098314, -10.0689564, 13.0044689, -21.6871567, 21.7180481
16: -20.9514503, 4.5130258, -20.9332199, 4.5048981, -25.2966080, 25.2108383
17: -23.1309242, 11.2661734, -23.0838261, 11.2603188, -34.3912430, 34.3499985
18: -11.2341022, 16.6084156, -11.2252111, 16.5953712, -27.0216751, 27.0447540
19: -7.3004827, 8.3759556, -7.2677479, 8.3751211, -14.7468357, 14.7140846
20: -6.6158805, 10.0608616, -6.5859289, 10.0567179, -15.4706116, 15.4407654
21: -7.6387854, 11.8042831, -7.6115341, 11.8009453, -18.4646339, 18.4400024
22: -5.1110716, 15.3695869, -5.0858002, 15.3686686, -18.3458481, 18.3855095
23: -3.0309706, 15.0639200, -2.9925203, 15.0625896, -15.8465805, 15.8050232
24: -5.4153743, 13.2966480, -5.3939028, 13.2923317, -14.4836082, 14.4639435
25: -1.0314975, 19.6314621, -0.9765520, 19.6277199, -15.5452271, 15.4842834
26: -12.1434116, 19.7397079, -12.1301384, 19.6833496, -31.8267612, 31.8698463
27: -9.4953604, 10.9612751, -9.4813232, 10.9585543, -19.7669601, 19.7621078
28: -4.2382154, 15.1492300, -4.2028160, 15.1476479, -17.6852379, 17.6454468
29: -3.9574366, 15.9170456, -3.9156656, 15.9166536, -16.3416328, 16.3179550
30: -10.9070063, 10.4226818, -10.8941612, 10.4160347, -17.8004074, 17.7916183
31: -6.9070482, 12.5886621, -6.8451881, 12.5863504, -18.8799744, 18.8166847
32: -26.5188656, -1.7798195, -26.5143967, -1.8201613, -22.6590118, 22.7000198
33: -43.5869331, -7.7190008, -43.5834389, -7.7894402, -28.8952484, 28.9859619
34: -36.2061615, -5.9537959, -36.2015076, -6.0288877, -22.9953537, 23.0796280
35: -26.7990074, 1.2849684, -26.7921619, 1.2267537, -24.9603882, 25.0231171
36: -27.0805645, 4.8968482, -27.0734959, 4.8319941, -31.4242401, 31.4846191
37: -44.2361755, -9.0773287, -44.2283478, -9.2092476, -28.5947189, 28.7414856
38: -31.5211697, 3.1633391, -31.5147076, 3.1021209, -34.6232910, 34.6780472
39: -48.4669647, -10.5571270, -48.4626541, -10.6583128, -33.8715363, 33.9857712
40: -44.5770187, -17.5896397, -44.5747452, -17.6993446, -19.9521790, 20.0845299
41: -30.4787903, -4.0458221, -30.4751358, -4.1219616, -21.3585510, 21.4477119
42: -19.9970131, -0.2057533, -19.9932728, -0.2387424, -15.3980103, 15.4307632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=128, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=150, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1299

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1725

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9276224, upper bound: 9.9440884
time: 22.67 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9440915, upper bound: 9.9440914
time: 22.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 47.21 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 47.21
Output dim: 25, lower bound: -9.9276224, upper bound: 9.9307216
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 47.21
Output dim: 25, lower bound: -9.9440915, upper bound: 9.9307246
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 47.21
Output dim: 25, lower bound: -9.9276224, upper bound: 9.9440884
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 47.21
Output dim: 25, lower bound: -9.9440915, upper bound: 9.9440914

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -22.4456844, 9.2771587, -22.4484406, 9.2879429, -31.6723633, 31.6579361
1: -12.0556145, 7.7997370, -12.0565033, 7.8036785, -19.8592930, 19.8562393
2: -11.6577015, 9.6142273, -11.6584482, 9.6251965, -18.7620926, 18.7388268
3: -17.6237354, 7.4549203, -17.6248245, 7.4635229, -23.5023346, 23.4733810
4: -19.6272545, 5.1687307, -19.6300468, 5.1731911, -22.3992004, 22.3676605
5: -15.5907288, 9.6678295, -15.5913525, 9.6832256, -24.1205215, 24.1043243
6: -31.9159927, -7.3930020, -31.9336491, -7.3913794, -19.8255844, 19.8708572
7: -21.6367378, 6.0193882, -21.6377983, 6.0284190, -26.2859802, 26.2577515
8: -23.6516571, 7.6002679, -23.6525135, 7.6053314, -29.5335999, 29.5006714
9: -13.7854919, 10.0473404, -13.7886562, 10.0513248, -20.7320709, 20.7095375
10: -13.9657841, 14.1376228, -13.9694500, 14.1458273, -27.6159058, 27.6080093
11: -10.2399855, 11.3939533, -10.2449274, 11.3944969, -17.5755596, 17.5908012
12: -23.2562294, 13.2460842, -23.2803154, 13.2483435, -34.3872833, 34.4103012
13: -25.3578854, 6.1394229, -25.3679199, 6.1441245, -30.9974365, 30.9824982
14: -26.2975655, 14.8859491, -26.3029251, 14.9022284, -39.5521240, 39.5390930
15: -10.0631809, 12.9966030, -10.0651512, 13.0027609, -21.6869812, 21.6747017
16: -20.9263115, 4.4965959, -20.9303665, 4.5005274, -25.2058868, 25.2286148
17: -23.0709114, 11.2546406, -23.0764351, 11.2591457, -34.3300552, 34.3310776
18: -11.2151089, 16.5914688, -11.2209492, 16.5922432, -26.9680328, 27.0076065
19: -7.2557755, 8.3584013, -7.2614489, 8.3652534, -14.6840363, 14.6924629
20: -6.5762372, 10.0362492, -6.5811281, 10.0449600, -15.4049606, 15.4126129
21: -7.5996122, 11.7840967, -7.6053143, 11.7911224, -18.4024887, 18.4141159
22: -5.0749750, 15.3625097, -5.0801744, 15.3684750, -18.3152161, 18.3441086
23: -2.9820843, 15.0389805, -2.9872847, 15.0488157, -15.7539520, 15.7791977
24: -5.3879881, 13.2744589, -5.3909473, 13.2823334, -14.4118729, 14.4429703
25: -0.9688621, 19.5913467, -0.9725323, 19.6082058, -15.4098244, 15.4518852
26: -12.0945072, 19.6782742, -12.1119671, 19.6801529, -31.7746601, 31.7902412
27: -9.4712029, 10.9426622, -9.4765911, 10.9496021, -19.7183838, 19.7403526
28: -4.1879978, 15.1216259, -4.1952782, 15.1332579, -17.5972710, 17.6137428
29: -3.9043770, 15.9037476, -3.9097528, 15.9111309, -16.2553978, 16.2996025
30: -10.8851795, 10.4044323, -10.8898048, 10.4094505, -17.7545204, 17.7673264
31: -6.8339882, 12.5538073, -6.8392348, 12.5683842, -18.7764664, 18.7813187
32: -26.4739819, -1.8242273, -26.4926434, -1.8221879, -22.6160736, 22.6581497
33: -43.5388031, -7.7958994, -43.5590973, -7.7923584, -28.8536072, 28.8168106
34: -36.1518631, -6.0326910, -36.1732101, -6.0309343, -22.9531937, 22.9500008
35: -26.7529373, 1.2249370, -26.7698040, 1.2262211, -24.9221115, 24.9158478
36: -27.0275879, 4.8296561, -27.0480099, 4.8308568, -31.3732910, 31.3840027
37: -44.1495438, -9.2126999, -44.1859894, -9.2109404, -28.5256729, 28.5186996
38: -31.4748249, 3.0979195, -31.4925938, 3.0999217, -34.5747452, 34.5905151
39: -48.4018974, -10.6652946, -48.4281502, -10.6614094, -33.8186035, 33.7686920
40: -44.5082092, -17.7028732, -44.5385132, -17.7011185, -19.8961678, 19.9269981
41: -30.4166222, -4.1262889, -30.4444275, -4.1242480, -21.3052673, 21.3332901
42: -19.9612751, -0.2452931, -19.9749584, -0.2422438, -15.3635273, 15.3841724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1299

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9384862, upper bound: 9.9251923
time: 20.33 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9385639, upper bound: 9.9251923
time: 21.60 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.4617462, 9.2991886, -22.3977699, 9.2696686, -31.6636963, 31.6293411
1: -12.0604134, 7.8097687, -12.0295944, 7.7875209, -19.8479347, 19.8393631
2: -11.6705894, 9.6376772, -11.6351261, 9.6234150, -18.7783051, 18.7495308
3: -17.6359005, 7.4753437, -17.5881939, 7.4461756, -23.5084457, 23.4766006
4: -19.6082649, 5.1830754, -19.5806999, 5.1507387, -22.3465195, 22.3530502
5: -15.6294184, 9.6985970, -15.5722771, 9.6822243, -24.1534805, 24.1127472
6: -31.9454880, -7.3522272, -31.9313202, -7.4047661, -19.8566818, 19.8704834
7: -21.6549339, 6.0391445, -21.6068707, 6.0197039, -26.3216248, 26.2589188
8: -23.6363735, 7.6134820, -23.6002045, 7.5783997, -29.5010834, 29.4861221
9: -13.7831612, 10.0592260, -13.7608089, 10.0301609, -20.7203751, 20.7081146
10: -13.9813862, 14.1561089, -13.9593391, 14.1380482, -27.6296463, 27.6183167
11: -10.2672892, 11.3854227, -10.2205486, 11.3629971, -17.5948086, 17.5519485
12: -23.3083801, 13.3284588, -23.2912140, 13.2370405, -34.4206848, 34.4995422
13: -25.3532562, 6.1632023, -25.3099442, 6.1043243, -30.9496155, 30.9638748
14: -26.3714294, 14.9106007, -26.2844086, 14.8970861, -39.6192474, 39.5522919
15: -10.0795097, 13.0057716, -10.0519133, 12.9885864, -21.6533585, 21.6946373
16: -20.9394760, 4.5093307, -20.9100952, 4.4863787, -25.2885971, 25.1847000
17: -23.1209030, 11.2631512, -23.0562820, 11.2479982, -34.3689003, 34.3194351
18: -11.2280741, 16.5792122, -11.1854362, 16.5438519, -26.9615326, 26.9686737
19: -7.2950516, 8.3614578, -7.2412262, 8.3491573, -14.7124119, 14.6706696
20: -6.6107717, 10.0449095, -6.5609374, 10.0282316, -15.4361382, 15.3992348
21: -7.6328526, 11.7817593, -7.5800533, 11.7599945, -18.4185944, 18.3866234
22: -5.1060486, 15.3537617, -5.0600142, 15.3417521, -18.3195190, 18.3427887
23: -3.0260448, 15.0326910, -2.9582448, 15.0068817, -15.7854004, 15.7366066
24: -5.4108815, 13.2709904, -5.3630347, 13.2475214, -14.4328651, 14.4016228
25: -1.0269194, 19.6020107, -0.9497309, 19.5769539, -15.4845772, 15.4128494
26: -12.1345510, 19.7177372, -12.0854263, 19.6444168, -31.7789688, 31.8031635
27: -9.4899960, 10.9375992, -9.4443951, 10.9151020, -19.7182617, 19.6973686
28: -4.2321267, 15.1213865, -4.1672020, 15.0976467, -17.6291008, 17.5805397
29: -3.9528990, 15.8935089, -3.8896093, 15.8768978, -16.2933502, 16.2610455
30: -10.9031372, 10.4008808, -10.8669577, 10.3756809, -17.7555466, 17.7432404
31: -6.9013300, 12.5626717, -6.8115511, 12.5383329, -18.8261566, 18.7548409
32: -26.5005760, -1.7837095, -26.4802990, -1.8409925, -22.6462555, 22.6691666
33: -43.5427475, -7.7230873, -43.5031586, -7.8279505, -28.8182602, 28.9163513
34: -36.1792526, -5.9555092, -36.1509552, -6.0451279, -22.9576721, 23.0358124
35: -26.7740402, 1.2826972, -26.7456322, 1.2035303, -24.9173660, 24.9787827
36: -27.0594101, 4.8946996, -27.0317822, 4.8132696, -31.3871765, 31.4427185
37: -44.1878014, -9.0802660, -44.1408119, -9.2415895, -28.5264893, 28.6600342
38: -31.4986649, 3.1613693, -31.4645710, 3.0829697, -34.5816345, 34.6259384
39: -48.4047661, -10.5598097, -48.3515930, -10.7033682, -33.7661133, 33.8779602
40: -44.5317116, -17.5919704, -44.4952049, -17.7315903, -19.9155502, 20.0295448
41: -30.4550056, -4.0490880, -30.4313126, -4.1427407, -21.3412857, 21.4108810
42: -19.9890347, -0.2094369, -19.9769077, -0.2495193, -15.3881912, 15.4093914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=150, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1299

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9220195, upper bound: 9.9385617
time: 33.75 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9220996, upper bound: 9.9385617
time: 21.26 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.4914513, 9.3027401, -22.4506397, 9.2988625, -31.7312012, 31.6845779
1: -12.0751534, 7.8123999, -12.0570211, 7.8073030, -19.8824558, 19.8694210
2: -11.6840124, 9.6404562, -11.6594496, 9.6364002, -18.8064728, 18.7658157
3: -17.6575356, 7.4793491, -17.6257324, 7.4724360, -23.5517273, 23.4951172
4: -19.6379395, 5.1864738, -19.6328602, 5.1773167, -22.4133530, 22.3864441
5: -15.6406555, 9.7034473, -15.5922289, 9.6988716, -24.1857758, 24.1387558
6: -31.9561138, -7.3478251, -31.9517326, -7.3899465, -19.8560677, 19.9215317
7: -21.6725349, 6.0424833, -21.6390762, 6.0365171, -26.3584518, 26.2803268
8: -23.6657028, 7.6167021, -23.6528683, 7.6092095, -29.5646362, 29.5166855
9: -13.8006973, 10.0630665, -13.7913713, 10.0550575, -20.7571793, 20.7217636
10: -13.9858885, 14.1623516, -13.9721146, 14.1528263, -27.6492233, 27.6377869
11: -10.2711391, 11.4035654, -10.2475252, 11.3953753, -17.6193848, 17.5992470
12: -23.3144989, 13.3351908, -23.3055305, 13.2499876, -34.4401703, 34.5241089
13: -25.3924465, 6.1682887, -25.3800125, 6.1471868, -31.0339355, 31.0212555
14: -26.3795090, 14.9224510, -26.3083916, 14.9196568, -39.6514893, 39.5796890
15: -10.0862379, 13.0095911, -10.0674191, 13.0040073, -21.6958694, 21.7137642
16: -20.9507866, 4.5127916, -20.9320946, 4.5044575, -25.2860641, 25.2306213
17: -23.1301346, 11.2660437, -23.0822983, 11.2600565, -34.3901901, 34.3483429
18: -11.2337894, 16.6075974, -11.2246695, 16.5936699, -26.9889603, 27.0440598
19: -7.3001022, 8.3752699, -7.2670202, 8.3737583, -14.7356091, 14.7128658
20: -6.6155486, 10.0600901, -6.5852766, 10.0551844, -15.4564133, 15.4389076
21: -7.6383495, 11.8036098, -7.6106853, 11.7996159, -18.4496918, 18.4381714
22: -5.1107693, 15.3692446, -5.0852494, 15.3681469, -18.3221970, 18.3831825
23: -3.0306635, 15.0630283, -2.9919724, 15.0607624, -15.8140182, 15.8035583
24: -5.4151821, 13.2958641, -5.3935480, 13.2907763, -14.4465904, 14.4634666
25: -1.0311980, 19.6306019, -0.9760060, 19.6262131, -15.4896393, 15.4834671
26: -12.1429577, 19.7390327, -12.1293793, 19.6822414, -31.8251991, 31.8684120
27: -9.4951077, 10.9605589, -9.4808464, 10.9571753, -19.7444191, 19.7609406
28: -4.2378535, 15.1483879, -4.2021351, 15.1459694, -17.6623154, 17.6439743
29: -3.9571638, 15.9168968, -3.9151688, 15.9163876, -16.2984467, 16.3173180
30: -10.9066782, 10.4220524, -10.8935900, 10.4147730, -17.7837982, 17.7901154
31: -6.9066000, 12.5879011, -6.8443518, 12.5848398, -18.8641891, 18.8150444
32: -26.5179367, -1.7799778, -26.5127659, -1.8204441, -22.6496658, 22.7167130
33: -43.5855980, -7.7192254, -43.5808716, -7.7898355, -28.8918304, 28.9264908
34: -36.2044983, -5.9540019, -36.1982803, -6.0293069, -22.9918289, 23.0520668
35: -26.7975864, 1.2848258, -26.7895775, 1.2265291, -24.9577103, 24.9966812
36: -27.0790577, 4.8966722, -27.0704632, 4.8317513, -31.4221802, 31.4727554
37: -44.2346115, -9.0774584, -44.2251816, -9.2094679, -28.5916214, 28.6948700
38: -31.5196552, 3.1631885, -31.5117092, 3.1017618, -34.6214180, 34.6748962
39: -48.4651260, -10.5573292, -48.4590263, -10.6587391, -33.8687592, 33.9086075
40: -44.5756073, -17.5897255, -44.5720291, -17.6995373, -19.9416084, 20.0755577
41: -30.4780273, -4.0459800, -30.4736481, -4.1223521, -21.3535919, 21.4448929
42: -19.9959259, -0.2060175, -19.9911156, -0.2392702, -15.3942604, 15.4366951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=127, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=150, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1644
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1693
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1337
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1427
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1709
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 1318
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1455
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1299

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 663

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9384862, upper bound: 9.9385639
time: 23.01 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 25, lower bound: -9.9385639, upper bound: 9.9385639
time: 21.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 46.38 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 46.38
Output dim: 25, lower bound: -9.9384862, upper bound: 9.9251923
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 46.38
Output dim: 25, lower bound: -9.9385639, upper bound: 9.9251923
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 46.38
Output dim: 25, lower bound: -9.9220195, upper bound: 9.9385617
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 46.38
Output dim: 25, lower bound: -9.9220996, upper bound: 9.9385617
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 46.38
Output dim: 25, lower bound: -9.9384862, upper bound: 9.9385639
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 46.38
Output dim: 25, lower bound: -9.9385639, upper bound: 9.9385639

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -22.4254608, 9.2235870, -22.4361954, 9.2553959, -31.6185760, 31.5928497
1: -12.0385342, 7.7663221, -12.0462360, 7.7834892, -19.8220234, 19.8125572
2: -11.6457748, 9.5407133, -11.6512690, 9.5809116, -18.7056656, 18.6587563
3: -17.6076164, 7.3647356, -17.6151562, 7.4082532, -23.4251938, 23.3624496
4: -19.6160450, 5.1463270, -19.6232586, 5.1596723, -22.3628693, 22.3244629
5: -15.5765457, 9.5549288, -15.5828438, 9.6131353, -24.0362396, 23.9817810
6: -31.9031734, -7.4542208, -31.9258995, -7.4283438, -19.7717476, 19.7971840
7: -21.6146317, 5.9228330, -21.6244984, 5.9684477, -26.1985931, 26.1379395
8: -23.6399422, 7.5545192, -23.6454773, 7.5777645, -29.4950104, 29.4496384
9: -13.6915350, 10.0363245, -13.7320786, 10.0446615, -20.6346283, 20.6414566
10: -13.8283520, 14.1184406, -13.8868647, 14.1342468, -27.4582825, 27.4989243
11: -10.1933098, 11.3715115, -10.2168941, 11.3800983, -17.5021877, 17.5215378
12: -23.0879040, 13.2282047, -23.1791706, 13.2374868, -34.2044678, 34.2870941
13: -25.3180313, 6.1259117, -25.3434811, 6.1359239, -30.9480286, 30.9413910
14: -26.1178017, 14.8804150, -26.1946869, 14.8988400, -39.3656311, 39.4213638
15: -10.0062160, 12.9745741, -10.0306053, 12.9895649, -21.6043396, 21.6100998
16: -20.8757172, 4.4585342, -20.8999405, 4.4768829, -25.1083603, 25.1230965
17: -22.9477196, 11.2406063, -23.0011234, 11.2506685, -34.1983871, 34.2417297
18: -11.1585579, 16.5751171, -11.1868668, 16.5824070, -26.8672028, 26.9326324
19: -7.2184649, 8.3378601, -7.2389698, 8.3527145, -14.6255112, 14.6418324
20: -6.5466290, 10.0303717, -6.5632305, 10.0414162, -15.3668594, 15.3852253
21: -7.5542936, 11.7783623, -7.5779800, 11.7876854, -18.3500443, 18.3761215
22: -4.9850264, 15.3512383, -5.0259199, 15.3617496, -18.2103157, 18.2737465
23: -2.9518542, 15.0221462, -2.9691000, 15.0386314, -15.7123871, 15.7445183
24: -5.3717556, 13.2651901, -5.3811264, 13.2767582, -14.3806419, 14.4158897
25: -0.9308858, 19.5863533, -0.9491706, 19.6052036, -15.3631821, 15.4204941
26: -11.9304562, 19.6620369, -12.0126019, 19.6703606, -31.6008167, 31.6746387
27: -9.4544945, 10.9208870, -9.4665060, 10.9359732, -19.6830978, 19.7074356
28: -4.1609583, 15.1157522, -4.1789269, 15.1297283, -17.5601807, 17.5802994
29: -3.8504639, 15.8963280, -3.8772788, 15.9066792, -16.1862335, 16.2526703
30: -10.8652601, 10.3958416, -10.8777323, 10.4042549, -17.7260704, 17.7420006
31: -6.8003769, 12.5155678, -6.8190022, 12.5454130, -18.7118416, 18.7175674
32: -26.4573746, -1.8378782, -26.4826927, -1.8304510, -22.5888977, 22.6324768
33: -43.5274239, -7.8332591, -43.5522995, -7.8149242, -28.8134842, 28.7645569
34: -36.1365204, -6.0710297, -36.1639404, -6.0539603, -22.9081345, 22.8971863
35: -26.7407494, 1.1937828, -26.7624474, 1.2074556, -24.8739853, 24.8543320
36: -27.0015774, 4.8109245, -27.0323925, 4.8195796, -31.3325958, 31.3464127
37: -44.1161880, -9.2407141, -44.1657486, -9.2277040, -28.4592209, 28.4596481
38: -31.4461460, 3.0618310, -31.4753628, 3.0781002, -34.5242462, 34.5371933
39: -48.3652763, -10.6956043, -48.4058609, -10.6796808, -33.7397156, 33.6985168
40: -44.5009880, -17.7557945, -44.5341949, -17.7330208, -19.8541107, 19.8680153
41: -30.4063873, -4.1795211, -30.4382801, -4.1574087, -21.2591248, 21.2716331
42: -19.9451771, -0.2864127, -19.9652367, -0.2671962, -15.3180885, 15.3302994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=126, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1299

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9319635, upper bound: 9.9227447
time: 32.00 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9360438, upper bound: 9.9227447
time: 22.49 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -22.5027122, 9.2880116, -22.4457474, 9.2842865, -31.7267761, 31.6650620
1: -12.0719566, 7.8089890, -12.0550365, 7.8002267, -19.8721828, 19.8640251
2: -11.7281141, 9.6211147, -11.6571608, 9.6220245, -18.8262100, 18.7277756
3: -17.6910019, 7.4533186, -17.6232643, 7.4515963, -23.5850449, 23.4740982
4: -19.6441345, 5.1804237, -19.6283169, 5.1711159, -22.4115219, 22.3662949
5: -15.7054100, 9.6762190, -15.5896492, 9.6766663, -24.2284546, 24.0870895
6: -31.9505177, -7.3958664, -31.9319515, -7.4066582, -19.8787079, 19.8629570
7: -21.7442474, 6.0309286, -21.6354561, 6.0231285, -26.3893280, 26.2496796
8: -23.6940460, 7.6131182, -23.6510849, 7.6024508, -29.5740051, 29.5007172
9: -13.7974529, 10.1278906, -13.7850742, 10.0497189, -20.7167435, 20.7800140
10: -13.9798746, 14.2510128, -13.9646358, 14.1424942, -27.6095810, 27.7155838
11: -10.2270947, 11.3961010, -10.2422132, 11.3831720, -17.5884552, 17.5900421
12: -23.2688026, 13.4370489, -23.2744617, 13.2460804, -34.3714294, 34.5941925
13: -25.3718319, 6.1968780, -25.3643341, 6.1422248, -31.0111237, 31.0317993
14: -26.3359756, 15.0738735, -26.2954407, 14.9009914, -39.5650024, 39.7179871
15: -10.0729456, 13.0404711, -10.0502090, 13.0004959, -21.6934967, 21.7274475
16: -20.9277668, 4.4994535, -20.9277096, 4.4911566, -25.2528534, 25.2051620
17: -23.0794182, 11.3604050, -23.0655613, 11.2575617, -34.3369789, 34.4259644
18: -11.2276583, 16.6206703, -11.2158117, 16.5902920, -26.9337692, 27.0881577
19: -7.2658844, 8.3597603, -7.2586374, 8.3639431, -14.6882362, 14.6825962
20: -6.5987368, 10.0650024, -6.5788651, 10.0441456, -15.4227486, 15.4386520
21: -7.6135845, 11.7957497, -7.6021709, 11.7905140, -18.4139023, 18.4162445
22: -5.0717411, 15.4010563, -5.0615168, 15.3671503, -18.3199081, 18.4239578
23: -3.0122705, 15.0373363, -2.9849825, 15.0451727, -15.7718849, 15.7730255
24: -5.4216213, 13.2813206, -5.3896437, 13.2813778, -14.4213562, 14.4657707
25: -0.9775519, 19.6184273, -0.9632850, 19.6071377, -15.4132309, 15.4792786
26: -12.1220121, 19.8464355, -12.1052132, 19.6777782, -31.7997894, 31.9516487
27: -9.5251484, 10.9457273, -9.4746819, 10.9471474, -19.7367897, 19.7672691
28: -4.2057524, 15.1260061, -4.1917615, 15.1320772, -17.6367111, 17.5925446
29: -3.9081182, 15.9422255, -3.8984604, 15.9101562, -16.2567749, 16.3498840
30: -10.9061470, 10.4319677, -10.8879805, 10.4078331, -17.7994576, 17.7665634
31: -6.8897438, 12.5595274, -6.8359685, 12.5669346, -18.8245735, 18.7701225
32: -26.4800549, -1.8007531, -26.4908371, -1.8237309, -22.6123199, 22.6837158
33: -43.5829468, -7.7813840, -43.5570297, -7.7945900, -28.8995361, 28.8204727
34: -36.1687927, -6.0250869, -36.1706085, -6.0333991, -22.9590302, 22.9589424
35: -26.7855797, 1.2337217, -26.7678871, 1.2241426, -24.9767532, 24.8953247
36: -27.0416794, 4.8426442, -27.0454941, 4.8294806, -31.3799438, 31.4037781
37: -44.1646194, -9.1979351, -44.1785316, -9.2125664, -28.5255814, 28.5513687
38: -31.4861240, 3.1061916, -31.4897060, 3.0892801, -34.5754051, 34.5958977
39: -48.4184189, -10.6509457, -48.4201736, -10.6635876, -33.8150330, 33.8002777
40: -44.5355225, -17.6979847, -44.5374184, -17.7046490, -19.9115982, 19.9182014
41: -30.4638252, -4.1167212, -30.4432869, -4.1306047, -21.3508301, 21.3275223
42: -19.9741993, -0.2347064, -19.9728966, -0.2503705, -15.3689194, 15.3934975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=126, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=149, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1299

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9320413, upper bound: 9.9227447
time: 25.51 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9361215, upper bound: 9.9227447
time: 25.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -22.4415970, 9.2455711, -22.3855858, 9.2371559, -31.6099243, 31.5642624
1: -12.0433254, 7.7763934, -12.0193071, 7.7672963, -19.8106213, 19.7957001
2: -11.6586447, 9.5641384, -11.6279345, 9.5790806, -18.7218781, 18.6694641
3: -17.6198215, 7.3851781, -17.5784988, 7.3909016, -23.4312973, 23.3656845
4: -19.5970802, 5.1606483, -19.5738964, 5.1371427, -22.3102417, 22.3098831
5: -15.6152678, 9.5856495, -15.5637712, 9.6121826, -24.0691681, 23.9902496
6: -31.9326878, -7.4134235, -31.9236107, -7.4417210, -19.8028564, 19.7967949
7: -21.6328602, 5.9425931, -21.5935745, 5.9596920, -26.2342911, 26.1391373
8: -23.6246357, 7.5677547, -23.5931721, 7.5508337, -29.4623871, 29.4351349
9: -13.6891880, 10.0481987, -13.7041874, 10.0235538, -20.6229858, 20.6400414
10: -13.8439426, 14.1369104, -13.8767490, 14.1264439, -27.4720001, 27.5091553
11: -10.2206182, 11.3630066, -10.1924648, 11.3485928, -17.5214329, 17.4826469
12: -23.1400433, 13.3106098, -23.1899910, 13.2262545, -34.2378235, 34.3764343
13: -25.3134460, 6.1497173, -25.2855072, 6.0961819, -30.9000549, 30.9227676
14: -26.1918335, 14.9050331, -26.1761436, 14.8937473, -39.4328613, 39.4345398
15: -10.0226097, 12.9837189, -10.0173998, 12.9753685, -21.5707550, 21.6300011
16: -20.8888741, 4.4713016, -20.8796482, 4.4627199, -25.1910782, 25.0791702
17: -22.9975586, 11.2490864, -22.9809113, 11.2395639, -34.2371216, 34.2299957
18: -11.1715422, 16.5628376, -11.1513462, 16.5340347, -26.8607483, 26.8936615
19: -7.2577753, 8.3409348, -7.2187443, 8.3366261, -14.6539078, 14.6200390
20: -6.5812268, 10.0390549, -6.5430512, 10.0246964, -15.3980904, 15.3718605
21: -7.5875282, 11.7760124, -7.5527267, 11.7565212, -18.3661575, 18.3486519
22: -5.0160904, 15.3424816, -5.0057421, 15.3349953, -18.2146339, 18.2724342
23: -2.9958658, 15.0158672, -2.9400578, 14.9966850, -15.7438469, 15.7019501
24: -5.3946271, 13.2616510, -5.3532300, 13.2419252, -14.4016190, 14.3745117
25: -0.9889846, 19.5969791, -0.9263921, 19.5739479, -15.4380188, 15.3814697
26: -11.9705391, 19.7015038, -11.9860306, 19.6346531, -31.6051922, 31.6875343
27: -9.4733019, 10.9158306, -9.4343395, 10.9014549, -19.6829872, 19.6644783
28: -4.2051482, 15.1155205, -4.1508818, 15.0941076, -17.5920753, 17.5471039
29: -3.8989916, 15.8860626, -3.8571177, 15.8724566, -16.2241402, 16.2141151
30: -10.8831692, 10.3922901, -10.8549118, 10.3704948, -17.7270927, 17.7179565
31: -6.8677907, 12.5244322, -6.7913394, 12.5153580, -18.7615738, 18.6911278
32: -26.4839630, -1.7973952, -26.4703102, -1.8492408, -22.6191025, 22.6435242
33: -43.5314713, -7.7603998, -43.4963188, -7.8504710, -28.7781677, 28.8641205
34: -36.1638870, -5.9937935, -36.1417122, -6.0681677, -22.9125977, 22.9829941
35: -26.7618198, 1.2515378, -26.7382202, 1.1847548, -24.8692627, 24.9173203
36: -27.0334148, 4.8759327, -27.0161324, 4.8020072, -31.3465576, 31.4051285
37: -44.1544724, -9.1082802, -44.1206245, -9.2583771, -28.4600830, 28.6009903
38: -31.4699936, 3.1252708, -31.4473038, 3.0611291, -34.5311241, 34.5725746
39: -48.3681717, -10.5900574, -48.3292389, -10.7216101, -33.6871948, 33.8078842
40: -44.5245171, -17.6448803, -44.4908905, -17.7634697, -19.8735657, 19.9703903
41: -30.4447708, -4.1022983, -30.4251804, -4.1758885, -21.2951279, 21.3492546
42: -19.9729614, -0.2505302, -19.9672337, -0.2744608, -15.3427620, 15.3556080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=126, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=150, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1299

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9154955, upper bound: 9.9361188
time: 57.33 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9195772, upper bound: 9.9361188
time: 21.62 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -22.5187969, 9.3100662, -22.3951359, 9.2660255, -31.7181549, 31.6363602
1: -12.0767288, 7.8190427, -12.0281105, 7.7840643, -19.8607941, 19.8471527
2: -11.7409534, 9.6445274, -11.6338224, 9.6202497, -18.8424149, 18.7384720
3: -17.7031517, 7.4736962, -17.5866108, 7.4342422, -23.5911331, 23.4772720
4: -19.6251640, 5.1948199, -19.5789757, 5.1486430, -22.3589096, 22.3517303
5: -15.7440872, 9.7069492, -15.5705709, 9.6756868, -24.2613525, 24.0955429
6: -31.9799919, -7.3550673, -31.9296532, -7.4200191, -19.9098358, 19.8625603
7: -21.7624588, 6.0505815, -21.6045761, 6.0143971, -26.4249496, 26.2508545
8: -23.6787682, 7.6263256, -23.5988121, 7.5755568, -29.5414734, 29.4861298
9: -13.7951078, 10.1397762, -13.7571955, 10.0286140, -20.7050705, 20.7786026
10: -13.9954529, 14.2694721, -13.9545269, 14.1346941, -27.6233521, 27.7258453
11: -10.2544060, 11.3875523, -10.2178288, 11.3516893, -17.6076584, 17.5511551
12: -23.3209972, 13.5194740, -23.2853203, 13.2348003, -34.4048004, 34.6835022
13: -25.3672447, 6.2207055, -25.3063335, 6.1024194, -30.9632568, 31.0131760
14: -26.4099503, 15.0985012, -26.2768974, 14.8959274, -39.6321716, 39.7311707
15: -10.0892353, 13.0496330, -10.0369978, 12.9862957, -21.6598129, 21.7473297
16: -20.9409332, 4.5122042, -20.9074078, 4.4770374, -25.3355560, 25.1612434
17: -23.1294193, 11.3689194, -23.0453453, 11.2464619, -34.3758812, 34.4142647
18: -11.2406263, 16.6084023, -11.1802816, 16.5419083, -26.9272079, 27.0492325
19: -7.3051586, 8.3628216, -7.2384195, 8.3478546, -14.7166080, 14.6608009
20: -6.6333203, 10.0736723, -6.5586824, 10.0274200, -15.4539757, 15.4252796
21: -7.6467886, 11.7934256, -7.5769300, 11.7593746, -18.4299889, 18.3887787
22: -5.1028857, 15.3922920, -5.0413408, 15.3404064, -18.3241997, 18.4226265
23: -3.0562439, 15.0310631, -2.9559259, 15.0032415, -15.8032990, 15.7304535
24: -5.4444666, 13.2778053, -5.3617468, 13.2465382, -14.4423256, 14.4244270
25: -1.0356116, 19.6290703, -0.9405155, 19.5758858, -15.4880371, 15.4402466
26: -12.1620045, 19.8858948, -12.0786543, 19.6420746, -31.8040791, 31.9645500
27: -9.5439930, 10.9406624, -9.4425182, 10.9126396, -19.7367401, 19.7243423
28: -4.2499399, 15.1257505, -4.1636953, 15.0964603, -17.6685753, 17.5593300
29: -3.9567080, 15.9319696, -3.8783364, 15.8759327, -16.2948494, 16.3113194
30: -10.9240551, 10.4284296, -10.8651924, 10.3740978, -17.8004875, 17.7424927
31: -6.9571304, 12.5683746, -6.8083115, 12.5368929, -18.8742599, 18.7436218
32: -26.5065880, -1.7603059, -26.4784584, -1.8425493, -22.6424866, 22.6947708
33: -43.5870705, -7.7085114, -43.5010490, -7.8301725, -28.8641891, 28.9199905
34: -36.1961365, -5.9479156, -36.1484032, -6.0475888, -22.9635162, 23.0447578
35: -26.8066940, 1.2915096, -26.7436600, 1.2014375, -24.9720383, 24.9582977
36: -27.0734978, 4.9076457, -27.0293236, 4.8118820, -31.3938599, 31.4624329
37: -44.2028999, -9.0655556, -44.1334152, -9.2432747, -28.5263596, 28.6926117
38: -31.5099335, 3.1696234, -31.4616928, 3.0723410, -34.5822754, 34.6313171
39: -48.4212837, -10.5454102, -48.3436050, -10.7055159, -33.7625122, 33.9095612
40: -44.5590591, -17.5870934, -44.4941330, -17.7351341, -19.9309921, 20.0206833
41: -30.5022068, -4.0395274, -30.4301872, -4.1490726, -21.3868332, 21.4051132
42: -20.0019798, -0.1988230, -19.9748745, -0.2576404, -15.3935738, 15.4187927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=126, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=150, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1299

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9155748, upper bound: 9.9361188
time: 23.04 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9196551, upper bound: 9.9361188
time: 13.10 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -22.4712524, 9.2491426, -22.4384785, 9.2663383, -31.6774750, 31.6193848
1: -12.0580559, 7.7790613, -12.0467224, 7.7871413, -19.8451977, 19.8257828
2: -11.6720915, 9.5669155, -11.6522636, 9.5921307, -18.7500610, 18.6857300
3: -17.6414871, 7.3891916, -17.6160564, 7.4171534, -23.4745712, 23.3842087
4: -19.6267376, 5.1640573, -19.6260910, 5.1637816, -22.3770981, 22.3432465
5: -15.6265240, 9.5905294, -15.5836906, 9.6287766, -24.1014481, 24.0162201
6: -31.9432812, -7.4090161, -31.9439964, -7.4269314, -19.8022461, 19.8478622
7: -21.6504555, 5.9459162, -21.6257496, 5.9765158, -26.2710876, 26.1604614
8: -23.6539536, 7.5709653, -23.6458778, 7.5816288, -29.5260010, 29.4656296
9: -13.7067204, 10.0520658, -13.7347612, 10.0484114, -20.6597824, 20.6536407
10: -13.8485012, 14.1431828, -13.8895159, 14.1412210, -27.4916077, 27.5286789
11: -10.2244415, 11.3811626, -10.2194386, 11.3809814, -17.5460052, 17.5299530
12: -23.1462135, 13.3172579, -23.2043533, 13.2392092, -34.2573547, 34.4009399
13: -25.3526211, 6.1547208, -25.3555698, 6.1390643, -30.9843750, 30.9801483
14: -26.1998863, 14.9169331, -26.2001801, 14.9162836, -39.4651184, 39.4619141
15: -10.0293236, 12.9875393, -10.0328798, 12.9907665, -21.6132507, 21.6491470
16: -20.9001675, 4.4747286, -20.9016151, 4.4808125, -25.1885147, 25.1250992
17: -23.0068016, 11.2519522, -23.0069618, 11.2515583, -34.2583618, 34.2589149
18: -11.1772442, 16.5912018, -11.1905975, 16.5838261, -26.8881378, 26.9691162
19: -7.2628307, 8.3547440, -7.2445374, 8.3612270, -14.6771278, 14.6622391
20: -6.5860043, 10.0542212, -6.5674105, 10.0516319, -15.4183693, 15.4115429
21: -7.5930347, 11.7978554, -7.5833292, 11.7961693, -18.3972664, 18.4001884
22: -5.0208478, 15.3579960, -5.0309963, 15.3613873, -18.2173080, 18.3128471
23: -3.0004869, 15.0461712, -2.9737749, 15.0505705, -15.7724838, 15.7689018
24: -5.3989158, 13.2865448, -5.3837633, 13.2852135, -14.4153786, 14.4363708
25: -0.9932456, 19.6255817, -0.9526672, 19.6232300, -15.4430618, 15.4520798
26: -11.9789591, 19.7228165, -12.0299625, 19.6724243, -31.6513824, 31.7527790
27: -9.4784174, 10.9388084, -9.4707699, 10.9435110, -19.7091408, 19.7280388
28: -4.2108850, 15.1425257, -4.1858187, 15.1424217, -17.6253395, 17.6105347
29: -3.9032683, 15.9094725, -3.8826647, 15.9119453, -16.2292404, 16.2703571
30: -10.8867502, 10.4134579, -10.8815880, 10.4095898, -17.7553482, 17.7648048
31: -6.8730936, 12.5496407, -6.8241310, 12.5618534, -18.7996216, 18.7513351
32: -26.5013256, -1.7936831, -26.5027618, -1.8287134, -22.6224899, 22.6910172
33: -43.5743752, -7.7565517, -43.5740166, -7.8123884, -28.8517761, 28.8742981
34: -36.1891289, -5.9922967, -36.1890182, -6.0523672, -22.9467545, 22.9992256
35: -26.7854042, 1.2536855, -26.7822342, 1.2077775, -24.9096069, 24.9351730
36: -27.0530319, 4.8778796, -27.0547943, 4.8204846, -31.3815308, 31.4351349
37: -44.2012787, -9.1054668, -44.2050247, -9.2262611, -28.5251694, 28.6358871
38: -31.4909744, 3.1271105, -31.4944954, 3.0799985, -34.5709724, 34.6216049
39: -48.4285660, -10.5875988, -48.4367180, -10.6769924, -33.7899780, 33.8385468
40: -44.5684242, -17.6426334, -44.5677109, -17.7314587, -19.8995552, 20.0165443
41: -30.4678154, -4.0992260, -30.4675102, -4.1555157, -21.3074112, 21.3832893
42: -19.9798336, -0.2470665, -19.9814320, -0.2642031, -15.3488274, 15.3828945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=126, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=150, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1299

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9319635, upper bound: 9.9361214
time: 23.68 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9360438, upper bound: 9.9361214
time: 23.43 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -22.5484543, 9.3135471, -22.4480515, 9.2952137, -31.7857208, 31.6916122
1: -12.0914650, 7.8216605, -12.0555172, 7.8038812, -19.8953457, 19.8771782
2: -11.7544165, 9.6472998, -11.6581316, 9.6332760, -18.8706131, 18.7547455
3: -17.7248173, 7.4776754, -17.6241589, 7.4605350, -23.6344376, 23.4958267
4: -19.6548061, 5.1982198, -19.6311474, 5.1752338, -22.4257202, 22.3851013
5: -15.7553406, 9.7118263, -15.5905037, 9.6923027, -24.2936859, 24.1215286
6: -31.9906025, -7.3506355, -31.9500713, -7.4052243, -19.9091949, 19.9136124
7: -21.7800331, 6.0539083, -21.6367321, 6.0312338, -26.4617615, 26.2722015
8: -23.7081356, 7.6295524, -23.6514740, 7.6063695, -29.6050873, 29.5167313
9: -13.8126163, 10.1436291, -13.7877522, 10.0534449, -20.7418671, 20.7922592
10: -13.9999599, 14.2757483, -13.9673061, 14.1495190, -27.6428833, 27.7453918
11: -10.2582350, 11.4056950, -10.2447748, 11.3840599, -17.6322269, 17.5984650
12: -23.3271275, 13.5261507, -23.2996712, 13.2477407, -34.4242706, 34.7079697
13: -25.4064331, 6.2256875, -25.3764324, 6.1452718, -31.0475769, 31.0706177
14: -26.4180031, 15.1103678, -26.3008671, 14.9184494, -39.6645203, 39.7585144
15: -10.0959501, 13.0534401, -10.0525045, 13.0017414, -21.7023010, 21.7664795
16: -20.9522324, 4.5156498, -20.9293785, 4.4951100, -25.3329849, 25.2071381
17: -23.1387005, 11.3717909, -23.0713921, 11.2584696, -34.3971710, 34.4431839
18: -11.2463913, 16.6367798, -11.2195377, 16.5917416, -26.9546738, 27.1246414
19: -7.3102016, 8.3766346, -7.2641954, 8.3724556, -14.7398052, 14.7029877
20: -6.6380949, 10.0888405, -6.5830288, 10.0543823, -15.4742737, 15.4649429
21: -7.6523380, 11.8152485, -7.6075459, 11.7989998, -18.4611206, 18.4403038
22: -5.1076503, 15.4077969, -5.0665631, 15.3667755, -18.3269157, 18.4630470
23: -3.0608668, 15.0613804, -2.9896250, 15.0571203, -15.8319283, 15.7973785
24: -5.4487762, 13.3026934, -5.3922691, 13.2898197, -14.4560661, 14.4862747
25: -1.0398974, 19.6576710, -0.9667678, 19.6251526, -15.4931049, 15.5108566
26: -12.1704035, 19.9072094, -12.1226826, 19.6799126, -31.8503151, 32.0298920
27: -9.5491257, 10.9636507, -9.4789429, 10.9546938, -19.7628746, 19.7878494
28: -4.2556510, 15.1527729, -4.1986561, 15.1447811, -17.7018051, 17.6227493
29: -3.9610271, 15.9553699, -3.9038820, 15.9154139, -16.2999649, 16.3676071
30: -10.9276323, 10.4495945, -10.8918171, 10.4131908, -17.8287239, 17.7893677
31: -6.9624119, 12.5935955, -6.8410759, 12.5834017, -18.9122772, 18.8038292
32: -26.5240040, -1.7565327, -26.5109119, -1.8220119, -22.6459274, 22.7422943
33: -43.6298676, -7.7046309, -43.5787354, -7.7920704, -28.9377441, 28.9302139
34: -36.2214241, -5.9464407, -36.1957016, -6.0318198, -22.9976959, 23.0609932
35: -26.8302383, 1.2936258, -26.7877045, 1.2244673, -25.0124359, 24.9761581
36: -27.0931072, 4.9096632, -27.0679855, 4.8303876, -31.4289093, 31.4925385
37: -44.2496338, -9.0627232, -44.2177429, -9.2111244, -28.5914917, 28.7275696
38: -31.5309124, 3.1714535, -31.5088501, 3.0911903, -34.6221008, 34.6803055
39: -48.4816704, -10.5429173, -48.4510117, -10.6608582, -33.8652344, 33.9402618
40: -44.6029129, -17.5848389, -44.5709229, -17.7030849, -19.9570427, 20.0667419
41: -30.5252247, -4.0364499, -30.4725037, -4.1286850, -21.3991318, 21.4391403
42: -20.0088577, -0.1953773, -19.9890594, -0.2473783, -15.3996334, 15.4460793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=126, inp2_unstable=127, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=150, inp2_unstable=149, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=15, inp2_unstable=15, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=34, inp2_unstable=34, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1693
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1337
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1709
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 1318
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1455
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1299

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 713

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9320413, upper bound: 9.9361214
time: 23.89 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 25, lower bound: -9.9361215, upper bound: 9.9361214
time: 24.29 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 50.22 seconds
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9319635, upper bound: 9.9227447
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9360438, upper bound: 9.9227447
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9320413, upper bound: 9.9227447
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9361215, upper bound: 9.9227447
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9154955, upper bound: 9.9361188
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9195772, upper bound: 9.9361188
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9155748, upper bound: 9.9361188
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9196551, upper bound: 9.9361188
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9319635, upper bound: 9.9361214
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9360438, upper bound: 9.9361214
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9320413, upper bound: 9.9361214
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 50.22
Output dim: 25, lower bound: -9.9361215, upper bound: 9.9361214

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 42.93 + 615.62 = 658.54 seconds
