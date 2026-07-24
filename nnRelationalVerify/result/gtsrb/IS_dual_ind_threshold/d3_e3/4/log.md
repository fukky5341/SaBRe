## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 7200 seconds
Split limit: 100
Threshold: 33.4998133448


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=223, inp2_unstable=223, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=212, inp2_unstable=212, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=18, inp2_unstable=18, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-29.3687057, 28.3436661, -29.3687057, 28.3436661, -57.7123718, 57.7123718)
1: (-14.3686533, 23.1169357, -14.3686533, 23.1169357, -37.4855881, 37.4855881)
2: (-20.8331318, 23.2205544, -20.8331318, 23.2205544, -43.2646713, 43.2646713)
3: (-27.1577568, 23.1048012, -27.1577568, 23.1048012, -48.4871216, 48.4871140)
4: (-32.6859093, 17.9857941, -32.6859093, 17.9857941, -50.2830048, 50.2830048)
5: (-28.1991749, 25.0023537, -28.1991749, 25.0023537, -53.1612701, 53.1612625)
6: (-41.8412781, 7.6574535, -41.8412781, 7.6574535, -47.5941086, 47.5941124)
7: (-28.7920551, 29.0157146, -28.7920551, 29.0157146, -57.8077698, 57.8077698)
8: (-36.2347603, 23.7399025, -36.2347603, 23.7399025, -58.9520111, 58.9520187)
9: (-22.6215839, 22.7077217, -22.6215839, 22.7077217, -45.3293076, 45.3293076)
10: (-30.6834221, 30.5571270, -30.6834221, 30.5571270, -61.2405472, 61.2405472)
11: (-24.4137306, 25.3683167, -24.4137306, 25.3683167, -49.1342850, 49.1342812)
12: (-46.1738358, 35.1112747, -46.1738358, 35.1112747, -81.2851105, 81.2851105)
13: (-48.3583984, 19.7234249, -48.3583984, 19.7234249, -68.0818253, 68.0818253)
14: (-64.3322525, 26.6960144, -64.3322525, 26.6960144, -89.7105408, 89.7105408)
15: (-34.0794830, 12.3438988, -34.0794830, 12.3438988, -46.4233818, 46.4233818)
16: (-29.4932594, 21.6498260, -29.4932594, 21.6498260, -51.1430855, 51.1430855)
17: (-59.0885239, 28.6373005, -59.0885239, 28.6373005, -87.7258224, 87.7258224)
18: (-28.2209415, 25.2926254, -28.2209415, 25.2926254, -53.5135651, 53.5135651)
19: (-24.2912693, 11.1597157, -24.2912693, 11.1597157, -35.0300903, 35.0300903)
20: (-16.6355476, 16.1454773, -16.6355476, 16.1454773, -32.7810249, 32.7810249)
21: (-23.1503315, 15.7911301, -23.1503315, 15.7911301, -38.9414597, 38.9414597)
22: (-29.8906784, 18.1935844, -29.8906784, 18.1935844, -48.0842628, 48.0842628)
23: (-17.7845211, 20.5397320, -17.7845211, 20.5397320, -37.9173088, 37.9173126)
24: (-22.4685593, 16.5004807, -22.4685593, 16.5004807, -38.0747757, 38.0747757)
25: (-21.8450851, 20.7178764, -21.8450851, 20.7178764, -41.8102646, 41.8102684)
26: (-48.7503052, 23.4067230, -48.7503052, 23.4067230, -72.1570282, 72.1570282)
27: (-27.3533859, 21.0031128, -27.3533859, 21.0031128, -48.2104263, 48.2104263)
28: (-18.8090534, 22.7977848, -18.8090534, 22.7977848, -40.2101440, 40.2101479)
29: (-27.1733360, 20.9911938, -27.1733360, 20.9911938, -47.1226959, 47.1226959)
30: (-21.8117905, 18.9658546, -21.8117905, 18.9658546, -39.9472885, 39.9472923)
31: (-25.4966908, 18.5137157, -25.4966908, 18.5137157, -44.0104065, 44.0104065)
32: (-33.1184769, 16.2344227, -33.1184769, 16.2344227, -49.2135391, 49.2135315)
33: (-63.7556000, 12.8170786, -63.7556000, 12.8170786, -74.6543121, 74.6543198)
34: (-47.6960907, 8.2922297, -47.6960907, 8.2922297, -51.6974487, 51.6974564)
35: (-45.0071297, 11.5211735, -45.0071297, 11.5211735, -54.6439056, 54.6439056)
36: (-51.4329529, 13.5137806, -51.4329529, 13.5137806, -64.9467316, 64.9467316)
37: (-67.4206390, 1.4809723, -67.4206390, 1.4809723, -66.6819916, 66.6819992)
38: (-54.0977440, 16.3466244, -54.0977440, 16.3466244, -70.4443665, 70.4443665)
39: (-73.4575043, 1.7182684, -73.4575043, 1.7182684, -75.0631104, 75.0631104)
40: (-58.5168610, 0.1258335, -58.5168610, 0.1258335, -55.5951614, 55.5951691)
41: (-42.3647079, 12.1001272, -42.3647079, 12.1001272, -50.3985748, 50.3985901)
42: (-26.1969643, 15.6938057, -26.1969643, 15.6938057, -38.9739914, 38.9739876)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.99 + 45.25 = 48.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 11, lower bound: -33.5199253, upper bound: 33.5199253

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 1689
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1624
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1671
type: A, layer: 1, pos: 1673
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1708
type: A, layer: 1, pos: 1720
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1675
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1688
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1450
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 1463
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1397
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1446
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1303
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1461
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 1482
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 1623
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 988
type: A, layer: 1, pos: 981
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 1479
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 937
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 886
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1719
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1704
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1346
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 952
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1529
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1621
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1302
type: A, layer: 1, pos: 896

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1657

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 11, lower bound: -33.5038545, upper bound: 33.4472876
time: 29.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 11, lower bound: -33.5038545, upper bound: 33.5038543
time: 30.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 59.51 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 59.51
Output dim: 11, lower bound: -33.5038545, upper bound: 33.4472876
IS_A2, status: Status.UNKNOWN, split count: 1, time: 59.51
Output dim: 11, lower bound: -33.5038545, upper bound: 33.5038543

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -29.1598206, 28.2950459, -29.2647114, 28.3192596, -57.4790802, 57.5597572
1: -14.2994547, 23.0899372, -14.3342590, 23.1034317, -37.4028854, 37.4241943
2: -20.6293259, 23.1891212, -20.7320976, 23.2048531, -43.1684952, 43.2645416
3: -26.9166164, 23.0633640, -27.0384827, 23.0840149, -48.0899277, 48.1929779
4: -32.4729691, 17.9465485, -32.5774612, 17.9662628, -50.3259277, 50.4290466
5: -27.9727459, 24.9606838, -28.0869980, 24.9813824, -52.9276886, 53.0171509
6: -41.7829514, 7.5294552, -41.8122482, 7.5922089, -47.4076004, 47.3416672
7: -28.5906429, 28.9799042, -28.6918831, 28.9978409, -57.5884857, 57.6717873
8: -36.0676308, 23.7046185, -36.1515808, 23.7222061, -58.7590027, 58.8254242
9: -22.5177174, 22.6364403, -22.5692902, 22.6708031, -45.1885223, 45.2057304
10: -30.6076488, 30.4653130, -30.6456680, 30.5114403, -61.1190872, 61.1109810
11: -24.3415833, 25.1943722, -24.3776722, 25.2778168, -48.9561882, 48.9068756
12: -46.1244965, 34.6305542, -46.1492538, 34.8728638, -80.9973602, 80.7798080
13: -48.2707939, 19.6089668, -48.3140717, 19.6661034, -67.9368973, 67.9230347
14: -64.2138824, 26.3683510, -64.2735443, 26.5329170, -89.4751434, 89.3620453
15: -33.9628830, 12.2808475, -34.0203094, 12.3124456, -46.2753296, 46.3011551
16: -29.4100342, 21.6045532, -29.4516773, 21.6270981, -51.0371323, 51.0562286
17: -59.0019264, 28.1966553, -59.0455704, 28.4182453, -87.4201736, 87.2422256
18: -28.1474667, 25.0704117, -28.1837597, 25.1822662, -53.3297348, 53.2541733
19: -24.2126770, 11.1435986, -24.2510128, 11.1517229, -35.1132278, 35.1437912
20: -16.5630646, 16.1086769, -16.5993481, 16.1271477, -32.6902122, 32.7080231
21: -23.0781708, 15.7568607, -23.1142349, 15.7740049, -38.8521767, 38.8710938
22: -29.8202457, 18.0865746, -29.8553543, 18.1400146, -47.9602585, 47.9419289
23: -17.7297649, 20.5211468, -17.7569008, 20.5304260, -37.8580704, 37.8784676
24: -22.4010086, 16.4702854, -22.4326477, 16.4855347, -38.1062546, 38.1352539
25: -21.7624989, 20.6797466, -21.8032913, 20.6988907, -41.7477341, 41.7697411
26: -48.6771164, 23.0959625, -48.7136688, 23.2529144, -71.9300308, 71.8096313
27: -27.2539635, 20.8564262, -27.3022938, 20.9304924, -48.0441666, 48.0153427
28: -18.7408066, 22.7314758, -18.7750111, 22.7645721, -40.1079712, 40.1098099
29: -27.1164284, 20.7764244, -27.1450062, 20.8842125, -47.0044785, 46.9236374
30: -21.7553234, 18.8668919, -21.7836246, 18.9150257, -39.8459244, 39.8209457
31: -25.3907986, 18.4928360, -25.4428310, 18.5032978, -43.8940964, 43.9356689
32: -33.0546112, 16.0568962, -33.0867920, 16.1434155, -49.1980286, 49.1436882
33: -63.4492073, 12.7432137, -63.6035233, 12.7803364, -74.1860504, 74.3081055
34: -47.6152344, 8.2464390, -47.6557503, 8.2691212, -51.5348816, 51.5529099
35: -44.8728676, 11.4795389, -44.9392776, 11.5004721, -54.4934387, 54.5405655
36: -51.3741913, 13.3689852, -51.4032784, 13.4419308, -64.8161240, 64.7722626
37: -67.3219452, 1.4055643, -67.3698120, 1.4433670, -66.5078735, 66.5167236
38: -54.0190239, 16.2676239, -54.0581627, 16.3065319, -70.3255539, 70.3257904
39: -73.2615509, 1.6732426, -73.3555908, 1.6959648, -74.9472351, 75.0206757
40: -58.4328918, 0.0927525, -58.4747124, 0.1092606, -55.3887329, 55.4205322
41: -42.2874641, 12.0020666, -42.3262177, 12.0505085, -50.2616653, 50.2481880
42: -26.1382179, 15.6189594, -26.1677361, 15.6539259, -38.9639053, 38.9311829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=222, inp2_unstable=223, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=211, inp2_unstable=211, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=18, inp2_unstable=18, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 896

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.4493716, upper bound: 33.4294272
time: 35.77 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.4855635, upper bound: 33.4294272
time: 26.54 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -29.4136143, 28.5772591, -29.3565941, 28.3405914, -57.7542038, 57.9338531
1: -14.4202852, 23.2820969, -14.3616180, 23.1148224, -37.5351067, 37.6437149
2: -20.8446808, 23.4684601, -20.8223000, 23.2182388, -43.2504272, 43.5292740
3: -27.1700211, 23.4033909, -27.1462727, 23.1016407, -48.4867172, 48.7777443
4: -32.7055588, 18.2978401, -32.6764145, 17.9833450, -50.2629700, 50.6282349
5: -28.2100639, 25.3092251, -28.1875744, 24.9995193, -53.1690979, 53.4608154
6: -42.0031128, 7.6794281, -41.8376007, 7.6293392, -47.7988892, 47.6029701
7: -28.8484421, 29.1924305, -28.7810917, 29.0132427, -57.8616867, 57.9735222
8: -36.2652817, 23.9715576, -36.2252731, 23.7368069, -58.9773407, 59.1810608
9: -22.6875668, 22.8736629, -22.6126137, 22.7023849, -45.3899536, 45.4862747
10: -30.8743725, 30.7062817, -30.6786308, 30.5505219, -61.4248962, 61.3849106
11: -24.9240265, 25.3750153, -24.4092140, 25.3571205, -49.6389275, 49.1305313
12: -46.6714287, 35.1394882, -46.1701050, 35.0856247, -81.7570496, 81.3095932
13: -48.3786697, 19.8600121, -48.3352318, 19.7141037, -68.0927734, 68.1952438
14: -64.6957245, 26.7078209, -64.3233185, 26.6791725, -90.0876312, 89.6962128
15: -34.0961533, 12.5769758, -34.0505981, 12.3393593, -46.4355125, 46.6275749
16: -29.7352810, 21.7843285, -29.4878960, 21.6460037, -51.3812866, 51.2722244
17: -59.7811356, 28.6479893, -59.0825920, 28.6145611, -88.3956985, 87.7305832
18: -28.5058556, 25.3223343, -28.2134438, 25.2817383, -53.7875938, 53.5357780
19: -24.5439301, 11.1864662, -24.2863312, 11.1586351, -35.2785721, 35.0889359
20: -16.7995567, 16.1782551, -16.6304054, 16.1427536, -32.9423103, 32.8086624
21: -23.4874649, 15.8112812, -23.1451950, 15.7849922, -39.2724571, 38.9564743
22: -30.0974312, 18.2419529, -29.8856392, 18.1850777, -48.2825089, 48.1275940
23: -18.0190430, 20.5604973, -17.7806721, 20.5380898, -38.1479073, 37.9527283
24: -22.5489273, 16.5691643, -22.4609413, 16.4986420, -38.1325150, 38.1700020
25: -21.9497375, 20.7702713, -21.8326759, 20.7152424, -41.9210472, 41.8767700
26: -49.2342682, 23.4184380, -48.7441559, 23.3912926, -72.6255646, 72.1625977
27: -27.5956345, 21.0230141, -27.3454266, 20.9960613, -48.4615173, 48.2035980
28: -19.0642433, 22.8072739, -18.8047581, 22.7937202, -40.4634857, 40.2165680
29: -27.4951153, 20.9941883, -27.1694927, 20.9789371, -47.4438324, 47.1220284
30: -22.0633545, 18.9833717, -21.8077908, 18.9571972, -40.2016449, 39.9547882
31: -25.6620770, 18.6289177, -25.4889240, 18.5125523, -44.1746292, 44.1178436
32: -33.2307968, 16.2708111, -33.1141968, 16.2146683, -49.3769150, 49.2386093
33: -63.7891884, 13.2443390, -63.7390747, 12.8120670, -74.6621704, 75.0794449
34: -47.7559128, 8.4969196, -47.6892929, 8.2886906, -51.7777481, 51.8939743
35: -45.0426369, 11.6947870, -44.9901123, 11.5177040, -54.6678162, 54.8250504
36: -51.6273956, 13.5640640, -51.4287834, 13.5060472, -65.1334457, 64.9928436
37: -67.5575714, 1.5458069, -67.4142914, 1.4744539, -66.8138428, 66.7328491
38: -54.2855988, 16.4451122, -54.0911331, 16.3355598, -70.6211548, 70.5362473
39: -73.5181885, 2.0968761, -73.4450607, 1.7149200, -75.1123199, 75.4406433
40: -58.5708847, 0.2603159, -58.5109940, 0.1233749, -55.6472626, 55.7238770
41: -42.4714317, 12.1236515, -42.3604050, 12.0759878, -50.5177612, 50.4179611
42: -26.3174953, 15.7403097, -26.1938438, 15.6781158, -39.1860619, 38.9701195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=222, inp2_unstable=223, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=212, inp2_unstable=212, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=18, inp2_unstable=18, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1624
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1671
type: B, layer: 1, pos: 1673
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1708
type: B, layer: 1, pos: 1720
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1675
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1688
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1450
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1463
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1397
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1446
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1303
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1461
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 1482
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1623
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 988
type: B, layer: 1, pos: 981
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 1479
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 937
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 886
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1719
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1346
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 952
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1529
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1706
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1621
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1302
type: B, layer: 1, pos: 896

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1689

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.4493716, upper bound: 33.4855635
time: 32.16 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 11, lower bound: -33.4855635, upper bound: 33.4855635
time: 30.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 64.67 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 64.67
Output dim: 11, lower bound: -33.4493716, upper bound: 33.4294272
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 64.67
Output dim: 11, lower bound: -33.4855635, upper bound: 33.4294272
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 64.67
Output dim: 11, lower bound: -33.4493716, upper bound: 33.4855635
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 64.67
Output dim: 11, lower bound: -33.4855635, upper bound: 33.4855635

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 48.24 + 188.93 = 237.17 seconds
