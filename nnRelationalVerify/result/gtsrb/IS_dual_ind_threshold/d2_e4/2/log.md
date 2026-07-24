## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 3600 seconds
Split limit: 100
Threshold: 46.7201561769


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-41.8581047, 32.1681671, -41.8581047, 32.1681671, -74.0262756, 74.0262756)
1: (-27.6238575, 29.8764305, -27.6238575, 29.8764305, -57.5002899, 57.5002899)
2: (-22.0896931, 25.5827484, -22.0896931, 25.5827484, -47.6724396, 47.6724396)
3: (-25.4082108, 31.8256626, -25.4082108, 31.8256626, -57.2338715, 57.2338715)
4: (-29.0485497, 28.7367477, -29.0485497, 28.7367477, -57.7852974, 57.7852974)
5: (-28.2786865, 29.0868778, -28.2786865, 29.0868778, -57.3655624, 57.3655624)
6: (-47.3176498, 14.1689835, -47.3176498, 14.1689835, -61.0804901, 61.0804787)
7: (-38.5263596, 25.7286282, -38.5263596, 25.7286282, -64.2549896, 64.2549896)
8: (-35.1668663, 29.4639416, -35.1668663, 29.4639416, -64.6308060, 64.6308060)
9: (-22.0164375, 30.4925995, -22.0164375, 30.4925995, -52.5090370, 52.5090370)
10: (-41.9452057, 32.8714218, -41.9452057, 32.8714218, -74.8166275, 74.8166275)
11: (-48.6259537, 19.9987221, -48.6259537, 19.9987221, -68.6246796, 68.6246796)
12: (-43.4501343, 25.0467262, -43.4501343, 25.0467262, -68.4968567, 68.4968567)
13: (-30.5095196, 38.7206345, -30.5095196, 38.7206345, -69.2301559, 69.2301559)
14: (-77.4074554, 5.2998600, -77.4074554, 5.2998600, -82.7073135, 82.7073135)
15: (-29.5153503, 36.1136436, -29.5153503, 36.1136436, -65.6289978, 65.6289978)
16: (-46.8135033, 28.1260738, -46.8135033, 28.1260738, -74.9395752, 74.9395752)
17: (-79.2747726, 16.1380520, -79.2747726, 16.1380520, -95.4128265, 95.4128265)
18: (-45.1141205, 18.2228985, -45.1141205, 18.2228985, -63.3370209, 63.3370209)
19: (-36.2994308, 11.3947449, -36.2994308, 11.3947449, -47.6941757, 47.6941757)
20: (-31.0101185, 15.6616783, -31.0101185, 15.6616783, -46.6717987, 46.6717987)
21: (-43.6235428, 14.5215607, -43.6235428, 14.5215607, -58.1451035, 58.1451035)
22: (-37.6735344, 21.3282833, -37.6735344, 21.3282833, -59.0018158, 59.0018158)
23: (-32.3645401, 18.2318859, -32.3645401, 18.2318859, -50.5964279, 50.5964279)
24: (-30.8573360, 16.2691994, -30.8573360, 16.2691994, -47.1265335, 47.1265335)
25: (-26.9438877, 24.1879005, -26.9438877, 24.1879005, -51.1317902, 51.1317902)
26: (-50.0373726, 24.3668594, -50.0373726, 24.3668594, -74.4042358, 74.4042358)
27: (-41.6567307, 12.1853085, -41.6567307, 12.1853085, -53.8420410, 53.8420410)
28: (-32.7484970, 22.9294147, -32.7484970, 22.9294147, -55.6779099, 55.6779099)
29: (-37.8354340, 17.8038578, -37.8354340, 17.8038578, -55.6392899, 55.6392899)
30: (-41.0930443, 21.7399483, -41.0930443, 21.7399483, -62.8329926, 62.8329926)
31: (-41.8162842, 14.3614702, -41.8162842, 14.3614702, -56.1777534, 56.1777534)
32: (-36.9501724, 18.5844498, -36.9501724, 18.5844498, -55.5346222, 55.5346222)
33: (-46.8246155, 30.6463928, -46.8246155, 30.6463928, -77.4710083, 77.4710083)
34: (-44.5230598, 25.8768120, -44.5230598, 25.8768120, -70.3998718, 70.3998718)
35: (-35.8817444, 29.5253220, -35.8817444, 29.5253220, -65.4070663, 65.4070663)
36: (-39.7122803, 26.8040981, -39.7122803, 26.8040981, -66.5163803, 66.5163803)
37: (-60.5529633, 23.0308647, -60.5529633, 23.0308647, -83.5838318, 83.5838318)
38: (-56.7473831, 26.3439636, -56.7473831, 26.3439636, -83.0913467, 83.0913467)
39: (-56.9161797, 21.4197426, -56.9161797, 21.4197426, -78.3359222, 78.3359222)
40: (-58.8069649, 16.5587921, -58.8069649, 16.5587921, -74.8190613, 74.8190613)
41: (-38.5574493, 19.0406857, -38.5574493, 19.0406857, -57.5981369, 57.5981369)
42: (-31.9986248, 16.9168606, -31.9986248, 16.9168606, -48.9154854, 48.9154854)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.77 + 67.11 = 69.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -46.7669231, upper bound: 46.7669231

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1725

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7178551, upper bound: 46.7639671
time: 93.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7178551, upper bound: 46.7646324
time: 70.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 164.64 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 164.64
Output dim: 13, lower bound: -46.7178551, upper bound: 46.7639671
IS_A2, status: Status.UNKNOWN, split count: 1, time: 164.64
Output dim: 13, lower bound: -46.7178551, upper bound: 46.7646324

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -41.7467155, 32.1067123, -41.8058128, 32.1562195, -73.9029388, 73.9125214
1: -27.5567398, 29.8329716, -27.5927887, 29.8653107, -57.4220505, 57.4257584
2: -22.0104942, 25.5193367, -22.0468063, 25.5717812, -47.5822754, 47.5661430
3: -25.3032475, 31.7317619, -25.3543091, 31.8117256, -57.1149750, 57.0860710
4: -28.9138718, 28.6460915, -28.9777279, 28.7252884, -57.6391602, 57.6238174
5: -28.1655807, 28.9723396, -28.2177448, 29.0682049, -57.2337875, 57.1900864
6: -47.2345619, 14.0843716, -47.2828522, 14.1462078, -61.0336761, 60.9630814
7: -38.4326820, 25.6528931, -38.4808578, 25.7133217, -64.1460037, 64.1337509
8: -35.0329666, 29.3566608, -35.0962639, 29.4464092, -64.4793777, 64.4529266
9: -21.9302788, 30.4515476, -21.9771309, 30.4790916, -52.4093704, 52.4286804
10: -41.8651657, 32.8119736, -41.9136505, 32.8505745, -74.7157440, 74.7256241
11: -48.5117798, 19.8706169, -48.6025276, 19.9335079, -68.4452896, 68.4731445
12: -43.3656960, 24.9720573, -43.4247742, 25.0158749, -68.3815689, 68.3968353
13: -30.3312130, 38.5928307, -30.4131851, 38.7066650, -69.0378799, 69.0060120
14: -77.2745514, 5.2193851, -77.3731384, 5.2606478, -82.5352020, 82.5925217
15: -29.4415188, 36.0396729, -29.4854183, 36.0990143, -65.5405350, 65.5250931
16: -46.7219391, 28.0558968, -46.7749367, 28.0907707, -74.8127136, 74.8308334
17: -79.1727982, 16.0665913, -79.2468109, 16.1167622, -95.2895584, 95.3134003
18: -44.9918137, 18.0945396, -45.0969086, 18.1549492, -63.1467628, 63.1914482
19: -36.1997566, 11.3103104, -36.2853699, 11.3480930, -47.5478516, 47.5956802
20: -30.9438400, 15.5943012, -30.9976788, 15.6289902, -46.5728302, 46.5919800
21: -43.5094070, 14.4083891, -43.6065903, 14.4594679, -57.9688759, 58.0149803
22: -37.5753098, 21.2421646, -37.6531601, 21.2825203, -58.8578300, 58.8953247
23: -32.2518005, 18.0833359, -32.3508415, 18.1512394, -50.4030380, 50.4341774
24: -30.7481441, 16.1630630, -30.8392086, 16.2124844, -46.9606285, 47.0022736
25: -26.8484077, 24.0600014, -26.9280720, 24.1206093, -50.9690170, 50.9880753
26: -49.8941269, 24.2500153, -50.0115242, 24.3073082, -74.2014313, 74.2615356
27: -41.5274353, 12.0785398, -41.6406479, 12.1292400, -53.6566772, 53.7191887
28: -32.6368904, 22.8027573, -32.7358475, 22.8620186, -55.4989090, 55.5386047
29: -37.7444458, 17.6811295, -37.8153419, 17.7373886, -55.4818344, 55.4964714
30: -41.0113449, 21.6357670, -41.0782318, 21.6895676, -62.7009125, 62.7139969
31: -41.6893616, 14.2584171, -41.8002548, 14.3079529, -55.9973145, 56.0586700
32: -36.8575516, 18.5165882, -36.9112167, 18.5645332, -55.4220848, 55.4278030
33: -46.6594238, 30.5651741, -46.7462921, 30.6215935, -77.2810211, 77.3114624
34: -44.4143219, 25.8230019, -44.4746552, 25.8661404, -70.2804642, 70.2976532
35: -35.7639847, 29.4672279, -35.8281021, 29.5076790, -65.2716675, 65.2953339
36: -39.5932426, 26.7581615, -39.6585808, 26.7913132, -66.3845520, 66.4167404
37: -60.4084702, 22.9698181, -60.4910736, 23.0134869, -83.4219589, 83.4608917
38: -56.5984573, 26.2906952, -56.6855164, 26.3298988, -82.9283600, 82.9762115
39: -56.6789856, 21.3572292, -56.7973061, 21.4096718, -78.0886536, 78.1545334
40: -58.6327591, 16.4871178, -58.7240372, 16.5463943, -74.6671371, 74.6641235
41: -38.4709663, 18.9974251, -38.5213089, 19.0249825, -57.4959488, 57.5187340
42: -31.9502468, 16.8693180, -31.9818096, 16.9000835, -48.8503304, 48.8511276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1790

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.6663558, upper bound: 46.7413496
time: 327.21 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7160923, upper bound: 46.7622042
time: 60.95 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -41.8509636, 32.1660233, -41.8537216, 32.1669044, -74.0178680, 74.0197449
1: -27.6181660, 29.8743858, -27.6204395, 29.8752098, -57.4933777, 57.4948273
2: -22.0860291, 25.5804577, -22.0874825, 25.5813942, -47.6674232, 47.6679382
3: -25.4027290, 31.8225746, -25.4049110, 31.8238392, -57.2265701, 57.2274857
4: -29.0415897, 28.7344952, -29.0440674, 28.7354393, -57.7770309, 57.7785645
5: -28.2720146, 29.0837364, -28.2747307, 29.0850315, -57.3570480, 57.3584671
6: -47.3120880, 14.1655788, -47.3143654, 14.1668968, -61.0499268, 61.0726242
7: -38.5190392, 25.7260075, -38.5219345, 25.7270508, -64.2460938, 64.2479401
8: -35.1599350, 29.4610634, -35.1624413, 29.4621983, -64.6221313, 64.6235046
9: -22.0104446, 30.4884071, -22.0128155, 30.4901142, -52.5005569, 52.5012207
10: -41.9408188, 32.8666954, -41.9425583, 32.8686371, -74.8094559, 74.8092499
11: -48.6211472, 19.9917564, -48.6230888, 19.9946117, -68.6157608, 68.6148453
12: -43.4465790, 25.0406685, -43.4480057, 25.0431099, -68.4896851, 68.4886780
13: -30.4991112, 38.7177582, -30.5033741, 38.7189255, -69.2180328, 69.2211304
14: -77.4013062, 5.2952337, -77.4037399, 5.2971220, -82.6984253, 82.6989746
15: -29.5084686, 36.1110306, -29.5111275, 36.1121140, -65.6205826, 65.6221619
16: -46.8056755, 28.1206741, -46.8088226, 28.1228752, -74.9285507, 74.9294968
17: -79.2692108, 16.1349335, -79.2714233, 16.1362190, -95.4054260, 95.4063568
18: -45.1105576, 18.2154999, -45.1119919, 18.2185287, -63.3290863, 63.3274918
19: -36.2955666, 11.3900585, -36.2971306, 11.3919868, -47.6875534, 47.6871872
20: -31.0074406, 15.6550999, -31.0085258, 15.6576271, -46.6650696, 46.6636276
21: -43.6189156, 14.5149412, -43.6207962, 14.5176687, -58.1365852, 58.1357384
22: -37.6698112, 21.3230019, -37.6713104, 21.3251762, -58.9949875, 58.9943123
23: -32.3613663, 18.2234535, -32.3626480, 18.2269173, -50.5882835, 50.5861015
24: -30.8526173, 16.2631569, -30.8545227, 16.2656364, -47.1182556, 47.1176796
25: -26.9392128, 24.1807404, -26.9411106, 24.1836624, -51.1228752, 51.1218491
26: -50.0324936, 24.3601227, -50.0344734, 24.3628769, -74.3953705, 74.3945923
27: -41.6535797, 12.1789055, -41.6548462, 12.1815491, -53.8351288, 53.8337517
28: -32.7456894, 22.9220657, -32.7468376, 22.9250755, -55.6707649, 55.6689034
29: -37.8316269, 17.7968674, -37.8331299, 17.7997379, -55.6313629, 55.6299973
30: -41.0888863, 21.7340088, -41.0905609, 21.7364082, -62.8252945, 62.8245697
31: -41.8119583, 14.3562708, -41.8137207, 14.3583851, -56.1703415, 56.1699905
32: -36.9449387, 18.5799866, -36.9469910, 18.5817757, -55.5267143, 55.5269775
33: -46.8139191, 30.6429634, -46.8180313, 30.6443291, -77.4582520, 77.4609985
34: -44.5138168, 25.8731422, -44.5175285, 25.8746147, -70.3884277, 70.3906708
35: -35.8716850, 29.5230103, -35.8757935, 29.5239658, -65.3956528, 65.3988037
36: -39.7032776, 26.8016930, -39.7068214, 26.8026505, -66.5059280, 66.5085144
37: -60.5416412, 23.0286713, -60.5462532, 23.0295334, -83.5711746, 83.5749207
38: -56.7370911, 26.3411369, -56.7411079, 26.3422546, -83.0793457, 83.0822449
39: -56.9027061, 21.4180527, -56.9082108, 21.4187374, -78.3214417, 78.3262634
40: -58.7973862, 16.5572834, -58.8012619, 16.5578995, -74.7879105, 74.8116302
41: -38.5513763, 19.0376148, -38.5538406, 19.0388508, -57.5902252, 57.5914536
42: -31.9959145, 16.9136734, -31.9970055, 16.9149265, -48.9108429, 48.9106789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1790

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1751

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.6663558, upper bound: 46.7418195
time: 56.15 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7628572, upper bound: 46.7628576
time: 54.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 112.80 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 112.80
Output dim: 13, lower bound: -46.6663558, upper bound: 46.7413496
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 112.80
Output dim: 13, lower bound: -46.7160923, upper bound: 46.7622042
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 112.80
Output dim: 13, lower bound: -46.6663558, upper bound: 46.7418195
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 112.80
Output dim: 13, lower bound: -46.7628572, upper bound: 46.7628576

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -41.6747589, 31.9785042, -41.5885544, 31.9351482, -73.6099091, 73.5670624
1: -27.5290012, 29.7082386, -27.4761868, 29.6496487, -57.1786499, 57.1844254
2: -21.9869232, 25.3980713, -21.9332466, 25.3586235, -47.3455467, 47.3313179
3: -25.2842598, 31.5697212, -25.2270374, 31.5280437, -56.8123016, 56.7967606
4: -28.8923588, 28.4961166, -28.8413963, 28.4607639, -57.3531227, 57.3375130
5: -28.1455383, 28.8458881, -28.1064663, 28.8445015, -56.9900398, 56.9523544
6: -47.0956726, 14.0566282, -47.0371552, 14.0157700, -60.7444305, 60.6695099
7: -38.4106293, 25.5266113, -38.3758545, 25.4928532, -63.9034805, 63.9024658
8: -35.0127983, 29.1455460, -34.9871063, 29.0790043, -64.0918045, 64.1326523
9: -21.9020004, 30.3852959, -21.8981609, 30.3607025, -52.2627029, 52.2834549
10: -41.8086548, 32.7806473, -41.7789650, 32.7763863, -74.5850372, 74.5596161
11: -48.4169388, 19.8374615, -48.4186287, 19.8583698, -68.2753067, 68.2560883
12: -43.2265778, 24.9444237, -43.1789589, 24.8833008, -68.1098785, 68.1233826
13: -30.2945461, 38.4626007, -30.2971191, 38.4747696, -68.7693176, 68.7597198
14: -77.2196350, 5.1608877, -77.2348709, 5.1422052, -82.3618393, 82.3957596
15: -29.4021492, 35.8850861, -29.3216457, 35.8257980, -65.2279510, 65.2067337
16: -46.6312370, 28.0350075, -46.5945091, 28.0301266, -74.6613617, 74.6295166
17: -79.1091309, 15.9633617, -79.0925598, 15.9219589, -95.0310898, 95.0559235
18: -44.8200760, 18.0715561, -44.7891998, 18.0484848, -62.8685608, 62.8607559
19: -36.1058197, 11.3015814, -36.1085129, 11.3035231, -47.4093437, 47.4100952
20: -30.8810101, 15.5601826, -30.8782978, 15.5218735, -46.4028854, 46.4384804
21: -43.4234390, 14.3917027, -43.4361649, 14.4048223, -57.8282623, 57.8278656
22: -37.5145569, 21.1932087, -37.5179863, 21.1917706, -58.7063293, 58.7111969
23: -32.1462059, 18.0615807, -32.1610298, 18.0882111, -50.2344170, 50.2226105
24: -30.6428089, 16.1475925, -30.6374702, 16.1683102, -46.8111191, 46.7850647
25: -26.7546539, 24.0378952, -26.7499237, 24.0551491, -50.8098030, 50.7878189
26: -49.8114166, 24.2136917, -49.8571815, 24.2062302, -74.0176468, 74.0708771
27: -41.4630775, 12.0521936, -41.5083237, 12.0645752, -53.5276527, 53.5605164
28: -32.5557861, 22.7771034, -32.5872574, 22.7778473, -55.3336334, 55.3643608
29: -37.6802177, 17.6262188, -37.6596909, 17.6337833, -55.3140030, 55.2859116
30: -40.9085846, 21.5977783, -40.8851013, 21.5867805, -62.4953651, 62.4828796
31: -41.5224533, 14.2425900, -41.4992867, 14.2411242, -55.7635765, 55.7418747
32: -36.7665024, 18.4883919, -36.7346840, 18.4712830, -55.2377853, 55.2230759
33: -46.5594254, 30.5468140, -46.5630569, 30.5388184, -77.0982437, 77.1098709
34: -44.3141212, 25.7987671, -44.2953262, 25.7619362, -70.0760574, 70.0940933
35: -35.6778412, 29.4502563, -35.6735001, 29.4419956, -65.1198349, 65.1237564
36: -39.5309906, 26.7442856, -39.5417252, 26.7358475, -66.2668381, 66.2860107
37: -60.2000580, 22.9529457, -60.1130676, 22.8998318, -83.0998917, 83.0660095
38: -56.4934921, 26.2634621, -56.4837799, 26.2223225, -82.7158127, 82.7472382
39: -56.5653725, 21.3392048, -56.5779953, 21.3393974, -77.9047699, 77.9171982
40: -58.4544220, 16.4604454, -58.3977890, 16.4182949, -74.3422012, 74.2910461
41: -38.3582878, 18.9787254, -38.3216095, 18.9520149, -57.3103027, 57.2956047
42: -31.8745708, 16.8410263, -31.8440380, 16.8014812, -48.6760521, 48.6850662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1752

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6533201, upper bound: 46.6945513
time: 50.23 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.6533201, upper bound: 46.7395662
time: 63.87 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -41.7403183, 32.0985374, -41.7952042, 32.1422043, -73.8825226, 73.8937378
1: -27.5536900, 29.8254509, -27.5877876, 29.8524513, -57.4061432, 57.4132385
2: -22.0078888, 25.5122776, -22.0424423, 25.5596428, -47.5675316, 47.5547180
3: -25.3009338, 31.7217827, -25.3505001, 31.7947216, -57.0956573, 57.0722809
4: -28.9115314, 28.6372013, -28.9737625, 28.7101479, -57.6216812, 57.6109619
5: -28.1634636, 28.9645195, -28.2143154, 29.0548687, -57.2183304, 57.1788330
6: -47.2261963, 14.0807657, -47.2686768, 14.1402168, -61.0200806, 60.9426270
7: -38.4298935, 25.6449757, -38.4763184, 25.6997509, -64.1296463, 64.1212921
8: -35.0303802, 29.3432674, -35.0919189, 29.4238701, -64.4542542, 64.4351883
9: -21.9266853, 30.4471588, -21.9710503, 30.4717331, -52.3984184, 52.4182091
10: -41.8581696, 32.8087997, -41.9019470, 32.8453217, -74.7034912, 74.7107468
11: -48.5025826, 19.8633118, -48.5871201, 19.9209232, -68.4235077, 68.4504318
12: -43.3559875, 24.9690933, -43.4085236, 25.0108490, -68.3668365, 68.3776169
13: -30.3261127, 38.5847778, -30.4047718, 38.6928139, -69.0189285, 68.9895477
14: -77.2659454, 5.2150364, -77.3586731, 5.2532215, -82.5191650, 82.5737076
15: -29.4361362, 36.0303192, -29.4764099, 36.0830193, -65.5191574, 65.5067291
16: -46.7111969, 28.0530300, -46.7567863, 28.0860634, -74.7972565, 74.8098145
17: -79.1655884, 16.0512581, -79.2349854, 16.0911064, -95.2566986, 95.2862396
18: -44.9811897, 18.0914803, -45.0788803, 18.1498070, -63.1309967, 63.1703606
19: -36.1931458, 11.3089819, -36.2742882, 11.3458672, -47.5390129, 47.5832710
20: -30.9388618, 15.5903721, -30.9892502, 15.6224308, -46.5612946, 46.5796204
21: -43.5017319, 14.4062214, -43.5937119, 14.4557657, -57.9574966, 57.9999313
22: -37.5690384, 21.2319603, -37.6429405, 21.2656269, -58.8346634, 58.8749008
23: -32.2461548, 18.0810699, -32.3421860, 18.1474209, -50.3935776, 50.4232559
24: -30.7405243, 16.1606369, -30.8262062, 16.2084885, -46.9490128, 46.9868431
25: -26.8384876, 24.0568771, -26.9112377, 24.1153221, -50.9538116, 50.9681168
26: -49.8873596, 24.2459393, -50.0005112, 24.3004761, -74.1878357, 74.2464523
27: -41.5210075, 12.0754299, -41.6297684, 12.1239414, -53.6449509, 53.7052002
28: -32.6301041, 22.8000603, -32.7258949, 22.8575268, -55.4876328, 55.5259552
29: -37.7376976, 17.6702385, -37.8042946, 17.7199440, -55.4576416, 55.4745331
30: -41.0040627, 21.6313210, -41.0659180, 21.6820221, -62.6860847, 62.6972389
31: -41.6790314, 14.2556038, -41.7827110, 14.3032265, -55.9822578, 56.0383148
32: -36.8504219, 18.5133591, -36.8994560, 18.5591354, -55.4095573, 55.4128151
33: -46.6519623, 30.5621490, -46.7335548, 30.6166458, -77.2686081, 77.2957001
34: -44.4069252, 25.8193474, -44.4621811, 25.8602104, -70.2671356, 70.2815247
35: -35.7573204, 29.4652519, -35.8166275, 29.5044289, -65.2617493, 65.2818756
36: -39.5878944, 26.7564011, -39.6495590, 26.7884064, -66.3762970, 66.4059601
37: -60.3941574, 22.9678307, -60.4668922, 23.0102463, -83.4044037, 83.4347229
38: -56.5902519, 26.2875328, -56.6729736, 26.3247032, -82.9149551, 82.9605103
39: -56.6701622, 21.3544350, -56.7822227, 21.4052029, -78.0753632, 78.1366577
40: -58.6210403, 16.4840145, -58.7039528, 16.5413933, -74.6502075, 74.6380157
41: -38.4620171, 18.9952927, -38.5075455, 19.0214272, -57.4834442, 57.5028381
42: -31.9446945, 16.8659725, -31.9728031, 16.8944664, -48.8391609, 48.8387756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1752

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.7064227, upper bound: 46.7186398
time: 72.45 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7143675, upper bound: 46.7604813
time: 65.77 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -41.7788925, 32.0377579, -41.6363487, 31.9458332, -73.7247238, 73.6741028
1: -27.5903759, 29.7496452, -27.5037975, 29.6595612, -57.2499390, 57.2534409
2: -22.0624771, 25.4591904, -21.9739208, 25.3682747, -47.4307518, 47.4331131
3: -25.3837376, 31.6605377, -25.2776489, 31.5401440, -56.9238815, 56.9381866
4: -29.0201626, 28.5844841, -28.9077454, 28.4708443, -57.4910049, 57.4922295
5: -28.2520294, 28.9572926, -28.1634445, 28.8613129, -57.1133423, 57.1207352
6: -47.1732063, 14.1377602, -47.0686836, 14.0364294, -60.7606735, 60.7789574
7: -38.4969749, 25.5996742, -38.4169273, 25.5066204, -64.0035934, 64.0166016
8: -35.1397667, 29.2499046, -35.0532722, 29.0948334, -64.2346039, 64.3031769
9: -21.9821548, 30.4221649, -21.9339066, 30.3717766, -52.3539314, 52.3560715
10: -41.8843002, 32.8353386, -41.8078728, 32.7944031, -74.6787033, 74.6432114
11: -48.5262527, 19.9586449, -48.4391785, 19.9194794, -68.4457321, 68.3978271
12: -43.3074837, 25.0129585, -43.2022400, 24.9104385, -68.2179260, 68.2151947
13: -30.4624691, 38.5874901, -30.3872986, 38.4870300, -68.9495010, 68.9747925
14: -77.3463287, 5.2367516, -77.2654800, 5.1786871, -82.5250168, 82.5022278
15: -29.4690151, 35.9564438, -29.3472748, 35.8388596, -65.3078766, 65.3037186
16: -46.7149391, 28.0997791, -46.6284256, 28.0621948, -74.7771301, 74.7282028
17: -79.2054520, 16.0317440, -79.1171265, 15.9414291, -95.1468811, 95.1488724
18: -44.9387779, 18.1925106, -44.8043175, 18.1120682, -63.0508461, 62.9968262
19: -36.2016144, 11.3813438, -36.1203003, 11.3474522, -47.5490646, 47.5016441
20: -30.9446068, 15.6209755, -30.8891659, 15.5505266, -46.4951324, 46.5101395
21: -43.5329437, 14.4982786, -43.4503403, 14.4630136, -57.9959564, 57.9486198
22: -37.6090775, 21.2740326, -37.5361137, 21.2344055, -58.8434830, 58.8101463
23: -32.2557678, 18.2017479, -32.1728439, 18.1639061, -50.4196739, 50.3745918
24: -30.7472553, 16.2476768, -30.6527939, 16.2214775, -46.9687347, 46.9004707
25: -26.8454266, 24.1586246, -26.7629547, 24.1182289, -50.9636536, 50.9215775
26: -49.9497910, 24.3238297, -49.8800888, 24.2617874, -74.2115784, 74.2039185
27: -41.5892105, 12.1525745, -41.5225220, 12.1169014, -53.7061119, 53.6750946
28: -32.6645584, 22.8964329, -32.5982094, 22.8408928, -55.5054512, 55.4946442
29: -37.7673798, 17.7419930, -37.6774864, 17.6961861, -55.4635658, 55.4194794
30: -40.9860687, 21.6960506, -40.8973885, 21.6335907, -62.6196594, 62.5934372
31: -41.6449738, 14.3404160, -41.5127563, 14.2915573, -55.9365311, 55.8531723
32: -36.8538589, 18.5517311, -36.7704926, 18.4884987, -55.3423576, 55.3222237
33: -46.7139778, 30.6245537, -46.6348572, 30.5615215, -77.2754974, 77.2594147
34: -44.4136696, 25.8489017, -44.3382187, 25.7704124, -70.1840820, 70.1871185
35: -35.7855949, 29.5059891, -35.7212410, 29.4582138, -65.2438049, 65.2272339
36: -39.6410561, 26.7877693, -39.5900269, 26.7472038, -66.3882599, 66.3777924
37: -60.3332672, 23.0117531, -60.1682663, 22.9158325, -83.2490997, 83.1800232
38: -56.6322327, 26.3138924, -56.5394249, 26.2346725, -82.8669052, 82.8533173
39: -56.7891426, 21.4000301, -56.6889496, 21.3484726, -78.1376190, 78.0889816
40: -58.6190987, 16.5305824, -58.4751282, 16.4297352, -74.4629822, 74.4385223
41: -38.4387436, 19.0188560, -38.3541718, 18.9658394, -57.3962479, 57.3694267
42: -31.9202423, 16.8852577, -31.8592205, 16.8162823, -48.7365265, 48.7444763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1752

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6533201, upper bound: 46.6948537
time: 58.10 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.6533201, upper bound: 46.7400354
time: 62.28 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -41.8445587, 32.1578369, -41.8431015, 32.1528778, -73.9974365, 74.0009384
1: -27.6151028, 29.8668842, -27.6154423, 29.8623295, -57.4774323, 57.4823265
2: -22.0834274, 25.5733986, -22.0831146, 25.5692787, -47.6527061, 47.6565132
3: -25.4003735, 31.8126316, -25.4010620, 31.8068180, -57.2071915, 57.2136917
4: -29.0392609, 28.7255802, -29.0401020, 28.7202358, -57.7594986, 57.7656822
5: -28.2699089, 29.0759125, -28.2713127, 29.0716629, -57.3415718, 57.3472252
6: -47.3037033, 14.1619902, -47.3001862, 14.1608973, -61.0363617, 61.0521736
7: -38.5162659, 25.7180290, -38.5174179, 25.7135105, -64.2297745, 64.2354431
8: -35.1573257, 29.4476299, -35.1580963, 29.4396400, -64.5969696, 64.6057281
9: -22.0068474, 30.4840488, -22.0067863, 30.4827633, -52.4896088, 52.4908371
10: -41.9338455, 32.8635406, -41.9308777, 32.8633652, -74.7972107, 74.7944183
11: -48.6119347, 19.9844379, -48.6076736, 19.9820480, -68.5939789, 68.5921097
12: -43.4368401, 25.0376568, -43.4317932, 25.0380554, -68.4748993, 68.4694519
13: -30.4940166, 38.7096672, -30.4949646, 38.7050629, -69.1990814, 69.2046356
14: -77.3926620, 5.2908535, -77.3893127, 5.2897148, -82.6823730, 82.6801682
15: -29.5030365, 36.1016769, -29.5020981, 36.0960999, -65.5991364, 65.6037750
16: -46.7948761, 28.1178303, -46.7906914, 28.1181393, -74.9130173, 74.9085236
17: -79.2619476, 16.1196404, -79.2595825, 16.1105881, -95.3725357, 95.3792267
18: -45.0999184, 18.2124481, -45.0939713, 18.2133904, -63.3133087, 63.3064194
19: -36.2889481, 11.3887348, -36.2860603, 11.3897572, -47.6787033, 47.6747971
20: -31.0024433, 15.6511545, -31.0001068, 15.6510658, -46.6535110, 46.6512604
21: -43.6112251, 14.5127296, -43.6078987, 14.5139484, -58.1251755, 58.1206284
22: -37.6635437, 21.3127689, -37.6610527, 21.3082561, -58.9718018, 58.9738235
23: -32.3557281, 18.2211800, -32.3540268, 18.2230759, -50.5788040, 50.5752068
24: -30.8450031, 16.2607269, -30.8415260, 16.2616577, -47.1066589, 47.1022530
25: -26.9292965, 24.1775970, -26.9242764, 24.1783676, -51.1076660, 51.1018753
26: -50.0256958, 24.3560371, -50.0234032, 24.3560734, -74.3817673, 74.3794403
27: -41.6471710, 12.1757936, -41.6439667, 12.1762524, -53.8234253, 53.8197594
28: -32.7389069, 22.9193649, -32.7368584, 22.9206161, -55.6595230, 55.6562233
29: -37.8248596, 17.7859802, -37.8220940, 17.7823124, -55.6071701, 55.6080742
30: -41.0815964, 21.7295437, -41.0782166, 21.7288742, -62.8104706, 62.8077621
31: -41.8016129, 14.3534336, -41.7961502, 14.3536701, -56.1552811, 56.1495819
32: -36.9377594, 18.5767651, -36.9352646, 18.5764160, -55.5141754, 55.5120316
33: -46.8065033, 30.6399384, -46.8053055, 30.6393547, -77.4458618, 77.4452438
34: -44.5064087, 25.8694572, -44.5050735, 25.8686829, -70.3750916, 70.3745270
35: -35.8649673, 29.5210094, -35.8643494, 29.5206642, -65.3856354, 65.3853607
36: -39.6979065, 26.7999191, -39.6978378, 26.7997437, -66.4976501, 66.4977570
37: -60.5273895, 23.0267181, -60.5220642, 23.0262833, -83.5536728, 83.5487823
38: -56.7288971, 26.3379459, -56.7286301, 26.3370686, -83.0659637, 83.0665741
39: -56.8938446, 21.4152660, -56.8930893, 21.4142666, -78.3081131, 78.3083572
40: -58.7856674, 16.5541763, -58.7812271, 16.5528412, -74.7709732, 74.7854767
41: -38.5424194, 19.0354614, -38.5400925, 19.0352898, -57.5777092, 57.5755539
42: -31.9903698, 16.9102669, -31.9879875, 16.9093018, -48.8996735, 48.8982544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=327, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1752

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7528268, upper bound: 46.7190136
time: 55.49 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.6533201, upper bound: 46.7611190
time: 59.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 117.39 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 117.39
Output dim: 13, lower bound: -46.6533201, upper bound: 46.6945513
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 117.39
Output dim: 13, lower bound: -46.6533201, upper bound: 46.7395662
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 117.39
Output dim: 13, lower bound: -46.7064227, upper bound: 46.7186398
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 117.39
Output dim: 13, lower bound: -46.7143675, upper bound: 46.7604813
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 117.39
Output dim: 13, lower bound: -46.6533201, upper bound: 46.6948537
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 117.39
Output dim: 13, lower bound: -46.6533201, upper bound: 46.7400354
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 117.39
Output dim: 13, lower bound: -46.7528268, upper bound: 46.7190136
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 117.39
Output dim: 13, lower bound: -46.6533201, upper bound: 46.7611190

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -41.6702156, 31.9721355, -41.5861359, 31.9318542, -73.6020660, 73.5582733
1: -27.5269871, 29.7017822, -27.4750881, 29.6462173, -57.1732025, 57.1768723
2: -21.9856148, 25.3913498, -21.9325676, 25.3551884, -47.3408051, 47.3239174
3: -25.2829361, 31.5619602, -25.2263470, 31.5240459, -56.8069839, 56.7883072
4: -28.8909950, 28.4880733, -28.8406582, 28.4566612, -57.3476562, 57.3287315
5: -28.1440544, 28.8383331, -28.1057339, 28.8406639, -56.9847183, 56.9440689
6: -47.0886917, 14.0546579, -47.0334702, 14.0147572, -60.7347565, 60.6661339
7: -38.4089508, 25.5199699, -38.3749771, 25.4895077, -63.8984604, 63.8949471
8: -35.0111237, 29.1331291, -34.9862518, 29.0726852, -64.0838089, 64.1193848
9: -21.8998299, 30.3809319, -21.8970814, 30.3580360, -52.2578659, 52.2780151
10: -41.8056030, 32.7777443, -41.7774010, 32.7748337, -74.5804367, 74.5551453
11: -48.4082680, 19.8350449, -48.4134293, 19.8571396, -68.2654114, 68.2484741
12: -43.2193069, 24.9425468, -43.1752472, 24.8823204, -68.1016235, 68.1177979
13: -30.2918053, 38.4549179, -30.2956963, 38.4707489, -68.7625580, 68.7506104
14: -77.2164154, 5.1565962, -77.2331696, 5.1399479, -82.3563614, 82.3897629
15: -29.3987503, 35.8779678, -29.3198757, 35.8220978, -65.2208481, 65.1978455
16: -46.6203537, 28.0334816, -46.5890045, 28.0293312, -74.6496887, 74.6224823
17: -79.1040497, 15.9538727, -79.0899200, 15.9169388, -95.0209885, 95.0437927
18: -44.8093147, 18.0701962, -44.7837486, 18.0477676, -62.8570824, 62.8539429
19: -36.1001396, 11.3008881, -36.1055832, 11.3031502, -47.4032898, 47.4064713
20: -30.8771324, 15.5577507, -30.8762398, 15.5205584, -46.3976898, 46.4339905
21: -43.4177437, 14.3905611, -43.4331741, 14.4042101, -57.8219528, 57.8237343
22: -37.5098381, 21.1867085, -37.5155640, 21.1884975, -58.6983337, 58.7022705
23: -32.1394157, 18.0602188, -32.1575508, 18.0875092, -50.2269249, 50.2177696
24: -30.6363029, 16.1460819, -30.6340656, 16.1675320, -46.8038330, 46.7801476
25: -26.7477036, 24.0361633, -26.7462311, 24.0542717, -50.8019753, 50.7823944
26: -49.8039856, 24.2119446, -49.8533936, 24.2052345, -74.0092163, 74.0653381
27: -41.4574890, 12.0507469, -41.5053864, 12.0638685, -53.5213585, 53.5561333
28: -32.5493584, 22.7753277, -32.5839691, 22.7768669, -55.3262253, 55.3592987
29: -37.6753197, 17.6209011, -37.6571312, 17.6310253, -55.3063431, 55.2780304
30: -40.9014664, 21.5949192, -40.8814316, 21.5852890, -62.4867554, 62.4763489
31: -41.5143051, 14.2411251, -41.4950409, 14.2403736, -55.7546768, 55.7361679
32: -36.7617874, 18.4862499, -36.7321968, 18.4701881, -55.2319756, 55.2184448
33: -46.5534821, 30.5450993, -46.5600319, 30.5378933, -77.0913773, 77.1051331
34: -44.3088379, 25.7966938, -44.2925797, 25.7608395, -70.0696793, 70.0892715
35: -35.6728363, 29.4491138, -35.6709061, 29.4413624, -65.1141968, 65.1200180
36: -39.5269089, 26.7434196, -39.5395660, 26.7354336, -66.2623444, 66.2829895
37: -60.1879463, 22.9517746, -60.1069336, 22.8992271, -83.0871735, 83.0587082
38: -56.4876518, 26.2613525, -56.4807320, 26.2211952, -82.7088470, 82.7420807
39: -56.5587654, 21.3372707, -56.5746651, 21.3384037, -77.8971710, 77.9119339
40: -58.4457474, 16.4585838, -58.3934174, 16.4172516, -74.3312683, 74.2869110
41: -38.3508682, 18.9774952, -38.3177376, 18.9513836, -57.3022537, 57.2910881
42: -31.8695984, 16.8388958, -31.8414249, 16.8003654, -48.6699638, 48.6803207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1790

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 607

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6508076, upper bound: 46.7132234
time: 73.23 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.6508076, upper bound: 46.7370654
time: 65.71 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -41.7358589, 32.0921898, -41.7928696, 32.1389999, -73.8748627, 73.8850555
1: -27.5516949, 29.8190536, -27.5867672, 29.8490334, -57.4007263, 57.4058228
2: -22.0065784, 25.5055618, -22.0417480, 25.5562553, -47.5628357, 47.5473099
3: -25.2996750, 31.7140369, -25.3498344, 31.7907677, -57.0904427, 57.0638733
4: -28.9101601, 28.6292305, -28.9730644, 28.7061214, -57.6162796, 57.6022949
5: -28.1620541, 28.9569759, -28.2136097, 29.0510426, -57.2130966, 57.1705856
6: -47.2193832, 14.0787773, -47.2652435, 14.1392078, -61.0107269, 60.9394989
7: -38.4282455, 25.6370449, -38.4754601, 25.6956692, -64.1239166, 64.1125031
8: -35.0287628, 29.3308258, -35.0911102, 29.4175682, -64.4463348, 64.4219360
9: -21.9245472, 30.4433289, -21.9699860, 30.4696236, -52.3941727, 52.4133148
10: -41.8551826, 32.8058777, -41.9004593, 32.8438416, -74.6990204, 74.7063370
11: -48.4945908, 19.8609276, -48.5830803, 19.9197273, -68.4143219, 68.4440079
12: -43.3488159, 24.9672241, -43.4048767, 25.0099030, -68.3587189, 68.3721008
13: -30.3233891, 38.5771637, -30.4033680, 38.6889420, -69.0123291, 68.9805298
14: -77.2626495, 5.2106295, -77.3569794, 5.2510452, -82.5136948, 82.5676117
15: -29.4327698, 36.0232048, -29.4746819, 36.0794449, -65.5122147, 65.4978867
16: -46.7003098, 28.0515194, -46.7512474, 28.0852718, -74.7855835, 74.8027649
17: -79.1605835, 16.0418186, -79.2324371, 16.0861912, -95.2467728, 95.2742538
18: -44.9705009, 18.0901985, -45.0735092, 18.1491222, -63.1196213, 63.1637077
19: -36.1874924, 11.3083420, -36.2714233, 11.3455458, -47.5330391, 47.5797653
20: -30.9349556, 15.5879374, -30.9872856, 15.6211576, -46.5561142, 46.5752220
21: -43.4960060, 14.4050980, -43.5908241, 14.4551592, -57.9511642, 57.9959221
22: -37.5643845, 21.2256107, -37.6405487, 21.2624321, -58.8268166, 58.8661575
23: -32.2393723, 18.0797005, -32.3387375, 18.1467056, -50.3860779, 50.4184380
24: -30.7340813, 16.1592331, -30.8228607, 16.2077847, -46.9418640, 46.9820938
25: -26.8316364, 24.0551758, -26.9076233, 24.1144676, -50.9461060, 50.9627991
26: -49.8794975, 24.2441483, -49.9961853, 24.2995110, -74.1790085, 74.2403336
27: -41.5155144, 12.0740318, -41.6269798, 12.1232338, -53.6387482, 53.7010117
28: -32.6236610, 22.7983170, -32.7226410, 22.8566246, -55.4802856, 55.5209579
29: -37.7327843, 17.6649532, -37.8017616, 17.7172127, -55.4499969, 55.4667130
30: -40.9969177, 21.6284962, -41.0622711, 21.6805935, -62.6775131, 62.6907654
31: -41.6709061, 14.2541943, -41.7785492, 14.3025045, -55.9734116, 56.0327454
32: -36.8457413, 18.5112572, -36.8970795, 18.5580654, -55.4038086, 55.4083366
33: -46.6461258, 30.5604324, -46.7305908, 30.6157475, -77.2618713, 77.2910233
34: -44.4017372, 25.8172417, -44.4593048, 25.8591347, -70.2608719, 70.2765503
35: -35.7523155, 29.4640865, -35.8140945, 29.5038376, -65.2561493, 65.2781830
36: -39.5835571, 26.7555618, -39.6473579, 26.7879696, -66.3715286, 66.4029236
37: -60.3821869, 22.9666939, -60.4608154, 23.0096512, -83.3918381, 83.4275055
38: -56.5846519, 26.2853928, -56.6701736, 26.3236122, -82.9082642, 82.9555664
39: -56.6634674, 21.3524933, -56.7787857, 21.4042053, -78.0676727, 78.1312790
40: -58.6124535, 16.4821053, -58.6996231, 16.5403919, -74.6393280, 74.6339340
41: -38.4546738, 18.9940243, -38.5037537, 19.0207787, -57.4754524, 57.4977798
42: -31.9400425, 16.8638134, -31.9704475, 16.8933754, -48.8334198, 48.8342590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1790

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 607

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6508076, upper bound: 46.6922959
time: 60.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7119205, upper bound: 46.7580294
time: 74.99 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -41.7743950, 32.0314026, -41.6339417, 31.9425716, -73.7169647, 73.6653442
1: -27.5883732, 29.7431927, -27.5027599, 29.6561089, -57.2444839, 57.2459526
2: -22.0611877, 25.4524918, -21.9732246, 25.3648338, -47.4260216, 47.4257164
3: -25.3824120, 31.6527882, -25.2769470, 31.5361538, -56.9185638, 56.9297333
4: -29.0187645, 28.5764580, -28.9070206, 28.4667358, -57.4855003, 57.4834785
5: -28.2505589, 28.9497375, -28.1626930, 28.8574696, -57.1080284, 57.1124306
6: -47.1662216, 14.1357803, -47.0649796, 14.0353699, -60.7510452, 60.7755890
7: -38.4953308, 25.5930710, -38.4160995, 25.5032578, -63.9985886, 64.0091705
8: -35.1380920, 29.2374687, -35.0524025, 29.0884800, -64.2265701, 64.2898712
9: -21.9799786, 30.4178238, -21.9327812, 30.3691177, -52.3490982, 52.3506050
10: -41.8812561, 32.8324280, -41.8062973, 32.7928810, -74.6741333, 74.6387253
11: -48.5176315, 19.9562016, -48.4339752, 19.9182301, -68.4358597, 68.3901749
12: -43.3001862, 25.0110321, -43.1985092, 24.9094601, -68.2096481, 68.2095413
13: -30.4597530, 38.5798111, -30.3858929, 38.4829865, -68.9427414, 68.9657059
14: -77.3430328, 5.2324600, -77.2637634, 5.1764545, -82.5194855, 82.4962234
15: -29.4655895, 35.9493256, -29.3455143, 35.8351746, -65.3007660, 65.2948380
16: -46.7040558, 28.0982780, -46.6228828, 28.0614052, -74.7654572, 74.7211609
17: -79.2003555, 16.0222340, -79.1144562, 15.9364471, -95.1368027, 95.1366882
18: -44.9280319, 18.1911697, -44.7988548, 18.1113415, -63.0393753, 62.9900246
19: -36.1959267, 11.3806362, -36.1173477, 11.3471031, -47.5430298, 47.4979858
20: -30.9407291, 15.6185303, -30.8871021, 15.5492029, -46.4899330, 46.5056305
21: -43.5272293, 14.4971209, -43.4473724, 14.4624157, -57.9896469, 57.9444923
22: -37.6043854, 21.2675323, -37.5336761, 21.2311363, -58.8355217, 58.8012085
23: -32.2489815, 18.2003746, -32.1693573, 18.1632118, -50.4121933, 50.3697319
24: -30.7407227, 16.2461605, -30.6493645, 16.2206879, -46.9614105, 46.8955231
25: -26.8384666, 24.1569176, -26.7592564, 24.1173534, -50.9558182, 50.9161758
26: -49.9423409, 24.3220215, -49.8763313, 24.2608299, -74.2031708, 74.1983490
27: -41.5836334, 12.1511459, -41.5195847, 12.1161509, -53.6997833, 53.6707306
28: -32.6581268, 22.8946323, -32.5949287, 22.8399334, -55.4980621, 55.4895630
29: -37.7624359, 17.7366657, -37.6748924, 17.6934395, -55.4558754, 55.4115601
30: -40.9789581, 21.6931896, -40.8937073, 21.6321011, -62.6110611, 62.5868988
31: -41.6368103, 14.3389759, -41.5084991, 14.2908039, -55.9276123, 55.8474731
32: -36.8491364, 18.5495701, -36.7679977, 18.4874077, -55.3365440, 55.3175659
33: -46.7080688, 30.6227989, -46.6318016, 30.5606060, -77.2686768, 77.2546005
34: -44.4083900, 25.8467865, -44.3355026, 25.7693043, -70.1776962, 70.1822891
35: -35.7805252, 29.5048065, -35.7186241, 29.4575920, -65.2381134, 65.2234344
36: -39.6369324, 26.7869205, -39.5878525, 26.7467461, -66.3836823, 66.3747711
37: -60.3212204, 23.0105934, -60.1621170, 22.9152298, -83.2364502, 83.1727142
38: -56.6263924, 26.3117714, -56.5364532, 26.2335510, -82.8599396, 82.8482208
39: -56.7825470, 21.3980865, -56.6856194, 21.3474407, -78.1299896, 78.0837097
40: -58.6104660, 16.5286903, -58.4706879, 16.4287014, -74.4520264, 74.4343719
41: -38.4312973, 19.0176010, -38.3503075, 18.9652023, -57.3873444, 57.3649216
42: -31.9152889, 16.8831329, -31.8566532, 16.8151550, -48.7304459, 48.7397842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1790

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 607

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.7084613, upper bound: 46.7136771
time: 57.73 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7084629, upper bound: 46.7375221
time: 61.76 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -41.7184830, 31.9957390, -41.7912369, 32.0602570, -73.7787399, 73.7869720
1: -27.5583630, 29.7112694, -27.5962410, 29.7742805, -57.3326416, 57.3075104
2: -22.0218849, 25.3931847, -22.0668259, 25.4666538, -47.4885406, 47.4600105
3: -25.3361149, 31.5977879, -25.3877106, 31.6843586, -57.0204735, 56.9854965
4: -28.9545059, 28.4955559, -29.0230255, 28.5891876, -57.5436935, 57.5185814
5: -28.2051392, 28.8807449, -28.2562065, 28.9602489, -57.1653900, 57.1369514
6: -47.1212578, 14.0829611, -47.1969147, 14.1378241, -60.8110657, 60.8776321
7: -38.4657211, 25.5571671, -38.5006180, 25.6219101, -64.0876312, 64.0577850
8: -35.0764847, 29.1150055, -35.1392975, 29.2494583, -64.3259430, 64.2543030
9: -21.9594135, 30.3910103, -21.9867306, 30.4305649, -52.3899765, 52.3777390
10: -41.8604889, 32.7991142, -41.8973770, 32.8325729, -74.6930618, 74.6964874
11: -48.4202995, 19.9181252, -48.4976425, 19.9567261, -68.3770294, 68.4157715
12: -43.2693634, 24.9527664, -43.3365822, 25.0121994, -68.2815628, 68.2893524
13: -30.4074860, 38.4846916, -30.4693985, 38.5784531, -68.9859390, 68.9540863
14: -77.2951660, 5.1602354, -77.3479156, 5.2192059, -82.5143738, 82.5081482
15: -29.4035130, 35.8945694, -29.4696827, 35.9790192, -65.3825302, 65.3642502
16: -46.6680336, 28.0654793, -46.7215958, 28.1005211, -74.7685547, 74.7870789
17: -79.1599274, 16.0155506, -79.2070847, 16.0581779, -95.2181091, 95.2226334
18: -44.8059387, 18.1394691, -44.9279709, 18.1994286, -63.0053673, 63.0674400
19: -36.1396103, 11.3569136, -36.2030029, 11.3837328, -47.5233421, 47.5599174
20: -30.9023933, 15.5829220, -30.9459743, 15.6269417, -46.5293350, 46.5288963
21: -43.4584923, 14.4649410, -43.5251732, 14.5020008, -57.9604950, 57.9901123
22: -37.5584373, 21.2807236, -37.6062393, 21.2919559, -58.8503952, 58.8869629
23: -32.1626053, 18.1531334, -32.2438583, 18.2047367, -50.3673401, 50.3969917
24: -30.6570740, 16.2225189, -30.7388268, 16.2507820, -46.9078560, 46.9613457
25: -26.7888145, 24.1365299, -26.8484783, 24.1616402, -50.9504547, 50.9850082
26: -49.8453255, 24.2812004, -49.9227943, 24.3322487, -74.1775742, 74.2039948
27: -41.4892235, 12.1153135, -41.5592728, 12.1579857, -53.6472092, 53.6745872
28: -32.5672188, 22.8463860, -32.6398163, 22.8991661, -55.4663849, 55.4862022
29: -37.7185364, 17.7544365, -37.7680473, 17.7677212, -55.4862595, 55.5224838
30: -40.8755150, 21.6397285, -40.9645615, 21.7002239, -62.5757370, 62.6042900
31: -41.5688095, 14.3157101, -41.6657677, 14.3425198, -55.9113312, 55.9814758
32: -36.8030319, 18.5016384, -36.8642578, 18.5496464, -55.3526764, 55.3658981
33: -46.6770287, 30.5948296, -46.7347794, 30.6254082, -77.3024368, 77.3296051
34: -44.3269348, 25.8011398, -44.4041214, 25.8521786, -70.1791153, 70.2052612
35: -35.7314987, 29.4843197, -35.7906151, 29.5095615, -65.2410583, 65.2749329
36: -39.5899582, 26.7647800, -39.6392288, 26.7905006, -66.3804626, 66.4040070
37: -60.2081833, 22.9559441, -60.3419609, 23.0120468, -83.2202301, 83.2979050
38: -56.5866432, 26.2726479, -56.6534157, 26.3160172, -82.9026642, 82.9260635
39: -56.7769928, 21.3863430, -56.8339577, 21.3986511, -78.1756439, 78.2202988
40: -58.5465126, 16.4778786, -58.6497955, 16.5317936, -74.4937592, 74.5558929
41: -38.3537216, 18.9670105, -38.4319572, 19.0170078, -57.3625336, 57.3989677
42: -31.8864765, 16.8310490, -31.9282169, 16.8832645, -48.7697411, 48.7592659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1790

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 607

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7503109, upper bound: 46.6926433
time: 67.45 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7503114, upper bound: 46.7164861
time: 63.63 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -41.8401108, 32.1515007, -41.8407555, 32.1496506, -73.9897614, 73.9922562
1: -27.6131344, 29.8604355, -27.6144066, 29.8589020, -57.4720383, 57.4748421
2: -22.0820847, 25.5666695, -22.0824203, 25.5659008, -47.6479874, 47.6490898
3: -25.3991184, 31.8048668, -25.4004326, 31.8028812, -57.2019997, 57.2052994
4: -29.0379028, 28.7176132, -29.0394344, 28.7162266, -57.7541275, 57.7570496
5: -28.2684879, 29.0683937, -28.2705879, 29.0678787, -57.3363647, 57.3389816
6: -47.2969322, 14.1600456, -47.2966805, 14.1598949, -61.0269775, 61.0490494
7: -38.5146141, 25.7100925, -38.5165482, 25.7094040, -64.2240143, 64.2266388
8: -35.1557159, 29.4351730, -35.1573029, 29.4333401, -64.5890579, 64.5924759
9: -22.0047112, 30.4802094, -22.0056782, 30.4806499, -52.4853592, 52.4858856
10: -41.9308319, 32.8605690, -41.9293442, 32.8618927, -74.7927246, 74.7899170
11: -48.6039581, 19.9820633, -48.6036415, 19.9808102, -68.5847702, 68.5857086
12: -43.4296799, 25.0357666, -43.4281311, 25.0370922, -68.4667740, 68.4638977
13: -30.4912682, 38.7020798, -30.4935684, 38.7011986, -69.1924667, 69.1956482
14: -77.3893890, 5.2864809, -77.3876038, 5.2875252, -82.6769104, 82.6740875
15: -29.4996738, 36.0945816, -29.5003796, 36.0925179, -65.5921936, 65.5949631
16: -46.7840157, 28.1163197, -46.7851448, 28.1173630, -74.9013824, 74.9014664
17: -79.2568970, 16.1101723, -79.2570419, 16.1056175, -95.3625183, 95.3672180
18: -45.0892220, 18.2111206, -45.0885925, 18.2127113, -63.3019333, 63.2997131
19: -36.2833023, 11.3881168, -36.2831955, 11.3894672, -47.6727676, 47.6713104
20: -30.9985485, 15.6487293, -30.9981251, 15.6497765, -46.6483231, 46.6468544
21: -43.6055145, 14.5116243, -43.6049881, 14.5133715, -58.1188850, 58.1166115
22: -37.6588974, 21.3064289, -37.6586685, 21.3050385, -58.9639359, 58.9650955
23: -32.3489342, 18.2198124, -32.3505630, 18.2223740, -50.5713081, 50.5703735
24: -30.8385487, 16.2593536, -30.8381996, 16.2609463, -47.0994949, 47.0975533
25: -26.9224186, 24.1758957, -26.9206371, 24.1775284, -51.0999451, 51.0965347
26: -50.0178413, 24.3542633, -50.0191002, 24.3550987, -74.3729401, 74.3733673
27: -41.6416702, 12.1744061, -41.6411705, 12.1755524, -53.8172226, 53.8155746
28: -32.7324524, 22.9176254, -32.7335777, 22.9197083, -55.6521606, 55.6512032
29: -37.8199501, 17.7806931, -37.8195496, 17.7795849, -55.5995331, 55.6002426
30: -41.0744247, 21.7267418, -41.0746040, 21.7274361, -62.8018608, 62.8013458
31: -41.7934875, 14.3520336, -41.7920074, 14.3529625, -56.1464500, 56.1440430
32: -36.9330902, 18.5746365, -36.9328537, 18.5753365, -55.5084267, 55.5074921
33: -46.8006439, 30.6381683, -46.8023224, 30.6384602, -77.4391022, 77.4404907
34: -44.5012283, 25.8673630, -44.5021973, 25.8675957, -70.3688202, 70.3695602
35: -35.8600006, 29.5198212, -35.8618050, 29.5200768, -65.3800812, 65.3816223
36: -39.6935959, 26.7990685, -39.6956100, 26.7993011, -66.4928970, 66.4946747
37: -60.5154037, 23.0255737, -60.5159988, 23.0256767, -83.5410767, 83.5415726
38: -56.7233047, 26.3358231, -56.7257767, 26.3359470, -83.0592499, 83.0615997
39: -56.8871956, 21.4133072, -56.8896599, 21.4133015, -78.3004990, 78.3029633
40: -58.7771072, 16.5522709, -58.7768898, 16.5518551, -74.7600861, 74.7813797
41: -38.5350876, 19.0342121, -38.5363159, 19.0346565, -57.5697441, 57.5705261
42: -31.9857330, 16.9081116, -31.9856148, 16.9082108, -48.8939438, 48.8937263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=327, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1790

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 607

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7586318, upper bound: 46.7347950
time: 58.50 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7586318, upper bound: 46.7586318
time: 161.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 222.76 seconds
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 222.76
Output dim: 13, lower bound: -46.6508076, upper bound: 46.7132234
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 222.76
Output dim: 13, lower bound: -46.6508076, upper bound: 46.7370654
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 222.76
Output dim: 13, lower bound: -46.6508076, upper bound: 46.6922959
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 222.76
Output dim: 13, lower bound: -46.7119205, upper bound: 46.7580294
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 222.76
Output dim: 13, lower bound: -46.7084613, upper bound: 46.7136771
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 222.76
Output dim: 13, lower bound: -46.7084629, upper bound: 46.7375221
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 222.76
Output dim: 13, lower bound: -46.7503109, upper bound: 46.6926433
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 222.76
Output dim: 13, lower bound: -46.7503114, upper bound: 46.7164861
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 222.76
Output dim: 13, lower bound: -46.7586318, upper bound: 46.7347950
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 222.76
Output dim: 13, lower bound: -46.7586318, upper bound: 46.7586318

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -41.6669922, 31.9606667, -41.6552277, 31.9344406, -73.6014328, 73.6158905
1: -27.5251560, 29.6914520, -27.5072136, 29.6484375, -57.1735916, 57.1986656
2: -21.9840965, 25.3856735, -21.9462299, 25.3644161, -47.3485107, 47.3319016
3: -25.2812119, 31.5572777, -25.2349529, 31.5489445, -56.8301544, 56.7922287
4: -28.8895454, 28.4806805, -28.8559570, 28.4685745, -57.3581200, 57.3366394
5: -28.1421623, 28.8365898, -28.1160507, 28.8709335, -57.0130959, 56.9526405
6: -47.0847931, 14.0518656, -47.0466957, 14.0748739, -60.7937012, 60.6717377
7: -38.4071503, 25.5087967, -38.3925934, 25.4928970, -63.9000473, 63.9013901
8: -35.0089989, 29.1205254, -34.9953995, 29.0816860, -64.0906830, 64.1159210
9: -21.8980312, 30.3775101, -21.9388752, 30.3682823, -52.2663116, 52.3163834
10: -41.8036079, 32.7730980, -41.8665123, 32.7918854, -74.5954895, 74.6396103
11: -48.4051666, 19.8311462, -48.4656258, 19.8627129, -68.2678833, 68.2967682
12: -43.2151566, 24.9381104, -43.1949539, 24.8925152, -68.1076736, 68.1330643
13: -30.2869053, 38.4524574, -30.3131237, 38.5258675, -68.8127747, 68.7655792
14: -77.2115097, 5.1521835, -77.3663483, 5.1417313, -82.3532410, 82.5185318
15: -29.3967896, 35.8710709, -29.3499680, 35.8318558, -65.2286453, 65.2210388
16: -46.6180000, 28.0296803, -46.6359825, 28.0413170, -74.6593170, 74.6656647
17: -79.1009674, 15.9506607, -79.1707840, 15.9334030, -95.0343704, 95.1214447
18: -44.8072891, 18.0673065, -44.8410721, 18.0530891, -62.8603783, 62.9083786
19: -36.0979004, 11.3000326, -36.1338577, 11.3080254, -47.4059258, 47.4338913
20: -30.8733673, 15.5565910, -30.8968754, 15.5312080, -46.4045753, 46.4534683
21: -43.4142532, 14.3877344, -43.4757004, 14.4070501, -57.8213043, 57.8634338
22: -37.5074539, 21.1849518, -37.5649261, 21.1927185, -58.7001724, 58.7498779
23: -32.1367188, 18.0579491, -32.2113113, 18.0943241, -50.2310410, 50.2692604
24: -30.6329727, 16.1444778, -30.6995049, 16.1706429, -46.8036156, 46.8439827
25: -26.7449570, 24.0345058, -26.7833824, 24.0639191, -50.8088760, 50.8178864
26: -49.8007126, 24.2094612, -49.9081192, 24.2171211, -74.0178375, 74.1175842
27: -41.4551849, 12.0482931, -41.5659943, 12.0679207, -53.5231056, 53.6142883
28: -32.5471306, 22.7736473, -32.6170349, 22.7885056, -55.3356361, 55.3906822
29: -37.6727180, 17.6187019, -37.7130814, 17.6366024, -55.3093185, 55.3317833
30: -40.8982010, 21.5923080, -40.9213181, 21.5946712, -62.4928741, 62.5136261
31: -41.5113602, 14.2398415, -41.5411301, 14.2475853, -55.7589455, 55.7809715
32: -36.7590256, 18.4843025, -36.7565231, 18.4819965, -55.2410202, 55.2408257
33: -46.5477829, 30.5429058, -46.5760269, 30.5909386, -77.1387177, 77.1189346
34: -44.3006973, 25.7951355, -44.2988815, 25.7720242, -70.0727234, 70.0940170
35: -35.6682663, 29.4478168, -35.6876221, 29.4986153, -65.1668854, 65.1354370
36: -39.5226250, 26.7416878, -39.5550308, 26.7855492, -66.3081741, 66.2967224
37: -60.1833725, 22.9499722, -60.1393471, 22.9365730, -83.1199493, 83.0893173
38: -56.4822845, 26.2597084, -56.4966812, 26.2457867, -82.7280731, 82.7563934
39: -56.5522575, 21.3353615, -56.6063271, 21.3708916, -77.9231491, 77.9416885
40: -58.4422379, 16.4563675, -58.4166489, 16.4454956, -74.3518677, 74.3027039
41: -38.3483200, 18.9751740, -38.3327446, 18.9744530, -57.3227730, 57.3019524
42: -31.8674355, 16.8320541, -31.8499260, 16.8100853, -48.6775208, 48.6819801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=326, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 606

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.6049392, upper bound: 46.7341393
time: 69.48 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.6049392, upper bound: 46.7341393
time: 60.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -41.7326279, 32.0806732, -41.8621407, 32.1416130, -73.8742371, 73.9428101
1: -27.5498447, 29.8086681, -27.6188984, 29.8512764, -57.4011230, 57.4275665
2: -22.0050430, 25.4998589, -22.0554695, 25.5654869, -47.5705299, 47.5553284
3: -25.2979698, 31.7093506, -25.3584404, 31.8156872, -57.1136551, 57.0677910
4: -28.9087143, 28.6218891, -28.9883862, 28.7179832, -57.6266975, 57.6102753
5: -28.1601353, 28.9552498, -28.2239361, 29.0813217, -57.2414551, 57.1791840
6: -47.2155304, 14.0759945, -47.2784653, 14.1993027, -61.0696716, 60.9450645
7: -38.4264526, 25.6258545, -38.4931107, 25.6990051, -64.1254578, 64.1189651
8: -35.0266266, 29.3182144, -35.1002922, 29.4265766, -64.4532013, 64.4185028
9: -21.9227390, 30.4399261, -22.0117836, 30.4798374, -52.4025764, 52.4517097
10: -41.8531799, 32.8012314, -41.9895630, 32.8609314, -74.7141113, 74.7907944
11: -48.4914932, 19.8570251, -48.6351929, 19.9252892, -68.4167786, 68.4922180
12: -43.3446465, 24.9628067, -43.4245758, 25.0201263, -68.3647766, 68.3873825
13: -30.3184681, 38.5747223, -30.4208260, 38.7441177, -69.0625839, 68.9955444
14: -77.2577515, 5.2062187, -77.4901199, 5.2528315, -82.5105820, 82.6963348
15: -29.4307899, 36.0163002, -29.5047512, 36.0891876, -65.5199738, 65.5210495
16: -46.6979637, 28.0476875, -46.7982979, 28.0972729, -74.7952347, 74.8459854
17: -79.1574554, 16.0386124, -79.3132935, 16.1026192, -95.2600708, 95.3519058
18: -44.9684944, 18.0872841, -45.1308136, 18.1544800, -63.1229744, 63.2180977
19: -36.1852608, 11.3074932, -36.2996902, 11.3504314, -47.5356903, 47.6071854
20: -30.9311962, 15.5868053, -31.0078964, 15.6318111, -46.5630074, 46.5947037
21: -43.4925728, 14.4022236, -43.6333008, 14.4580374, -57.9506111, 58.0355225
22: -37.5619965, 21.2238388, -37.6899376, 21.2666492, -58.8286438, 58.9137764
23: -32.2366638, 18.0774155, -32.3924904, 18.1535091, -50.3901749, 50.4699059
24: -30.7307243, 16.1576481, -30.8882866, 16.2109013, -46.9416275, 47.0459366
25: -26.8288784, 24.0535030, -26.9447136, 24.1241302, -50.9530106, 50.9982147
26: -49.8762131, 24.2417183, -50.0508652, 24.3113670, -74.1875763, 74.2925873
27: -41.5132294, 12.0715714, -41.6875992, 12.1273022, -53.6405334, 53.7591705
28: -32.6214600, 22.7966213, -32.7556915, 22.8682384, -55.4897003, 55.5523148
29: -37.7301865, 17.6627731, -37.8578262, 17.7228432, -55.4530296, 55.5205994
30: -40.9936600, 21.6259327, -41.1021461, 21.6900005, -62.6836624, 62.7280807
31: -41.6679573, 14.2529144, -41.8246803, 14.3098278, -55.9777832, 56.0775948
32: -36.8429718, 18.5092869, -36.9214325, 18.5699272, -55.4128990, 55.4307175
33: -46.6403809, 30.5582104, -46.7465744, 30.6687737, -77.3091583, 77.3047867
34: -44.3936195, 25.8156872, -44.4655800, 25.8703403, -70.2639618, 70.2812653
35: -35.7477570, 29.4628143, -35.8307495, 29.5610504, -65.3088074, 65.2935638
36: -39.5793076, 26.7538071, -39.6628571, 26.8380852, -66.4173889, 66.4166641
37: -60.3775978, 22.9648666, -60.4932213, 23.0469799, -83.4245758, 83.4580841
38: -56.5792198, 26.2837620, -56.6861382, 26.3482056, -82.9274292, 82.9699020
39: -56.6569519, 21.3505459, -56.8104591, 21.4367199, -78.0936737, 78.1610031
40: -58.6089745, 16.4799004, -58.7228508, 16.5686073, -74.6599808, 74.6496887
41: -38.4521599, 18.9917278, -38.5187798, 19.0438309, -57.4959908, 57.5105057
42: -31.9378796, 16.8569679, -31.9789162, 16.9030762, -48.8409576, 48.8358841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=326, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=356, inp2_unstable=358, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 606

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6049392, upper bound: 46.7132157
time: 69.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7090760, upper bound: 46.7551481
time: 101.49 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -41.7711487, 32.0199127, -41.7030182, 31.9451485, -73.7162933, 73.7229309
1: -27.5865364, 29.7328415, -27.5348263, 29.6583633, -57.2448997, 57.2676697
2: -22.0596771, 25.4468002, -21.9869175, 25.3740654, -47.4337425, 47.4337158
3: -25.3807068, 31.6480732, -25.2855415, 31.5610676, -56.9417725, 56.9336166
4: -29.0173206, 28.5690975, -28.9223213, 28.4786835, -57.4960022, 57.4914169
5: -28.2486382, 28.9480152, -28.1730099, 28.8877277, -57.1363678, 57.1210251
6: -47.1623306, 14.1329975, -47.0782013, 14.0954924, -60.8099518, 60.7811508
7: -38.4935608, 25.5819016, -38.4336700, 25.5066814, -64.0002441, 64.0155716
8: -35.1359444, 29.2248821, -35.0615730, 29.0975227, -64.2334671, 64.2864532
9: -21.9781990, 30.4144058, -21.9745235, 30.3793221, -52.3575211, 52.3889313
10: -41.8792496, 32.8277931, -41.8954010, 32.8099632, -74.6892090, 74.7231903
11: -48.5145073, 19.9523258, -48.4861450, 19.9238548, -68.4383621, 68.4384689
12: -43.2960587, 25.0066357, -43.2182617, 24.9196625, -68.2157211, 68.2248993
13: -30.4548645, 38.5773582, -30.4032993, 38.5381088, -68.9929733, 68.9806595
14: -77.3381424, 5.2280445, -77.3969116, 5.1782341, -82.5163727, 82.6249542
15: -29.4636669, 35.9423904, -29.3755646, 35.8449478, -65.3086166, 65.3179550
16: -46.7016830, 28.0944519, -46.6697922, 28.0734024, -74.7750854, 74.7642441
17: -79.1972656, 16.0190182, -79.1952896, 15.9528923, -95.1501617, 95.2143097
18: -44.9260330, 18.1882687, -44.8561859, 18.1166744, -63.0427094, 63.0444565
19: -36.1936874, 11.3797970, -36.1456223, 11.3519306, -47.5456161, 47.5254211
20: -30.9369564, 15.6173725, -30.9077415, 15.5598669, -46.4968224, 46.5251160
21: -43.5237694, 14.4942884, -43.4898987, 14.4652166, -57.9889870, 57.9841881
22: -37.6019669, 21.2657814, -37.5830231, 21.2353592, -58.8373260, 58.8488045
23: -32.2462845, 18.1981220, -32.2231102, 18.1700211, -50.4163055, 50.4212341
24: -30.7373943, 16.2445755, -30.7148170, 16.2238007, -46.9611969, 46.9593925
25: -26.8356972, 24.1552734, -26.7963905, 24.1269684, -50.9626656, 50.9516640
26: -49.9390793, 24.3195667, -49.9310226, 24.2727509, -74.2118301, 74.2505875
27: -41.5813522, 12.1486340, -41.5802002, 12.1202202, -53.7015724, 53.7288361
28: -32.6559105, 22.8930073, -32.6279831, 22.8515701, -55.5074806, 55.5209885
29: -37.7598381, 17.7344990, -37.7308655, 17.6990299, -55.4588699, 55.4653625
30: -40.9756851, 21.6905918, -40.9336166, 21.6415272, -62.6172104, 62.6242065
31: -41.6338692, 14.3376999, -41.5545845, 14.2980156, -55.9318848, 55.8922844
32: -36.8463707, 18.5476189, -36.7923508, 18.4992199, -55.3455887, 55.3399696
33: -46.7023849, 30.6206036, -46.6478310, 30.6136398, -77.3160248, 77.2684326
34: -44.4002533, 25.8452034, -44.3417892, 25.7804756, -70.1807251, 70.1869965
35: -35.7759933, 29.5035286, -35.7353287, 29.5148525, -65.2908478, 65.2388611
36: -39.6327171, 26.7851677, -39.6033859, 26.7968788, -66.4295959, 66.3885498
37: -60.3166428, 23.0087242, -60.1945534, 22.9526100, -83.2692566, 83.2032776
38: -56.6210365, 26.3101063, -56.5524254, 26.2581787, -82.8792114, 82.8625336
39: -56.7760468, 21.3961658, -56.7172432, 21.3799438, -78.1559906, 78.1134109
40: -58.6069641, 16.5264778, -58.4939499, 16.4569016, -74.4726715, 74.4501266
41: -38.4287491, 19.0153122, -38.3653526, 18.9882698, -57.4083633, 57.3758049
42: -31.9130898, 16.8762913, -31.8651104, 16.8248978, -48.7379875, 48.7414017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=326, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 606

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.6625677, upper bound: 46.7346008
time: 60.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7055401, upper bound: 46.7346008
time: 56.17 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -41.7061958, 31.9772053, -41.7694931, 32.0297394, -73.7359314, 73.7466965
1: -27.5521832, 29.6948948, -27.5853233, 29.7467194, -57.2989044, 57.2802200
2: -22.0149536, 25.3857841, -22.0544319, 25.4535046, -47.4684601, 47.4402161
3: -25.3287010, 31.5895481, -25.3744812, 31.6697292, -56.9984283, 56.9640274
4: -28.9481583, 28.4860191, -29.0117779, 28.5724564, -57.5206146, 57.4977951
5: -28.1908875, 28.8731003, -28.2307816, 28.9467583, -57.1376457, 57.1038818
6: -47.0909348, 14.0762463, -47.1424637, 14.1259031, -60.7687988, 60.8161888
7: -38.4599037, 25.5463181, -38.4902573, 25.6029453, -64.0628510, 64.0365753
8: -35.0697403, 29.0950718, -35.1273384, 29.2142563, -64.2839966, 64.2224121
9: -21.9491005, 30.3660412, -21.9685650, 30.3871498, -52.3362503, 52.3346062
10: -41.8513184, 32.7564163, -41.8810654, 32.7557983, -74.6071167, 74.6374817
11: -48.4138412, 19.8932266, -48.4862823, 19.9119911, -68.3258362, 68.3795090
12: -43.2516823, 24.9433670, -43.3052444, 24.9957771, -68.2474594, 68.2486115
13: -30.3578873, 38.4781799, -30.3812218, 38.5668335, -68.9247208, 68.8594055
14: -77.2813416, 5.1242437, -77.3231049, 5.1550217, -82.4363632, 82.4473495
15: -29.3956242, 35.8815384, -29.4556522, 35.9565659, -65.3521881, 65.3371887
16: -46.6569481, 28.0345802, -46.7020264, 28.0452251, -74.7021713, 74.7366028
17: -79.1496429, 15.9973221, -79.1890564, 16.0253239, -95.1749649, 95.1863785
18: -44.8001862, 18.1132851, -44.9177933, 18.1527081, -62.9528961, 63.0310783
19: -36.1331558, 11.3523140, -36.1917877, 11.3755007, -47.5086555, 47.5441017
20: -30.8954811, 15.5767126, -30.9338512, 15.6158791, -46.5113602, 46.5105629
21: -43.4510880, 14.4538002, -43.5121765, 14.4820595, -57.9331474, 57.9659767
22: -37.5496597, 21.2709236, -37.5907364, 21.2743645, -58.8240242, 58.8616600
23: -32.1545410, 18.1334915, -32.2297211, 18.1696854, -50.3242264, 50.3632126
24: -30.6496410, 16.2073154, -30.7256031, 16.2234421, -46.8730850, 46.9329185
25: -26.7794266, 24.1246185, -26.8318634, 24.1402836, -50.9197083, 50.9564819
26: -49.8345795, 24.2619743, -49.9037552, 24.2976875, -74.1322632, 74.1657257
27: -41.4832497, 12.0943661, -41.5485954, 12.1205578, -53.6038055, 53.6429596
28: -32.5606537, 22.8340988, -32.6283035, 22.8769951, -55.4376488, 55.4624023
29: -37.7110138, 17.7397461, -37.7547417, 17.7415237, -55.4525375, 55.4944878
30: -40.8691711, 21.6231194, -40.9533691, 21.6707478, -62.5399170, 62.5764885
31: -41.5604019, 14.3090000, -41.6509781, 14.3306847, -55.8910866, 55.9599762
32: -36.7808456, 18.4966927, -36.8245850, 18.5408764, -55.3217239, 55.3212776
33: -46.6253357, 30.5870972, -46.6421967, 30.6115799, -77.2369156, 77.2292938
34: -44.3152351, 25.7939568, -44.3831787, 25.8393898, -70.1546249, 70.1771393
35: -35.6890335, 29.4777279, -35.7150040, 29.4977436, -65.1867752, 65.1927338
36: -39.5460129, 26.7615547, -39.5607147, 26.7848072, -66.3308182, 66.3222656
37: -60.1704903, 22.9506321, -60.2746773, 23.0025749, -83.1730652, 83.2253113
38: -56.5374908, 26.2672234, -56.5652962, 26.3064232, -82.8439178, 82.8325195
39: -56.7155800, 21.3821831, -56.7237892, 21.3912506, -78.1068268, 78.1059723
40: -58.5182266, 16.4731026, -58.5992737, 16.5234013, -74.4568024, 74.5002518
41: -38.3342056, 18.9614849, -38.3970909, 19.0071774, -57.3326416, 57.3585739
42: -31.8755341, 16.8241749, -31.9090939, 16.8710365, -48.7465706, 48.7332687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=326, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 606

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6049392, upper bound: 46.6892923
time: 58.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7473876, upper bound: 46.6892923
time: 62.58 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -41.7152367, 31.9842110, -41.8603897, 32.0628853, -73.7781219, 73.8446045
1: -27.5565376, 29.7009163, -27.6283340, 29.7765713, -57.3331070, 57.3292503
2: -22.0203743, 25.3875046, -22.0805168, 25.4758987, -47.4962730, 47.4680214
3: -25.3344154, 31.5931149, -25.3962898, 31.7092648, -57.0436783, 56.9894028
4: -28.9530449, 28.4881763, -29.0383034, 28.6010914, -57.5541382, 57.5264816
5: -28.2032299, 28.8790112, -28.2665501, 28.9904842, -57.1937141, 57.1455612
6: -47.1173553, 14.0801849, -47.2101746, 14.1979523, -60.8700256, 60.8831863
7: -38.4639435, 25.5459728, -38.5182457, 25.6252975, -64.0892410, 64.0642166
8: -35.0743256, 29.1023636, -35.1484680, 29.2585030, -64.3328247, 64.2508316
9: -21.9575844, 30.3876019, -22.0284672, 30.4407749, -52.3983612, 52.4160690
10: -41.8584671, 32.7944565, -41.9864693, 32.8496399, -74.7081070, 74.7809296
11: -48.4172096, 19.9142094, -48.5498085, 19.9623375, -68.3795471, 68.4640198
12: -43.2652435, 24.9483452, -43.3563614, 25.0224056, -68.2876511, 68.3047028
13: -30.4025707, 38.4822350, -30.4868374, 38.6335907, -69.0361633, 68.9690704
14: -77.2903061, 5.1557951, -77.4810410, 5.2210493, -82.5113525, 82.6368332
15: -29.4015579, 35.8876495, -29.4997158, 35.9888153, -65.3903732, 65.3873672
16: -46.6657028, 28.0616627, -46.7685356, 28.1125069, -74.7782135, 74.8302002
17: -79.1567993, 16.0123634, -79.2879562, 16.0745831, -95.2313843, 95.3003235
18: -44.8039513, 18.1366043, -44.9852791, 18.2047997, -63.0087509, 63.1218834
19: -36.1373749, 11.3560820, -36.2312851, 11.3886480, -47.5260239, 47.5873680
20: -30.8986340, 15.5817852, -30.9666023, 15.6375904, -46.5362244, 46.5483856
21: -43.4550362, 14.4621096, -43.5677071, 14.5048380, -57.9598732, 58.0298157
22: -37.5560532, 21.2789536, -37.6556206, 21.2961845, -58.8522377, 58.9345741
23: -32.1598969, 18.1508636, -32.2975922, 18.2115402, -50.3714371, 50.4484558
24: -30.6537266, 16.2209244, -30.8042870, 16.2538891, -46.9076157, 47.0252113
25: -26.7860756, 24.1348648, -26.8855839, 24.1712914, -50.9573669, 51.0204468
26: -49.8420753, 24.2787380, -49.9774704, 24.3441162, -74.1861877, 74.2562103
27: -41.4869385, 12.1128407, -41.6198807, 12.1620502, -53.6489868, 53.7327194
28: -32.5649986, 22.8447361, -32.6728745, 22.9107437, -55.4757423, 55.5176086
29: -37.7159462, 17.7522678, -37.8241310, 17.7733746, -55.4893188, 55.5764008
30: -40.8722420, 21.6371365, -41.0044518, 21.7096481, -62.5818901, 62.6415863
31: -41.5658646, 14.3144073, -41.7118607, 14.3498373, -55.9157028, 56.0262680
32: -36.8002625, 18.4996758, -36.8886108, 18.5614891, -55.3617516, 55.3882866
33: -46.6713257, 30.5926380, -46.7507401, 30.6783924, -77.3497162, 77.3433762
34: -44.3188171, 25.7995758, -44.4104118, 25.8633575, -70.1821747, 70.2099915
35: -35.7268982, 29.4830742, -35.8073120, 29.5668106, -65.2937088, 65.2903900
36: -39.5857315, 26.7630463, -39.6547241, 26.8406219, -66.4263535, 66.4177704
37: -60.2035675, 22.9541016, -60.3743973, 23.0493774, -83.2529449, 83.3284988
38: -56.5812340, 26.2710323, -56.6693649, 26.3406086, -82.9218445, 82.9403992
39: -56.7704849, 21.3843994, -56.8656273, 21.4311371, -78.2016220, 78.2500305
40: -58.5429916, 16.4756508, -58.6729965, 16.5600224, -74.5143433, 74.5717010
41: -38.3512039, 18.9647141, -38.4469948, 19.0400753, -57.3835754, 57.4117088
42: -31.8843231, 16.8242016, -31.9366894, 16.8929977, -48.7773209, 48.7608910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=326, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=358, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 606

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.7044233, upper bound: 46.6892923
time: 880.31 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7473876, upper bound: 46.7135620
time: 73.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -41.8278046, 32.1329918, -41.8189888, 32.1191711, -73.9469757, 73.9519806
1: -27.6069527, 29.8440933, -27.6035252, 29.8313713, -57.4383240, 57.4476166
2: -22.0751629, 25.5593109, -22.0700493, 25.5527649, -47.6279297, 47.6293602
3: -25.3917160, 31.7966137, -25.3872261, 31.7882481, -57.1799622, 57.1838379
4: -29.0315247, 28.7080765, -29.0281544, 28.6994972, -57.7310219, 57.7362289
5: -28.2542763, 29.0607948, -28.2451496, 29.0543747, -57.3086510, 57.3059464
6: -47.2666054, 14.1533279, -47.2422638, 14.1479597, -60.9846878, 60.9876251
7: -38.5087585, 25.6993122, -38.5061874, 25.6904716, -64.1992340, 64.2054977
8: -35.1489830, 29.4152622, -35.1452942, 29.3981571, -64.5471420, 64.5605545
9: -21.9944248, 30.4552727, -21.9875221, 30.4372368, -52.4316635, 52.4427948
10: -41.9216843, 32.8179283, -41.9130173, 32.7850838, -74.7067719, 74.7309418
11: -48.5974922, 19.9571667, -48.5922508, 19.9360943, -68.5335846, 68.5494156
12: -43.4119797, 25.0263672, -43.3968124, 25.0207176, -68.4326935, 68.4231796
13: -30.4416637, 38.6955299, -30.4053822, 38.6895599, -69.1312256, 69.1009140
14: -77.3755341, 5.2505207, -77.3627853, 5.2232742, -82.5988083, 82.6133041
15: -29.4917793, 36.0815506, -29.4863758, 36.0700417, -65.5618210, 65.5679245
16: -46.7728806, 28.0854263, -46.7655945, 28.0620728, -74.8349533, 74.8510208
17: -79.2466888, 16.0919857, -79.2390366, 16.0728798, -95.3195648, 95.3310242
18: -45.0835190, 18.1849289, -45.0784149, 18.1659393, -63.2494583, 63.2633438
19: -36.2768631, 11.3834972, -36.2719765, 11.3812084, -47.6580734, 47.6554718
20: -30.9916363, 15.6425028, -30.9860172, 15.6387424, -46.6303787, 46.6285210
21: -43.5980949, 14.5004702, -43.5919800, 14.4933996, -58.0914955, 58.0924492
22: -37.6501007, 21.2966232, -37.6431618, 21.2874584, -58.9375610, 58.9397850
23: -32.3408966, 18.2001419, -32.3364334, 18.1873150, -50.5282135, 50.5365753
24: -30.8310986, 16.2441292, -30.8249512, 16.2336121, -47.0647125, 47.0690804
25: -26.9130039, 24.1639843, -26.9040394, 24.1561680, -51.0691719, 51.0680237
26: -50.0071220, 24.3350372, -50.0000191, 24.3205795, -74.3276978, 74.3350525
27: -41.6356735, 12.1534433, -41.6304779, 12.1380911, -53.7737656, 53.7839203
28: -32.7258949, 22.9053230, -32.7220917, 22.8975410, -55.6234360, 55.6274147
29: -37.8124084, 17.7660332, -37.8062325, 17.7534065, -55.5658150, 55.5722656
30: -41.0680847, 21.7101650, -41.0634003, 21.6979408, -62.7660255, 62.7735672
31: -41.7850723, 14.3453054, -41.7772141, 14.3410892, -56.1261597, 56.1225204
32: -36.9108887, 18.5697117, -36.8932190, 18.5665417, -55.4774323, 55.4629288
33: -46.7489471, 30.6304588, -46.7097816, 30.6246414, -77.3735886, 77.3402405
34: -44.4895134, 25.8601303, -44.4812660, 25.8548012, -70.3443146, 70.3414001
35: -35.8175507, 29.5132217, -35.7862091, 29.5082378, -65.3257904, 65.2994308
36: -39.6496658, 26.7958717, -39.6171341, 26.7936459, -66.4433136, 66.4130096
37: -60.4777184, 23.0202408, -60.4487038, 23.0161743, -83.4938965, 83.4689484
38: -56.6741753, 26.3303871, -56.6376915, 26.3263817, -83.0005569, 82.9680786
39: -56.8257446, 21.4091434, -56.7795448, 21.4058475, -78.2315903, 78.1886902
40: -58.7488708, 16.5475254, -58.7264175, 16.5434837, -74.7231750, 74.7257690
41: -38.5155792, 19.0286980, -38.5014992, 19.0247917, -57.5403709, 57.5301971
42: -31.9747715, 16.9012146, -31.9665108, 16.8959942, -48.8707657, 48.8677254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=326, inp2_unstable=326, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 606

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7127482, upper bound: 46.7314386
time: 63.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7557238, upper bound: 46.7314386
time: 84.26 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 150.33 seconds
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.6049392, upper bound: 46.7341393
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.6049392, upper bound: 46.7341393
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.6049392, upper bound: 46.7132157
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.7090760, upper bound: 46.7551481
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.6625677, upper bound: 46.7346008
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.7055401, upper bound: 46.7346008
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.6049392, upper bound: 46.6892923
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.7473876, upper bound: 46.6892923
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.7044233, upper bound: 46.6892923
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.7473876, upper bound: 46.7135620
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.7127482, upper bound: 46.7314386
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 150.33
Output dim: 13, lower bound: -46.7557238, upper bound: 46.7314386
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 150.33
Output dim: 13, lower bound: -46.7586318, upper bound: 46.7586318

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 69.88 + 3575.64 = 3645.53 seconds
