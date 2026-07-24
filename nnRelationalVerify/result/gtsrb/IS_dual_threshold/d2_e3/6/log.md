## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 6)
Time budget: 3600 seconds
Split limit: 100
Threshold: 13.5190456218


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=237, inp2_unstable=237, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-29.1035843, 4.0819659, -29.1035843, 4.0819659, -26.8179855, 26.8179855)
1: (-10.2002726, 6.6784601, -10.2002726, 6.6784601, -13.5941849, 13.5941849)
2: (-14.3255644, 4.4790411, -14.3255644, 4.4790411, -14.3140030, 14.3140030)
3: (-21.0076447, 0.8029056, -21.0076447, 0.8029056, -19.6006126, 19.6006088)
4: (-22.1597424, 3.3729157, -22.1597424, 3.3729157, -19.5998955, 19.5998993)
5: (-20.5483322, 5.7820101, -20.5483322, 5.7820101, -23.1066132, 23.1066132)
6: (-22.5417976, 3.2172041, -22.5417976, 3.2172041, -21.2610626, 21.2610588)
7: (-21.4470673, 4.0145011, -21.4470673, 4.0145011, -21.1534576, 21.1534576)
8: (-34.1548691, -4.0730276, -34.1548691, -4.0730276, -20.8828506, 20.8828545)
9: (-12.3205452, 16.6536083, -12.3205452, 16.6536083, -26.4273911, 26.4273911)
10: (-6.4234352, 20.6932125, -6.4234352, 20.6932125, -23.7357330, 23.7357330)
11: (-6.9631867, 13.9954357, -6.9631867, 13.9954357, -18.5698509, 18.5698509)
12: (0.6847205, 35.3360062, 0.6847205, 35.3360062, -28.7007065, 28.7007065)
13: (-10.7516613, 24.3071651, -10.7516613, 24.3071651, -30.1250763, 30.1250763)
14: (-33.1121063, 10.5235691, -33.1121063, 10.5235691, -38.0085297, 38.0085297)
15: (-20.7350578, 0.3150561, -20.7350578, 0.3150561, -18.6258430, 18.6258430)
16: (-14.5553141, 7.4691143, -14.5553141, 7.4691143, -22.0244293, 22.0244293)
17: (-21.3317719, 18.8276081, -21.3317719, 18.8276081, -36.7759247, 36.7759323)
18: (-14.7242126, 9.5297651, -14.7242126, 9.5297651, -21.1603165, 21.1603127)
19: (-10.8561993, 6.8248925, -10.8561993, 6.8248925, -14.9675522, 14.9675522)
20: (-15.0861101, 4.9781151, -15.0861101, 4.9781151, -17.9665451, 17.9665375)
21: (-11.3853331, 9.6212339, -11.3853331, 9.6212339, -18.6507721, 18.6507721)
22: (-9.6240492, 7.8821964, -9.6240492, 7.8821964, -15.4127159, 15.4127159)
23: (-14.0681372, 7.3755002, -14.0681372, 7.3755002, -19.5985527, 19.5985489)
24: (-17.4821281, 6.2210522, -17.4821281, 6.2210522, -18.7441673, 18.7441711)
25: (-11.3766947, 10.1624947, -11.3766947, 10.1624947, -20.4965668, 20.4965706)
26: (-16.2205467, 9.8273706, -16.2205467, 9.8273706, -24.7661095, 24.7661133)
27: (-27.5573406, 0.8424377, -27.5573406, 0.8424377, -20.3572273, 20.3572235)
28: (-16.3368778, 7.3289914, -16.3368778, 7.3289914, -20.8372993, 20.8372993)
29: (-7.3822088, 10.4129000, -7.3822088, 10.4129000, -16.1900558, 16.1900597)
30: (-19.1822834, 7.4232187, -19.1822834, 7.4232187, -21.8333130, 21.8333130)
31: (-13.1223335, 9.5279808, -13.1223335, 9.5279808, -19.0579376, 19.0579338)
32: (-12.5482302, 9.3123322, -12.5482302, 9.3123322, -18.1630745, 18.1630707)
33: (-45.4771042, -9.3842297, -45.4771042, -9.3842297, -31.3682098, 31.3682175)
34: (-42.0075378, -14.0827579, -42.0075378, -14.0827579, -19.8698883, 19.8698921)
35: (-29.0599117, -2.4565258, -29.0599117, -2.4565258, -21.8463821, 21.8463821)
36: (-23.7434120, 3.7731578, -23.7434120, 3.7731578, -23.5975952, 23.5975914)
37: (-43.6823730, -4.6296034, -43.6823730, -4.6296034, -36.2220230, 36.2220230)
38: (-30.0186844, 1.3893299, -30.0186844, 1.3893299, -29.3143082, 29.3143082)
39: (-38.9617386, -3.9614925, -38.9617386, -3.9614925, -32.4811401, 32.4811478)
40: (-44.3805313, -12.4105377, -44.3805313, -12.4105377, -26.2817078, 26.2817078)
41: (-24.3003750, 5.0327868, -24.3003750, 5.0327868, -23.3723526, 23.3723526)
42: (-19.4863701, 2.2711205, -19.4863701, 2.2711205, -16.6306076, 16.6306076)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.53 + 47.11 = 49.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 12, lower bound: -13.5325782, upper bound: 13.5325782

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1689
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1462
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1789
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 979
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1492
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1689

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5270205, upper bound: 13.4992576
time: 36.18 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5270205, upper bound: 13.5270203
time: 24.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 60.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 60.72
Output dim: 12, lower bound: -13.5270205, upper bound: 13.4992576
IS_A2, status: Status.UNKNOWN, split count: 1, time: 60.72
Output dim: 12, lower bound: -13.5270205, upper bound: 13.5270203

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -29.0930195, 4.0647273, -29.0994663, 4.0751143, -26.8002167, 26.7956696
1: -10.1970205, 6.6673579, -10.1989918, 6.6740618, -13.5878906, 13.5826225
2: -14.3185463, 4.4691973, -14.3227577, 4.4752302, -14.3029366, 14.3015099
3: -20.9902191, 0.7945735, -21.0008736, 0.7995965, -19.5804977, 19.5851822
4: -22.1459446, 3.3632917, -22.1542091, 3.3691134, -19.5716095, 19.5819016
5: -20.5381508, 5.7700996, -20.5443039, 5.7773743, -23.0834732, 23.0849304
6: -22.5271721, 3.2136421, -22.5358753, 3.2158098, -21.2402573, 21.2480011
7: -21.4347343, 4.0009413, -21.4421730, 4.0092416, -21.1277313, 21.1298943
8: -34.1496773, -4.0856628, -34.1528664, -4.0779920, -20.8740768, 20.8702927
9: -12.3104849, 16.6399193, -12.3166113, 16.6482315, -26.3966904, 26.4037628
10: -6.4137688, 20.6781254, -6.4196529, 20.6872234, -23.7165756, 23.7169495
11: -6.9492145, 13.9898539, -6.9576731, 13.9932747, -18.5550919, 18.5573273
12: 0.6997776, 35.2869415, 0.6905327, 35.3166008, -28.6642609, 28.6396484
13: -10.7427044, 24.2756920, -10.7481604, 24.2947636, -30.1063690, 30.0965118
14: -33.0887413, 10.4007969, -33.1029091, 10.4756927, -37.9414215, 37.8796844
15: -20.7264290, 0.3071406, -20.7317123, 0.3119676, -18.6093369, 18.6115913
16: -14.5467339, 7.4597969, -14.5519543, 7.4654779, -22.0122108, 22.0117512
17: -21.3129921, 18.7184887, -21.3244133, 18.7849731, -36.7172394, 36.6600723
18: -14.6977320, 9.5213490, -14.7136803, 9.5264931, -21.1307106, 21.1378670
19: -10.8461113, 6.8206267, -10.8522415, 6.8232422, -14.9561348, 14.9591408
20: -15.0731783, 4.9714751, -15.0810280, 4.9755344, -17.9515419, 17.9547234
21: -11.3693619, 9.6155043, -11.3790083, 9.6189642, -18.6332626, 18.6358910
22: -9.6150703, 7.8766537, -9.6205788, 7.8800073, -15.4011364, 15.4025536
23: -14.0568752, 7.3682270, -14.0636616, 7.3726764, -19.5830688, 19.5813828
24: -17.4700718, 6.2168870, -17.4772854, 6.2194200, -18.7329025, 18.7344666
25: -11.3668528, 10.1535749, -11.3728123, 10.1589985, -20.4830208, 20.4813271
26: -16.2036514, 9.8245411, -16.2138729, 9.8262863, -24.7438164, 24.7493057
27: -27.5327988, 0.8367038, -27.5475998, 0.8402677, -20.3363495, 20.3431931
28: -16.3267727, 7.3226671, -16.3328781, 7.3265557, -20.8252182, 20.8254890
29: -7.3737936, 10.3902922, -7.3789158, 10.4039259, -16.1730728, 16.1641121
30: -19.1695442, 7.4139385, -19.1772327, 7.4195776, -21.8173752, 21.8180351
31: -13.1080475, 9.5233765, -13.1167259, 9.5261974, -19.0406685, 19.0464668
32: -12.5323792, 9.3051357, -12.5420570, 9.3095045, -18.1381340, 18.1483383
33: -45.4287796, -9.3926029, -45.4582214, -9.3875523, -31.3308716, 31.3446732
34: -41.9648209, -14.0882702, -41.9908791, -14.0848913, -19.8399239, 19.8531914
35: -29.0430794, -2.4589570, -29.0532761, -2.4574716, -21.8334312, 21.8359680
36: -23.7307549, 3.7683632, -23.7384300, 3.7712932, -23.5808411, 23.5842056
37: -43.6354065, -4.6352091, -43.6638641, -4.6317987, -36.1842194, 36.1945190
38: -30.0036030, 1.3859138, -30.0127354, 1.3879960, -29.2944489, 29.2903595
39: -38.9275703, -3.9669569, -38.9480934, -3.9636559, -32.4506378, 32.4620819
40: -44.3141785, -12.4165859, -44.3545761, -12.4129381, -26.2255402, 26.2528381
41: -24.2625599, 5.0286598, -24.2851505, 5.0311780, -23.3401260, 23.3550797
42: -19.4725952, 2.2648211, -19.4809189, 2.2686653, -16.6130791, 16.6169472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=237, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1589

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5246955, upper bound: 13.4697143
time: 24.04 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5265694, upper bound: 13.4988066
time: 28.24 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -29.1159344, 4.1058183, -29.1010990, 4.0798645, -26.8277817, 26.8455582
1: -10.2226114, 6.6971731, -10.1993694, 6.6765013, -13.6166267, 13.6238518
2: -14.3259583, 4.4947577, -14.3228302, 4.4783335, -14.3145409, 14.3313751
3: -21.0091515, 0.8480990, -21.0054436, 0.8021920, -19.6003571, 19.6445389
4: -22.1556873, 3.4076028, -22.1545868, 3.3717051, -19.5928307, 19.6498528
5: -20.5442810, 5.8128476, -20.5442295, 5.7811174, -23.0936050, 23.1487350
6: -22.5528851, 3.2450247, -22.5351658, 3.2168226, -21.2706070, 21.2944679
7: -21.4565773, 4.0469594, -21.4425316, 4.0136070, -21.1563950, 21.1956406
8: -34.1637955, -4.0519543, -34.1539993, -4.0741367, -20.8989563, 20.9077110
9: -12.3288097, 16.6925602, -12.3188314, 16.6521473, -26.4109650, 26.4935760
10: -6.4375362, 20.7253723, -6.4226484, 20.6911201, -23.7317734, 23.7891235
11: -7.0239573, 14.0042152, -6.9605742, 13.9948549, -18.6438560, 18.5736046
12: 0.5844698, 35.3206825, 0.6857805, 35.3260078, -28.8192749, 28.6817017
13: -10.8616114, 24.3137169, -10.7506714, 24.3013954, -30.2385788, 30.1361923
14: -33.2910004, 10.5157127, -33.1100121, 10.5147038, -38.1965332, 37.9847717
15: -20.7451286, 0.3739209, -20.7338657, 0.3133001, -18.6434479, 18.6809425
16: -14.5708399, 7.5462103, -14.5530891, 7.4681492, -22.0389900, 22.0993004
17: -21.5002594, 18.8185711, -21.3303051, 18.8175278, -36.9425354, 36.7578430
18: -14.7329855, 9.5468969, -14.7207718, 9.5289803, -21.1766052, 21.1639328
19: -10.8816442, 6.8369107, -10.8553200, 6.8245144, -14.9972687, 14.9800606
20: -15.1067038, 5.0002170, -15.0836029, 4.9776382, -17.9885406, 17.9923553
21: -11.4152117, 9.6275997, -11.3838463, 9.6203384, -18.6806030, 18.6673813
22: -9.6709299, 7.9114847, -9.6231575, 7.8799119, -15.4576931, 15.4451408
23: -14.0868721, 7.4005795, -14.0662775, 7.3749628, -19.6478310, 19.6054955
24: -17.4930286, 6.2444444, -17.4801216, 6.2204766, -18.7788467, 18.7562714
25: -11.4255066, 10.1836872, -11.3758583, 10.1588001, -20.5542679, 20.5130730
26: -16.2815971, 9.8372850, -16.2186661, 9.8267536, -24.8476067, 24.7585526
27: -27.5512009, 0.8897152, -27.5504570, 0.8420353, -20.3630753, 20.4089355
28: -16.3626938, 7.3511305, -16.3355103, 7.3284903, -20.8888588, 20.8491020
29: -7.4466095, 10.4207344, -7.3812332, 10.4108467, -16.2531357, 16.1931267
30: -19.2015781, 7.4465017, -19.1808376, 7.4221926, -21.8538017, 21.8548584
31: -13.1461830, 9.5478554, -13.1211624, 9.5273418, -19.0849419, 19.0792084
32: -12.5565710, 9.3412828, -12.5405436, 9.3117485, -18.1729355, 18.2088165
33: -45.5057449, -9.2795534, -45.4734879, -9.3866234, -31.4107132, 31.4747734
34: -42.0038910, -13.9870510, -42.0009842, -14.0838461, -19.8767624, 19.9681816
35: -29.0792007, -2.4262350, -29.0567513, -2.4581640, -21.9015350, 21.8523788
36: -23.7706795, 3.8118658, -23.7387238, 3.7727580, -23.6603928, 23.6013947
37: -43.7095184, -4.5761924, -43.6780777, -4.6302528, -36.2709579, 36.2235641
38: -30.0515347, 1.4034839, -30.0159416, 1.3889904, -29.3908386, 29.2824249
39: -38.9878311, -3.8882139, -38.9573898, -3.9636354, -32.5249252, 32.5290833
40: -44.3912125, -12.3172045, -44.3751755, -12.4112072, -26.2920685, 26.3773880
41: -24.3059139, 5.1107931, -24.2967472, 5.0324464, -23.3730621, 23.4475555
42: -19.4982185, 2.2938933, -19.4814301, 2.2704983, -16.6571732, 16.6556435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=237, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1589
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1589

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5246955, upper bound: 13.4974738
time: 20.03 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5265694, upper bound: 13.5265690
time: 28.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 50.32 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 50.32
Output dim: 12, lower bound: -13.5246955, upper bound: 13.4697143
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 50.32
Output dim: 12, lower bound: -13.5265694, upper bound: 13.4988066
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 50.32
Output dim: 12, lower bound: -13.5246955, upper bound: 13.4974738
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 50.32
Output dim: 12, lower bound: -13.5265694, upper bound: 13.5265690

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -29.0903130, 4.0357022, -29.0940399, 4.0174136, -26.7378159, 26.7602921
1: -10.1962862, 6.6538582, -10.1975441, 6.6474886, -13.5612144, 13.5680599
2: -14.3165607, 4.4477291, -14.3187857, 4.4329834, -14.2573090, 14.2753143
3: -20.9886150, 0.7881472, -20.9977226, 0.7869489, -19.5665627, 19.5742455
4: -22.1451759, 3.3425555, -22.1526337, 3.3285980, -19.5312576, 19.5544472
5: -20.5359039, 5.7644758, -20.5399208, 5.7662005, -23.0642471, 23.0670776
6: -22.4957008, 3.2123280, -22.4735088, 3.2131863, -21.2061615, 21.1842537
7: -21.4319229, 3.9867988, -21.4365864, 3.9811907, -21.0949097, 21.1036835
8: -34.1485367, -4.1101789, -34.1504822, -4.1263757, -20.8235359, 20.8428955
9: -12.3076115, 16.6282005, -12.3109293, 16.6248627, -26.3699646, 26.3859406
10: -6.4105444, 20.6716213, -6.4132643, 20.6743660, -23.6975479, 23.7022285
11: -6.9417729, 13.9885483, -6.9430122, 13.9906597, -18.5442009, 18.5397682
12: 0.7282381, 35.2826767, 0.7473230, 35.3082085, -28.6246338, 28.5814133
13: -10.7380409, 24.2641277, -10.7388582, 24.2718830, -30.0776749, 30.0747032
14: -33.0826492, 10.3782196, -33.0908165, 10.4318190, -37.8816223, 37.8398514
15: -20.7241325, 0.2924879, -20.7271423, 0.2832470, -18.5767403, 18.5903130
16: -14.5440702, 7.4519920, -14.5466671, 7.4499426, -21.9940128, 21.9986591
17: -21.3083534, 18.7032585, -21.3152084, 18.7550640, -36.6753540, 36.6315308
18: -14.6936321, 9.5184975, -14.7056465, 9.5207653, -21.1157150, 21.1178665
19: -10.8417921, 6.8134618, -10.8436966, 6.8093896, -14.9387169, 14.9430466
20: -15.0662556, 4.9701095, -15.0675087, 4.9728403, -17.9408264, 17.9384613
21: -11.3588848, 9.6138735, -11.3584843, 9.6157265, -18.6194115, 18.6137085
22: -9.6064453, 7.8740640, -9.6034489, 7.8748174, -15.3867016, 15.3828354
23: -14.0531645, 7.3601604, -14.0563183, 7.3565187, -19.5630913, 19.5658264
24: -17.4670639, 6.2086897, -17.4712315, 6.2032342, -18.7143593, 18.7197685
25: -11.3631620, 10.1498966, -11.3654890, 10.1516695, -20.4704132, 20.4699173
26: -16.1971283, 9.8173828, -16.2009315, 9.8122473, -24.7204514, 24.7243652
27: -27.5262661, 0.8325262, -27.5347977, 0.8324065, -20.3173332, 20.3165627
28: -16.3210068, 7.3195910, -16.3215027, 7.3203802, -20.8080444, 20.8084259
29: -7.3626237, 10.3891497, -7.3570309, 10.4016724, -16.1581535, 16.1420364
30: -19.1470451, 7.4116116, -19.1328411, 7.4149551, -21.7888603, 21.7703362
31: -13.1045914, 9.5194168, -13.1098576, 9.5183296, -19.0278931, 19.0341301
32: -12.5144558, 9.3039112, -12.5064116, 9.3070374, -18.1183624, 18.1123390
33: -45.4198990, -9.3975182, -45.4405212, -9.3971996, -31.3068390, 31.3178635
34: -41.9455795, -14.0907145, -41.9529610, -14.0897570, -19.8159637, 19.8131981
35: -29.0338707, -2.4606369, -29.0355644, -2.4608150, -21.8189163, 21.8159866
36: -23.7233829, 3.7667613, -23.7238846, 3.7681019, -23.5670967, 23.5648079
37: -43.6284256, -4.6398592, -43.6500092, -4.6410313, -36.1654205, 36.1738968
38: -29.9999237, 1.3792450, -30.0054359, 1.3752239, -29.2736511, 29.2702255
39: -38.9250374, -3.9789109, -38.9430962, -3.9857717, -32.4260330, 32.4460983
40: -44.3065643, -12.4195547, -44.3394318, -12.4187803, -26.2099152, 26.2332993
41: -24.2526150, 5.0267248, -24.2655983, 5.0273418, -23.3250275, 23.3311768
42: -19.4634991, 2.2628617, -19.4633522, 2.2647305, -16.5961189, 16.5966110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1492
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1733

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4770106, upper bound: 13.4660032
time: 37.89 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5209989, upper bound: 13.4660032
time: 24.67 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -29.0925713, 4.0623617, -29.1611786, 4.0749493, -26.7780533, 26.8578720
1: -10.1968966, 6.6659431, -10.2198658, 6.6758595, -13.5803223, 13.6031647
2: -14.3184052, 4.4678378, -14.3715534, 4.4752326, -14.2851219, 14.3487015
3: -20.9900131, 0.7938952, -21.0105591, 0.8008041, -19.5791893, 19.5902557
4: -22.1457863, 3.3602586, -22.1869526, 3.3687272, -19.5577087, 19.5995598
5: -20.5378113, 5.7683463, -20.5528355, 5.7781048, -23.0826645, 23.0821838
6: -22.5239410, 3.2135458, -22.5323849, 3.2569809, -21.2785950, 21.2231407
7: -21.4343224, 4.0000696, -21.4654312, 4.0107327, -21.1172333, 21.1336975
8: -34.1495590, -4.0877490, -34.1935997, -4.0782099, -20.8515778, 20.9108086
9: -12.3100815, 16.6385250, -12.3357086, 16.6507645, -26.3966446, 26.4210815
10: -6.4117393, 20.6769104, -6.4183478, 20.6931229, -23.7166824, 23.7184105
11: -6.9470906, 13.9896917, -6.9598370, 14.0004749, -18.5609589, 18.5591583
12: 0.7011652, 35.2865448, 0.6900301, 35.3648529, -28.6888199, 28.6127167
13: -10.7422018, 24.2740803, -10.7723446, 24.2960358, -30.1054688, 30.1206284
14: -33.0879631, 10.3991947, -33.1332893, 10.4796200, -37.9372864, 37.9156189
15: -20.7261715, 0.3051612, -20.7486572, 0.3146513, -18.6018028, 18.6281624
16: -14.5462990, 7.4589019, -14.5684919, 7.4714632, -22.0177612, 22.0273933
17: -21.3125114, 18.7163391, -21.3309994, 18.7833405, -36.7047882, 36.6705475
18: -14.6969261, 9.5199375, -14.7144661, 9.5301933, -21.1417656, 21.1261063
19: -10.8454771, 6.8197813, -10.8687973, 6.8224330, -14.9515877, 14.9745102
20: -15.0718470, 4.9712958, -15.0842266, 4.9778185, -17.9550018, 17.9572372
21: -11.3675652, 9.6153641, -11.3815117, 9.6354589, -18.6512566, 18.6358261
22: -9.6137543, 7.8761921, -9.6227112, 7.8994231, -15.4162369, 15.4019470
23: -14.0564194, 7.3674598, -14.0819769, 7.3735065, -19.5812683, 19.6023941
24: -17.4696598, 6.2160482, -17.4888000, 6.2250385, -18.7325592, 18.7420120
25: -11.3651581, 10.1530743, -11.3753595, 10.1746855, -20.4882736, 20.4856796
26: -16.2027493, 9.8233652, -16.2263546, 9.8326769, -24.7444267, 24.7525940
27: -27.5314465, 0.8355365, -27.5472794, 0.8478937, -20.3463135, 20.3297424
28: -16.3258514, 7.3214512, -16.3422279, 7.3262954, -20.8240814, 20.8421135
29: -7.3718143, 10.3901138, -7.3805432, 10.4220152, -16.1795235, 16.1629562
30: -19.1663914, 7.4136105, -19.1757736, 7.4704876, -21.8693695, 21.8015747
31: -13.1074657, 9.5213432, -13.1229420, 9.5258045, -19.0413742, 19.0523491
32: -12.5305061, 9.3049059, -12.5459652, 9.3310022, -18.1588440, 18.1387901
33: -45.4256516, -9.3935719, -45.4675369, -9.3774176, -31.3321533, 31.3492355
34: -41.9635887, -14.0885534, -41.9922905, -14.0489149, -19.8722229, 19.8370171
35: -29.0415897, -2.4592900, -29.0610352, -2.4488733, -21.8350525, 21.8404884
36: -23.7293720, 3.7681730, -23.7503624, 3.7731884, -23.5808868, 23.5939713
37: -43.6335487, -4.6359487, -43.6718445, -4.6259341, -36.1878662, 36.2018051
38: -30.0029430, 1.3839583, -30.0331039, 1.3855805, -29.2931137, 29.2972183
39: -38.9271812, -3.9703348, -38.9705162, -3.9674568, -32.4431915, 32.4802933
40: -44.3114395, -12.4170866, -44.3595543, -12.4078445, -26.2275543, 26.2549057
41: -24.2601967, 5.0283413, -24.2924728, 5.0356059, -23.3429565, 23.3567810
42: -19.4683933, 2.2644587, -19.4771633, 2.2746410, -16.6069527, 16.6193504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1733

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4788849, upper bound: 13.4950941
time: 31.83 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5228725, upper bound: 13.4950941
time: 25.19 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -29.1132050, 4.0769148, -29.0956917, 4.0221739, -26.7653656, 26.8101349
1: -10.2218962, 6.6836576, -10.1979399, 6.6499462, -13.5899544, 13.6092968
2: -14.3239775, 4.4732771, -14.3188696, 4.4361100, -14.2689018, 14.3051872
3: -21.0075550, 0.8416848, -21.0022755, 0.7895193, -19.5864220, 19.6335983
4: -22.1548805, 3.3868504, -22.1530056, 3.3311195, -19.5524597, 19.6224174
5: -20.5420456, 5.8071842, -20.5398064, 5.7699594, -23.0744247, 23.1308289
6: -22.5214748, 3.2437477, -22.4728107, 3.2141800, -21.2365112, 21.2307777
7: -21.4537678, 4.0327916, -21.4369011, 3.9854941, -21.1235352, 21.1694336
8: -34.1626320, -4.0764418, -34.1516876, -4.1225657, -20.8484116, 20.8802948
9: -12.3259525, 16.6808872, -12.3131552, 16.6287994, -26.3842087, 26.4757614
10: -6.4343085, 20.7188568, -6.4162588, 20.6782875, -23.7126770, 23.7744141
11: -7.0165157, 14.0028934, -6.9459124, 13.9922333, -18.6329231, 18.5560570
12: 0.6128707, 35.3163681, 0.7425394, 35.3175583, -28.7796173, 28.6234589
13: -10.8568993, 24.3021374, -10.7413454, 24.2785568, -30.2098312, 30.1143761
14: -33.2849388, 10.4931707, -33.0979004, 10.4708538, -38.1368103, 37.9449387
15: -20.7428112, 0.3592303, -20.7292957, 0.2845683, -18.6108360, 18.6596718
16: -14.5681763, 7.5384145, -14.5478096, 7.4526234, -22.0207996, 22.0862236
17: -21.4956722, 18.8033333, -21.3210926, 18.7875977, -36.9007568, 36.7293015
18: -14.7288761, 9.5440178, -14.7127419, 9.5232620, -21.1616325, 21.1439133
19: -10.8773003, 6.8297534, -10.8467617, 6.8106689, -14.9798584, 14.9639435
20: -15.0997829, 4.9988480, -15.0700922, 4.9749489, -17.9778252, 17.9761238
21: -11.4047413, 9.6259642, -11.3633022, 9.6170826, -18.6667404, 18.6452446
22: -9.6623230, 7.9088707, -9.6060333, 7.8747182, -15.4432507, 15.4254074
23: -14.0831518, 7.3925018, -14.0589285, 7.3587980, -19.6278763, 19.5898857
24: -17.4899750, 6.2362018, -17.4741344, 6.2042685, -18.7602654, 18.7415314
25: -11.4218216, 10.1800327, -11.3685236, 10.1514444, -20.5416451, 20.5016365
26: -16.2750950, 9.8301392, -16.2057419, 9.8127003, -24.8242035, 24.7336655
27: -27.5446987, 0.8855066, -27.5376472, 0.8341885, -20.3440323, 20.3823166
28: -16.3569031, 7.3480444, -16.3241158, 7.3223133, -20.8717041, 20.8320274
29: -7.4354315, 10.4195995, -7.3593407, 10.4085989, -16.2382317, 16.1710548
30: -19.1791000, 7.4441853, -19.1364250, 7.4175797, -21.8252563, 21.8071785
31: -13.1426983, 9.5438824, -13.1142979, 9.5194492, -19.0721664, 19.0668793
32: -12.5386505, 9.3400421, -12.5048733, 9.3092833, -18.1531792, 18.1728020
33: -45.4968643, -9.2844353, -45.4557838, -9.3963194, -31.3866882, 31.4479294
34: -41.9846878, -13.9894896, -41.9630661, -14.0887117, -19.8528061, 19.9281998
35: -29.0699902, -2.4278872, -29.0390453, -2.4614685, -21.8870468, 21.8323364
36: -23.7633038, 3.8102674, -23.7241974, 3.7696168, -23.6466141, 23.5820160
37: -43.7025566, -4.5808411, -43.6641960, -4.6395216, -36.2521362, 36.2029419
38: -30.0478325, 1.3968134, -30.0086021, 1.3762641, -29.3700409, 29.2622452
39: -38.9852829, -3.9001906, -38.9523277, -3.9857194, -32.5002975, 32.5130615
40: -44.3836441, -12.3201103, -44.3600388, -12.4170427, -26.2764587, 26.3578529
41: -24.2959385, 5.1088047, -24.2772217, 5.0285826, -23.3579483, 23.4237061
42: -19.4891376, 2.2919226, -19.4638367, 2.2665625, -16.6402130, 16.6353149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1733

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4770106, upper bound: 13.4937805
time: 35.21 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5209989, upper bound: 13.4937805
time: 29.72 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -29.1155128, 4.1035037, -29.1628323, 4.0796785, -26.8055801, 26.9077072
1: -10.2224922, 6.6957469, -10.2202415, 6.6783390, -13.6090546, 13.6444054
2: -14.3258572, 4.4933801, -14.3716288, 4.4783525, -14.2967377, 14.3785667
3: -21.0089016, 0.8474455, -21.0151291, 0.8033612, -19.5990639, 19.6495895
4: -22.1555309, 3.4045320, -22.1873455, 3.3712707, -19.5789490, 19.6675377
5: -20.5439777, 5.8110800, -20.5527077, 5.7818966, -23.0928040, 23.1459961
6: -22.5497475, 3.2448978, -22.5317307, 3.2579556, -21.3089142, 21.2695923
7: -21.4561920, 4.0461049, -21.4657955, 4.0151014, -21.1458969, 21.1994476
8: -34.1636810, -4.0540552, -34.1947556, -4.0742955, -20.8764343, 20.9481735
9: -12.3284388, 16.6912098, -12.3379059, 16.6546936, -26.4108810, 26.5108795
10: -6.4355173, 20.7242069, -6.4213314, 20.6970520, -23.7318192, 23.7905884
11: -7.0218210, 14.0040512, -6.9627352, 14.0020466, -18.6497154, 18.5754547
12: 0.5858564, 35.3202286, 0.6851993, 35.3742905, -28.8438110, 28.6547546
13: -10.8610773, 24.3120956, -10.7748642, 24.3026619, -30.2377090, 30.1603279
14: -33.2902451, 10.5141668, -33.1403656, 10.5186348, -38.1924744, 38.0206757
15: -20.7448616, 0.3719232, -20.7508354, 0.3159850, -18.6358948, 18.6975021
16: -14.5704012, 7.5453315, -14.5696182, 7.4741387, -22.0445404, 22.1149502
17: -21.4997349, 18.8164558, -21.3369255, 18.8159389, -36.9301147, 36.7683334
18: -14.7321653, 9.5454483, -14.7215319, 9.5326900, -21.1876793, 21.1521797
19: -10.8809834, 6.8360701, -10.8718681, 6.8236809, -14.9927330, 14.9954033
20: -15.1053791, 5.0000129, -15.0868092, 4.9799128, -17.9919968, 17.9948883
21: -11.4133930, 9.6274662, -11.3863487, 9.6368027, -18.6986008, 18.6673355
22: -9.6695957, 7.9110312, -9.6253195, 7.8993273, -15.4727898, 15.4445229
23: -14.0863876, 7.3998365, -14.0846062, 7.3757429, -19.6460571, 19.6264534
24: -17.4926319, 6.2435923, -17.4916573, 6.2260580, -18.7784882, 18.7637939
25: -11.4238110, 10.1831970, -11.3784161, 10.1744690, -20.5594940, 20.5173836
26: -16.2806892, 9.8361206, -16.2311745, 9.8331623, -24.8481827, 24.7618866
27: -27.5498695, 0.8885164, -27.5501480, 0.8497143, -20.3730316, 20.3954849
28: -16.3617668, 7.3499212, -16.3448257, 7.3282118, -20.8877068, 20.8657227
29: -7.4446135, 10.4205513, -7.3828464, 10.4289627, -16.2596130, 16.1919785
30: -19.1984692, 7.4461899, -19.1793537, 7.4731112, -21.9058075, 21.8384056
31: -13.1455765, 9.5458031, -13.1273985, 9.5269241, -19.0856705, 19.0850716
32: -12.5546923, 9.3410368, -12.5444403, 9.3332653, -18.1936569, 18.1992722
33: -45.5026627, -9.2804747, -45.4827576, -9.3765745, -31.4120026, 31.4792938
34: -42.0026932, -13.9873314, -42.0024452, -14.0478611, -19.9090691, 19.9519958
35: -29.0777245, -2.4265668, -29.0645065, -2.4495444, -21.9031982, 21.8568993
36: -23.7692928, 3.8116915, -23.7506485, 3.7747073, -23.6604576, 23.6111641
37: -43.7076263, -4.5769444, -43.6860580, -4.6244011, -36.2746048, 36.2307739
38: -30.0508823, 1.4015145, -30.0362568, 1.3866355, -29.3895340, 29.2893066
39: -38.9874611, -3.8915691, -38.9797668, -3.9674003, -32.5174408, 32.5472717
40: -44.3884773, -12.3176289, -44.3801918, -12.4060860, -26.2941208, 26.3794632
41: -24.3035183, 5.1104760, -24.3040257, 5.0368662, -23.3758774, 23.4492950
42: -19.4940281, 2.2935028, -19.4776764, 2.2764587, -16.6510277, 16.6580620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=236, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1733
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1492
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1733

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4788849, upper bound: 13.5228724
time: 46.29 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5228725, upper bound: 13.5228724
time: 26.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 75.03 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 75.03
Output dim: 12, lower bound: -13.4770106, upper bound: 13.4660032
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 75.03
Output dim: 12, lower bound: -13.5209989, upper bound: 13.4660032
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 75.03
Output dim: 12, lower bound: -13.4788849, upper bound: 13.4950941
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 75.03
Output dim: 12, lower bound: -13.5228725, upper bound: 13.4950941
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 75.03
Output dim: 12, lower bound: -13.4770106, upper bound: 13.4937805
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 75.03
Output dim: 12, lower bound: -13.5209989, upper bound: 13.4937805
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 75.03
Output dim: 12, lower bound: -13.4788849, upper bound: 13.5228724
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 75.03
Output dim: 12, lower bound: -13.5228725, upper bound: 13.5228724

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -29.1019402, 4.0459771, -29.0930977, 4.0167594, -26.7471008, 26.7733459
1: -10.1984634, 6.6640019, -10.1965237, 6.6471415, -13.5605202, 13.5804691
2: -14.3272848, 4.4533920, -14.3185053, 4.4323206, -14.2697182, 14.2823906
3: -21.0131035, 0.7960348, -20.9974174, 0.7851796, -19.5974083, 19.5781555
4: -22.1551743, 3.3461080, -22.1522369, 3.3277526, -19.5290909, 19.5831337
5: -20.5768356, 5.7745376, -20.5396709, 5.7638006, -23.1091385, 23.0729599
6: -22.4983463, 3.2596235, -22.4701271, 3.2128906, -21.2059326, 21.2247238
7: -21.4642124, 3.9885509, -21.4359856, 3.9789028, -21.1258545, 21.1029968
8: -34.1629829, -4.1125040, -34.1501770, -4.1291213, -20.8317947, 20.8476028
9: -12.3141384, 16.6691132, -12.3081245, 16.6244164, -26.3767319, 26.4279022
10: -6.4069338, 20.7129002, -6.4103613, 20.6737347, -23.6969681, 23.7319717
11: -6.9843502, 13.9890947, -6.9421611, 13.9887943, -18.5907478, 18.5339661
12: 0.7285986, 35.3928185, 0.7513986, 35.3078918, -28.6166000, 28.6893921
13: -10.7398024, 24.3256569, -10.7363129, 24.2715855, -30.0737305, 30.1323357
14: -33.1395836, 10.3770142, -33.0901108, 10.4270277, -37.9310150, 37.8408203
15: -20.7452965, 0.2949266, -20.7266960, 0.2818706, -18.5945816, 18.5879478
16: -14.5471745, 7.4713435, -14.5441360, 7.4494085, -21.9965820, 22.0154800
17: -21.3149681, 18.7297497, -21.3137665, 18.7546177, -36.6730957, 36.6614761
18: -14.7384434, 9.5200205, -14.7051182, 9.5185394, -21.1517181, 21.1202354
19: -10.8799629, 6.8115039, -10.8434153, 6.8080883, -14.9790459, 14.9382744
20: -15.1193895, 4.9702368, -15.0670681, 4.9707918, -17.9943771, 17.9349976
21: -11.4071236, 9.6119003, -11.3581200, 9.6130486, -18.6672935, 18.6068153
22: -9.6445923, 7.8742094, -9.6029968, 7.8724575, -15.4241142, 15.3786049
23: -14.1056747, 7.3610201, -14.0560532, 7.3546143, -19.6160812, 19.5623932
24: -17.5330925, 6.2084489, -17.4710617, 6.2007017, -18.7788315, 18.7121582
25: -11.4222612, 10.1530027, -11.3652992, 10.1488457, -20.5306320, 20.4687805
26: -16.2398720, 9.8184299, -16.2002544, 9.8107605, -24.7405701, 24.7287979
27: -27.6083527, 0.8301015, -27.5346165, 0.8296795, -20.3985100, 20.3053551
28: -16.3808918, 7.3181877, -16.3211212, 7.3183851, -20.8722000, 20.8030853
29: -7.4037108, 10.3893442, -7.3567257, 10.3999004, -16.2025719, 16.1404877
30: -19.2134171, 7.4118271, -19.1324196, 7.4120646, -21.8582726, 21.7614441
31: -13.1551113, 9.5181217, -13.1094723, 9.5162621, -19.0809860, 19.0291862
32: -12.5212936, 9.3643532, -12.5042267, 9.3068495, -18.1189766, 18.1716919
33: -45.4298401, -9.3414688, -45.4381104, -9.3979492, -31.3094025, 31.3809204
34: -41.9543686, -14.0872526, -41.9528503, -14.0900507, -19.8269424, 19.8200569
35: -29.0432625, -2.4527702, -29.0343513, -2.4610975, -21.8206406, 21.8364067
36: -23.7344971, 3.7777567, -23.7226143, 3.7679431, -23.5785370, 23.5766144
37: -43.6426086, -4.5757232, -43.6477203, -4.6414886, -36.1697006, 36.2443008
38: -30.0058899, 1.3969505, -30.0042648, 1.3750267, -29.2849121, 29.2891464
39: -38.9318161, -3.9139023, -38.9398270, -3.9862323, -32.4252853, 32.5191727
40: -44.3161240, -12.3635330, -44.3369751, -12.4189205, -26.2146454, 26.2888603
41: -24.2608032, 5.0892644, -24.2621727, 5.0271354, -23.3287582, 23.3962784
42: -19.4640732, 2.3172083, -19.4610214, 2.2643247, -16.5895233, 16.6498108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1492
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1605

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5200601, upper bound: 13.4418152
time: 35.25 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5200680, upper bound: 13.4643937
time: 29.50 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -29.1042480, 4.0725951, -29.1602631, 4.0742602, -26.7873077, 26.8709412
1: -10.1990805, 6.6760645, -10.2188444, 6.6755424, -13.5796127, 13.6155853
2: -14.3291340, 4.4735270, -14.3712816, 4.4745827, -14.2975426, 14.3557968
3: -21.0144691, 0.8017836, -21.0102673, 0.7990348, -19.6099777, 19.5941429
4: -22.1558056, 3.3637800, -22.1865730, 3.3678803, -19.5555458, 19.6282272
5: -20.5787334, 5.7784128, -20.5525665, 5.7756853, -23.1274872, 23.0881042
6: -22.5265694, 3.2608318, -22.5289879, 3.2566929, -21.2783432, 21.2636070
7: -21.4666271, 4.0018339, -21.4648247, 4.0085063, -21.1482162, 21.1329575
8: -34.1640205, -4.0900764, -34.1932602, -4.0809250, -20.8598251, 20.9154892
9: -12.3166180, 16.6794205, -12.3329048, 16.6502838, -26.4034348, 26.4630203
10: -6.4081192, 20.7182007, -6.4154530, 20.6924763, -23.7160873, 23.7481232
11: -6.9896250, 13.9902296, -6.9589515, 13.9986382, -18.6075249, 18.5533409
12: 0.7015438, 35.3966866, 0.6941056, 35.3646622, -28.6808167, 28.7206879
13: -10.7439137, 24.3356171, -10.7698135, 24.2957268, -30.1015472, 30.1782761
14: -33.1448669, 10.3979492, -33.1325760, 10.4748287, -37.9866638, 37.9166107
15: -20.7472916, 0.3075809, -20.7482071, 0.3132563, -18.6196365, 18.6257858
16: -14.5494118, 7.4782381, -14.5659714, 7.4709358, -22.0203476, 22.0442085
17: -21.3191109, 18.7428894, -21.3296013, 18.7829723, -36.7025452, 36.7005157
18: -14.7417316, 9.5214911, -14.7139492, 9.5279865, -21.1777458, 21.1285210
19: -10.8836508, 6.8178349, -10.8685226, 6.8211203, -14.9919052, 14.9697342
20: -15.1249475, 4.9713850, -15.0837898, 4.9757910, -18.0085526, 17.9537849
21: -11.4157963, 9.6133957, -11.3811741, 9.6327581, -18.6991348, 18.6289177
22: -9.6518879, 7.8763437, -9.6222744, 7.8970661, -15.4536438, 15.3977394
23: -14.1089153, 7.3683519, -14.0817146, 7.3715911, -19.6342697, 19.5989647
24: -17.5357285, 6.2158999, -17.4886360, 6.2225351, -18.7970428, 18.7344017
25: -11.4242287, 10.1561766, -11.3751755, 10.1718674, -20.5484962, 20.4845352
26: -16.2455063, 9.8244381, -16.2256470, 9.8311882, -24.7645721, 24.7570419
27: -27.6135368, 0.8331280, -27.5471039, 0.8452115, -20.4274826, 20.3185425
28: -16.3857803, 7.3200760, -16.3418217, 7.3242927, -20.8882484, 20.8367996
29: -7.4128723, 10.3903027, -7.3802295, 10.4202518, -16.2239380, 16.1614304
30: -19.2327652, 7.4138393, -19.1753521, 7.4676037, -21.9387970, 21.7926826
31: -13.1579800, 9.5200663, -13.1225758, 9.5237217, -19.0944748, 19.0474129
32: -12.5373421, 9.3653698, -12.5437851, 9.3308296, -18.1594620, 18.1981468
33: -45.4356003, -9.3375492, -45.4650803, -9.3781776, -31.3346939, 31.4122505
34: -41.9724045, -14.0851107, -41.9921494, -14.0491943, -19.8832207, 19.8438873
35: -29.0509415, -2.4514291, -29.0598526, -2.4491808, -21.8367958, 21.8609467
36: -23.7405052, 3.7791710, -23.7490997, 3.7730660, -23.5923195, 23.6058083
37: -43.6477585, -4.5718045, -43.6695213, -4.6263924, -36.1921463, 36.2722168
38: -30.0089455, 1.4016685, -30.0319862, 1.3853676, -29.3043442, 29.3161354
39: -38.9339371, -3.9052718, -38.9672508, -3.9679475, -32.4424362, 32.5534210
40: -44.3209839, -12.3611078, -44.3571358, -12.4079695, -26.2322693, 26.3104782
41: -24.2683506, 5.0909028, -24.2890053, 5.0353851, -23.3466644, 23.4218521
42: -19.4689522, 2.3188100, -19.4748516, 2.2742391, -16.6003456, 16.6725502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1492
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1605

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5219586, upper bound: 13.4628667
time: 36.17 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5220219, upper bound: 13.4942441
time: 28.63 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -29.1248512, 4.0871291, -29.0947628, 4.0215087, -26.7746048, 26.8231964
1: -10.2240486, 6.6938062, -10.1969194, 6.6496429, -13.5892334, 13.6216736
2: -14.3347178, 4.4789410, -14.3185987, 4.4354630, -14.2813339, 14.3122749
3: -21.0320206, 0.8495438, -21.0019779, 0.7877662, -19.6172485, 19.6375160
4: -22.1649170, 3.3904057, -22.1526489, 3.3303041, -19.5502853, 19.6510811
5: -20.5829620, 5.8172765, -20.5395393, 5.7675657, -23.1192551, 23.1367798
6: -22.5240936, 3.2910213, -22.4694557, 3.2138939, -21.2362900, 21.2712593
7: -21.4860725, 4.0345464, -21.4363251, 3.9832313, -21.1544724, 21.1687546
8: -34.1770782, -4.0787845, -34.1513290, -4.1252508, -20.8566284, 20.8850403
9: -12.3324738, 16.7217941, -12.3103285, 16.6283131, -26.3909531, 26.5177155
10: -6.4307070, 20.7601433, -6.4133654, 20.6776314, -23.7121048, 23.8041840
11: -7.0590096, 14.0034437, -6.9450769, 13.9904041, -18.6794319, 18.5502586
12: 0.6132817, 35.4265022, 0.7466154, 35.3173523, -28.7715759, 28.7314301
13: -10.8586636, 24.3636475, -10.7388391, 24.2782688, -30.2058182, 30.1719894
14: -33.3418274, 10.4919243, -33.0972328, 10.4660788, -38.1862335, 37.9458771
15: -20.7639046, 0.3616841, -20.7288532, 0.2831929, -18.6286354, 18.6573067
16: -14.5712996, 7.5577526, -14.5452824, 7.4520850, -22.0233841, 22.1030350
17: -21.5022163, 18.8298378, -21.3196907, 18.7872028, -36.8984833, 36.7592392
18: -14.7736778, 9.5455875, -14.7121773, 9.5210438, -21.1976585, 21.1462822
19: -10.9154644, 6.8277922, -10.8464661, 6.8093567, -15.0202217, 14.9591637
20: -15.1529026, 4.9989476, -15.0696726, 4.9729214, -18.0313492, 17.9726524
21: -11.4529991, 9.6240139, -11.3629608, 9.6143665, -18.7146683, 18.6383553
22: -9.7004471, 7.9090471, -9.6055927, 7.8723488, -15.4806709, 15.4211941
23: -14.1356506, 7.3933930, -14.0586777, 7.3568850, -19.6808624, 19.5864944
24: -17.5560360, 6.2360029, -17.4739418, 6.2017450, -18.8247910, 18.7339478
25: -11.4809170, 10.1831541, -11.3683262, 10.1486435, -20.6018753, 20.5005035
26: -16.3178101, 9.8312016, -16.2050323, 9.8112011, -24.8443298, 24.7380524
27: -27.6267738, 0.8831015, -27.5374813, 0.8315229, -20.4252548, 20.3711205
28: -16.4168282, 7.3466597, -16.3237476, 7.3203030, -20.9358749, 20.8267097
29: -7.4765244, 10.4198246, -7.3590522, 10.4068546, -16.2826614, 16.1695480
30: -19.2455196, 7.4443836, -19.1360493, 7.4147196, -21.8946571, 21.7982712
31: -13.1931896, 9.5425968, -13.1139193, 9.5173779, -19.1252594, 19.0619240
32: -12.5455227, 9.4005146, -12.5026741, 9.3090868, -18.1537857, 18.2321358
33: -45.5068054, -9.2284756, -45.4533348, -9.3970604, -31.3892212, 31.5109482
34: -41.9934082, -13.9860773, -41.9629364, -14.0890036, -19.8637886, 19.9350586
35: -29.0793724, -2.4200270, -29.0378571, -2.4618201, -21.8887863, 21.8527374
36: -23.7744408, 3.8212585, -23.7228985, 3.7694426, -23.6580353, 23.5938759
37: -43.7167435, -4.5167418, -43.6618652, -4.6400070, -36.2563934, 36.2733231
38: -30.0538368, 1.4145203, -30.0074291, 1.3760529, -29.3812866, 29.2811546
39: -38.9920120, -3.8351750, -38.9490814, -3.9862528, -32.4995193, 32.5861511
40: -44.3932114, -12.2641897, -44.3576012, -12.4172134, -26.2812576, 26.4133949
41: -24.3041286, 5.1713753, -24.2737579, 5.0284233, -23.3616791, 23.4887848
42: -19.4897079, 2.3462572, -19.4615135, 2.2661753, -16.6336021, 16.6885223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1605

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5200601, upper bound: 13.4695869
time: 27.51 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5200680, upper bound: 13.4921700
time: 31.09 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -29.1097851, 4.1005559, -29.1605415, 4.0784922, -26.7984009, 26.9026108
1: -10.2177401, 6.6945429, -10.2183723, 6.6778488, -13.6024666, 13.6407719
2: -14.3236217, 4.4920940, -14.3707428, 4.4778481, -14.2941551, 14.3764038
3: -21.0064240, 0.8422284, -21.0141277, 0.8012547, -19.5919418, 19.6382942
4: -22.1525803, 3.4023118, -22.1861458, 3.3703804, -19.5708694, 19.6619682
5: -20.5417805, 5.8007631, -20.5518608, 5.7776523, -23.0836487, 23.1277084
6: -22.5313034, 3.2431488, -22.5243797, 3.2572594, -21.2916565, 21.2594223
7: -21.4518948, 4.0351954, -21.4640541, 4.0107069, -21.1365891, 21.1854324
8: -34.1613770, -4.0584435, -34.1938248, -4.0760951, -20.8714867, 20.9429817
9: -12.3159161, 16.6883545, -12.3329067, 16.6535683, -26.3959656, 26.5024185
10: -6.4266644, 20.7195606, -6.4178009, 20.6951675, -23.7205887, 23.7811050
11: -7.0168505, 13.9831944, -6.9607201, 13.9933491, -18.6356010, 18.5511551
12: 0.6313004, 35.3180618, 0.7032447, 35.3734665, -28.7980499, 28.6347809
13: -10.8321362, 24.3103981, -10.7634172, 24.3020172, -30.2082901, 30.1469955
14: -33.2853165, 10.4985085, -33.1384506, 10.5124540, -38.1824799, 38.0073547
15: -20.7425613, 0.3578246, -20.7499199, 0.3103480, -18.6275406, 18.6841812
16: -14.5580997, 7.5429645, -14.5647249, 7.4732018, -22.0313015, 22.1076889
17: -21.4932022, 18.8138351, -21.3342972, 18.8148117, -36.9184113, 36.7608414
18: -14.7304411, 9.5263910, -14.7208290, 9.5251284, -21.1795578, 21.1366577
19: -10.8787003, 6.8203540, -10.8709621, 6.8173599, -14.9841309, 14.9789619
20: -15.1024942, 4.9761229, -15.0856571, 4.9704437, -17.9797745, 17.9700775
21: -11.4108887, 9.6071758, -11.3853359, 9.6287518, -18.6878433, 18.6450272
22: -9.6680222, 7.8906813, -9.6246719, 7.8912177, -15.4630814, 15.4231377
23: -14.0841255, 7.3784666, -14.0836849, 7.3671942, -19.6355820, 19.6040688
24: -17.4918633, 6.2120352, -17.4913597, 6.2135720, -18.7652626, 18.7320442
25: -11.4223242, 10.1562824, -11.3778248, 10.1637917, -20.5473099, 20.4902878
26: -16.2782707, 9.8184414, -16.2301903, 9.8260708, -24.8381805, 24.7465591
27: -27.5483418, 0.8541818, -27.5495262, 0.8360991, -20.3579407, 20.3609085
28: -16.3583050, 7.3287344, -16.3434486, 7.3198261, -20.8762169, 20.8432426
29: -7.4431753, 10.4058933, -7.3822832, 10.4231148, -16.2523689, 16.1763763
30: -19.1959000, 7.4127121, -19.1783180, 7.4597301, -21.8896484, 21.8028183
31: -13.1431351, 9.5217133, -13.1264229, 9.5172596, -19.0736389, 19.0603180
32: -12.5296583, 9.3401661, -12.5341301, 9.3328981, -18.1701889, 18.1884613
33: -45.4867935, -9.2858591, -45.4763412, -9.3787251, -31.3883743, 31.4657745
34: -42.0017357, -13.9898186, -42.0020409, -14.0488768, -19.9068146, 19.9487534
35: -29.0742035, -2.4290550, -29.0631104, -2.4505398, -21.8944092, 21.8510780
36: -23.7659988, 3.8106673, -23.7493324, 3.7742839, -23.6557465, 23.6085129
37: -43.6859093, -4.5806465, -43.6773911, -4.6258626, -36.2437439, 36.2153778
38: -30.0480518, 1.3992548, -30.0351601, 1.3856943, -29.3834686, 29.2848663
39: -38.9650879, -3.8955781, -38.9706459, -3.9690597, -32.4881058, 32.5329285
40: -44.3648643, -12.3187466, -44.3706207, -12.4065046, -26.2663651, 26.3674316
41: -24.2803078, 5.1089211, -24.2948418, 5.0362492, -23.3514938, 23.4378586
42: -19.4706478, 2.2913213, -19.4680729, 2.2755737, -16.6259499, 16.6461391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1605

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4779720, upper bound: 13.4906326
time: 20.45 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4780358, upper bound: 13.5220215
time: 38.34 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -29.1271648, 4.1137848, -29.1619225, 4.0789976, -26.8148575, 26.9207916
1: -10.2246552, 6.7058821, -10.2192402, 6.6780071, -13.6083527, 13.6568146
2: -14.3365536, 4.4990520, -14.3713408, 4.4777088, -14.3091507, 14.3856773
3: -21.0333900, 0.8553135, -21.0148468, 0.8015964, -19.6298714, 19.6534729
4: -22.1655598, 3.4080806, -22.1869431, 3.3704057, -19.5767517, 19.6961861
5: -20.5848579, 5.8211651, -20.5524521, 5.7794929, -23.1376190, 23.1519241
6: -22.5523911, 3.2922401, -22.5283051, 3.2576828, -21.3086929, 21.3100815
7: -21.4884548, 4.0478382, -21.4651337, 4.0128326, -21.1768341, 21.1987457
8: -34.1780891, -4.0563664, -34.1944160, -4.0770564, -20.8846626, 20.9529266
9: -12.3349428, 16.7320957, -12.3350935, 16.6542168, -26.4176559, 26.5528412
10: -6.4319077, 20.7654762, -6.4184365, 20.6963882, -23.7312317, 23.8203545
11: -7.0643234, 14.0045834, -6.9618874, 14.0002022, -18.6962051, 18.5696449
12: 0.5862641, 35.4303360, 0.6892462, 35.3740692, -28.8357773, 28.7627411
13: -10.8628225, 24.3735981, -10.7723503, 24.3024178, -30.2336807, 30.2179337
14: -33.3470993, 10.5128727, -33.1396713, 10.5138426, -38.2418823, 38.0216217
15: -20.7659302, 0.3743508, -20.7503624, 0.3145905, -18.6537094, 18.6951523
16: -14.5735226, 7.5646439, -14.5670948, 7.4736185, -22.0471420, 22.1317387
17: -21.5063400, 18.8429985, -21.3355675, 18.8154812, -36.9278107, 36.7982330
18: -14.7769833, 9.5470123, -14.7210064, 9.5304756, -21.2236671, 21.1545715
19: -10.9191513, 6.8341260, -10.8715696, 6.8223858, -15.0330925, 14.9906235
20: -15.1584740, 5.0001059, -15.0863771, 4.9778986, -18.0455399, 17.9914322
21: -11.4616661, 9.6254921, -11.3859987, 9.6341019, -18.7464828, 18.6604347
22: -9.7077427, 7.9111919, -9.6248646, 7.8969636, -15.5102119, 15.4403343
23: -14.1389256, 7.4007163, -14.0843267, 7.3738308, -19.6990280, 19.6230507
24: -17.5586777, 6.2434301, -17.4914856, 6.2235522, -18.8429909, 18.7561989
25: -11.4828968, 10.1863270, -11.3781948, 10.1716681, -20.6197433, 20.5162506
26: -16.3234253, 9.8371773, -16.2304688, 9.8316641, -24.8683472, 24.7662659
27: -27.6319256, 0.8861051, -27.5499268, 0.8470163, -20.4542618, 20.3842735
28: -16.4216957, 7.3485236, -16.3444386, 7.3262410, -20.9519196, 20.8603897
29: -7.4856853, 10.4207773, -7.3825340, 10.4271965, -16.3040161, 16.1904602
30: -19.2648849, 7.4463863, -19.1789398, 7.4702406, -21.9752121, 21.8295135
31: -13.1960630, 9.5445461, -13.1270046, 9.5248566, -19.1387634, 19.0801315
32: -12.5615654, 9.4015007, -12.5422478, 9.3330698, -18.1942749, 18.2586060
33: -45.5125885, -9.2245359, -45.4802628, -9.3772860, -31.4145355, 31.5422974
34: -42.0114212, -13.9839191, -42.0023155, -14.0481548, -19.9200592, 19.9588699
35: -29.0870876, -2.4186792, -29.0633240, -2.4498487, -21.9049454, 21.8772736
36: -23.7804527, 3.8226848, -23.7493896, 3.7745550, -23.6718826, 23.6230316
37: -43.7218933, -4.5128093, -43.6836891, -4.6248755, -36.2789383, 36.3012085
38: -30.0568752, 1.4192278, -30.0350952, 1.3863850, -29.4007645, 29.3081970
39: -38.9941521, -3.8265667, -38.9765091, -3.9679375, -32.5166702, 32.6203842
40: -44.3980675, -12.2617188, -44.3777542, -12.4062576, -26.2988815, 26.4350281
41: -24.3117046, 5.1729870, -24.3005829, 5.0367045, -23.3796158, 23.5143509
42: -19.4946308, 2.3478422, -19.4753475, 2.2760472, -16.6444283, 16.7112656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=236, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1605
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1605

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5219586, upper bound: 13.4906326
time: 29.12 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5220219, upper bound: 13.5220215
time: 34.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 66.08 seconds
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 66.08
Output dim: 12, lower bound: -13.5200601, upper bound: 13.4418152
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 66.08
Output dim: 12, lower bound: -13.5200680, upper bound: 13.4643937
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 66.08
Output dim: 12, lower bound: -13.5219586, upper bound: 13.4628667
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 66.08
Output dim: 12, lower bound: -13.5220219, upper bound: 13.4942441
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 66.08
Output dim: 12, lower bound: -13.5200601, upper bound: 13.4695869
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 66.08
Output dim: 12, lower bound: -13.5200680, upper bound: 13.4921700
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 66.08
Output dim: 12, lower bound: -13.4779720, upper bound: 13.4906326
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 66.08
Output dim: 12, lower bound: -13.4780358, upper bound: 13.5220215
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 66.08
Output dim: 12, lower bound: -13.5219586, upper bound: 13.4906326
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 66.08
Output dim: 12, lower bound: -13.5220219, upper bound: 13.5220215

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -29.1010494, 4.0358925, -29.0910683, 3.9936152, -26.7234192, 26.7612457
1: -10.1982679, 6.6599522, -10.1960468, 6.6379104, -13.5515594, 13.5754929
2: -14.3266191, 4.4425631, -14.3170195, 4.4074936, -14.2443085, 14.2702179
3: -21.0125504, 0.7921860, -20.9961338, 0.7763941, -19.5873032, 19.5721741
4: -22.1551037, 3.3334441, -22.1521111, 3.2996635, -19.5003891, 19.5684814
5: -20.5762711, 5.7679944, -20.5383911, 5.7490954, -23.0923462, 23.0631180
6: -22.4825249, 3.2593451, -22.4339981, 3.2123184, -21.1896439, 21.1885567
7: -21.4631729, 3.9800675, -21.4336224, 3.9593880, -21.1042633, 21.0905495
8: -34.1624069, -4.1241865, -34.1489296, -4.1558747, -20.8045540, 20.8345146
9: -12.3137245, 16.6666718, -12.3071661, 16.6187744, -26.3705368, 26.4242935
10: -6.4054871, 20.7104473, -6.4071803, 20.6681099, -23.6882172, 23.7257233
11: -6.9817295, 13.9888210, -6.9362168, 13.9882603, -18.5867500, 18.5272102
12: 0.7450252, 35.3917389, 0.7891269, 35.3054657, -28.5969391, 28.6503296
13: -10.7389345, 24.3217697, -10.7343235, 24.2629414, -30.0633392, 30.1261292
14: -33.1375542, 10.3652802, -33.0855942, 10.4002466, -37.9026794, 37.8247147
15: -20.7450333, 0.2892380, -20.7261486, 0.2692182, -18.5821304, 18.5817833
16: -14.5460653, 7.4700537, -14.5416298, 7.4464540, -21.9925194, 22.0116844
17: -21.3132076, 18.7222157, -21.3098526, 18.7378178, -36.6544037, 36.6502151
18: -14.7376175, 9.5180922, -14.7031717, 9.5141125, -21.1456223, 21.1136513
19: -10.8784342, 6.8070951, -10.8398590, 6.7982554, -14.9681854, 14.9304161
20: -15.1174164, 4.9684668, -15.0625839, 4.9667912, -17.9879379, 17.9285698
21: -11.4048834, 9.6113529, -11.3530045, 9.6118507, -18.6637306, 18.6012459
22: -9.6434822, 7.8725934, -9.6004829, 7.8688097, -15.4188004, 15.3740120
23: -14.1039953, 7.3586864, -14.0521803, 7.3494673, -19.6089745, 19.5562210
24: -17.5322132, 6.2047968, -17.4690704, 6.1924410, -18.7683144, 18.7057114
25: -11.4214582, 10.1517248, -11.3634281, 10.1459742, -20.5255508, 20.4646759
26: -16.2383862, 9.8159199, -16.1968346, 9.8050575, -24.7314987, 24.7198486
27: -27.6060467, 0.8281260, -27.5292549, 0.8251715, -20.3896294, 20.2942009
28: -16.3784733, 7.3172073, -16.3155727, 7.3161402, -20.8635788, 20.7941475
29: -7.4002151, 10.3888721, -7.3487434, 10.3988686, -16.1954842, 16.1311340
30: -19.2076340, 7.4113407, -19.1191044, 7.4109802, -21.8511238, 21.7480278
31: -13.1538143, 9.5140991, -13.1065378, 9.5071983, -19.0701294, 19.0220184
32: -12.5106630, 9.3641167, -12.4799080, 9.3063011, -18.1079407, 18.1474457
33: -45.4206390, -9.3432465, -45.4170189, -9.4020767, -31.2954712, 31.3601379
34: -41.9419441, -14.0878010, -41.9243965, -14.0912962, -19.8132858, 19.7909584
35: -29.0371857, -2.4531555, -29.0205421, -2.4620743, -21.8136520, 21.8230896
36: -23.7299328, 3.7772961, -23.7121773, 3.7668979, -23.5724411, 23.5655098
37: -43.6333771, -4.5771966, -43.6268272, -4.6449399, -36.1564331, 36.2213593
38: -30.0040359, 1.3966324, -29.9999943, 1.3743224, -29.2797012, 29.2804642
39: -38.9312515, -3.9161510, -38.9385529, -3.9911427, -32.4164276, 32.5106812
40: -44.3065872, -12.3640537, -44.3152618, -12.4200859, -26.2036514, 26.2667770
41: -24.2506199, 5.0887842, -24.2389374, 5.0261164, -23.3174286, 23.3721695
42: -19.4513836, 2.3168669, -19.4328003, 2.2635336, -16.5754662, 16.6218300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1716
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1651
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1492
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1319
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1706

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5150323, upper bound: 13.4059819
time: 29.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5175888, upper bound: 13.4389741
time: 33.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -29.1017017, 4.0443931, -29.1377850, 4.0165863, -26.7393112, 26.8166199
1: -10.1984253, 6.6633196, -10.2120495, 6.6484499, -13.5597076, 13.5906830
2: -14.3271732, 4.4516630, -14.3647985, 4.4302936, -14.2592812, 14.3272476
3: -21.0130024, 0.7947054, -21.0174198, 0.7837496, -19.5938263, 19.5944977
4: -22.1551285, 3.3441458, -22.1956387, 3.3265238, -19.5175743, 19.6131859
5: -20.5767174, 5.7734380, -20.5691662, 5.7639112, -23.1032181, 23.0964355
6: -22.4958839, 3.2595882, -22.4670162, 3.2573185, -21.2472458, 21.2103004
7: -21.4640713, 3.9871781, -21.4692421, 3.9782989, -21.1178055, 21.1309891
8: -34.1628723, -4.1143637, -34.1913452, -4.1313868, -20.8184967, 20.8878517
9: -12.3140202, 16.6684952, -12.3182383, 16.6265240, -26.3777924, 26.4363785
10: -6.4050169, 20.7122097, -6.4086843, 20.6771450, -23.6932068, 23.7350159
11: -6.9837327, 13.9887543, -6.9454794, 13.9919014, -18.5927200, 18.5367393
12: 0.7312260, 35.3925705, 0.7547498, 35.3689957, -28.6756897, 28.6720123
13: -10.7393188, 24.3245850, -10.7549067, 24.2721272, -30.0729828, 30.1501694
14: -33.1391220, 10.3751993, -33.1281853, 10.4250450, -37.9256439, 37.8834610
15: -20.7451820, 0.2933614, -20.7406006, 0.2822406, -18.5932579, 18.5998116
16: -14.5469704, 7.4707775, -14.5523100, 7.4531574, -22.0001278, 22.0230865
17: -21.3143921, 18.7286053, -21.3302383, 18.7535763, -36.6688995, 36.6772537
18: -14.7382479, 9.5174236, -14.7071762, 9.5188084, -21.1582870, 21.1103096
19: -10.8795662, 6.8100457, -10.8583202, 6.8061624, -14.9756088, 14.9535179
20: -15.1189404, 4.9680758, -15.0738392, 4.9681673, -17.9935150, 17.9418182
21: -11.4065781, 9.6114616, -11.3622341, 9.6175385, -18.6726646, 18.6107750
22: -9.6442242, 7.8736701, -9.6049862, 7.8790026, -15.4292812, 15.3808212
23: -14.1053209, 7.3601599, -14.0646515, 7.3537321, -19.6160965, 19.5732994
24: -17.5328846, 6.2058039, -17.4838066, 6.2006316, -18.7775726, 18.7247887
25: -11.4219704, 10.1528139, -11.3687477, 10.1574707, -20.5339355, 20.4740715
26: -16.2395020, 9.8152695, -16.2100258, 9.8124304, -24.7485886, 24.7265167
27: -27.6079235, 0.8273621, -27.5376358, 0.8317852, -20.4036407, 20.2962418
28: -16.3803444, 7.3169298, -16.3267174, 7.3175120, -20.8701324, 20.8144722
29: -7.4028411, 10.3892784, -7.3583465, 10.4105444, -16.2027359, 16.1419411
30: -19.2113667, 7.4116526, -19.1313667, 7.4416456, -21.8831024, 21.7580681
31: -13.1547537, 9.5159950, -13.1236057, 9.5144663, -19.0777245, 19.0441589
32: -12.5195675, 9.3642712, -12.5046806, 9.3382950, -18.1500320, 18.1639252
33: -45.4281693, -9.3417892, -45.4398994, -9.3611088, -31.3340073, 31.3760681
34: -41.9523315, -14.0874395, -41.9499893, -14.0436449, -19.8711205, 19.8062973
35: -29.0420647, -2.4528034, -29.0346622, -2.4410343, -21.8352051, 21.8335381
36: -23.7334213, 3.7776008, -23.7249184, 3.7762151, -23.5850067, 23.5781326
37: -43.6410294, -4.5759873, -43.6473770, -4.6103687, -36.1972961, 36.2385101
38: -30.0049057, 1.3968463, -30.0080814, 1.3758671, -29.2896271, 29.2881737
39: -38.9313431, -3.9150491, -38.9434128, -3.9869320, -32.4262543, 32.5160980
40: -44.3146286, -12.3636675, -44.3359985, -12.3895063, -26.2433167, 26.2815704
41: -24.2591801, 5.0890923, -24.2615967, 5.0566530, -23.3559265, 23.3889389
42: -19.4629555, 2.3171010, -19.4615879, 2.3078818, -16.6255913, 16.6408348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1706

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4710608, upper bound: 13.4285707
time: 27.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5175970, upper bound: 13.4615533
time: 30.29 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -29.1033287, 4.0625272, -29.1581917, 4.0510707, -26.7632523, 26.8587952
1: -10.1988525, 6.6720343, -10.2183704, 6.6662812, -13.5702515, 13.6106625
2: -14.3284855, 4.4626598, -14.3697605, 4.4497657, -14.2720833, 14.3432846
3: -21.0139065, 0.7979543, -21.0089684, 0.7902544, -19.5998840, 19.5881271
4: -22.1557484, 3.3511338, -22.1864777, 3.3383203, -19.5269928, 19.6132812
5: -20.5781784, 5.7718573, -20.5512943, 5.7609720, -23.1114349, 23.0781937
6: -22.5107975, 3.2605324, -22.4928436, 3.2561150, -21.2620392, 21.2273521
7: -21.4655590, 3.9933250, -21.4624577, 3.9889812, -21.1265869, 21.1203156
8: -34.1634598, -4.1017904, -34.1920319, -4.1076865, -20.8325424, 20.9023743
9: -12.3161926, 16.6769638, -12.3319082, 16.6446533, -26.3972244, 26.4594345
10: -6.4067125, 20.7157593, -6.4122529, 20.6868801, -23.7072449, 23.7418518
11: -6.9870405, 13.9899807, -6.9529905, 13.9980755, -18.6035347, 18.5464211
12: 0.7180042, 35.3955841, 0.7317929, 35.3622322, -28.6607208, 28.6816330
13: -10.7430906, 24.3317451, -10.7678185, 24.2868195, -30.0911255, 30.1720543
14: -33.1428986, 10.3862915, -33.1279907, 10.4479361, -37.9582367, 37.9004517
15: -20.7470627, 0.3018682, -20.7476559, 0.3002143, -18.6071358, 18.6196022
16: -14.5482788, 7.4769554, -14.5634375, 7.4679852, -22.0162640, 22.0403938
17: -21.3173523, 18.7353306, -21.3256626, 18.7657166, -36.6836090, 36.6892090
18: -14.7409134, 9.5195065, -14.7120104, 9.5235653, -21.1716690, 21.1217232
19: -10.8820839, 6.8134446, -10.8650103, 6.8110294, -14.9808922, 14.9616432
20: -15.1229744, 4.9696341, -15.0792561, 4.9717870, -18.0021706, 17.9473114
21: -11.4135551, 9.6128693, -11.3759575, 9.6316032, -18.6957016, 18.6233292
22: -9.6507797, 7.8747282, -9.6197290, 7.8934250, -15.4484100, 15.3930950
23: -14.1072464, 7.3660078, -14.0779476, 7.3664436, -19.6269760, 19.5928154
24: -17.5348434, 6.2119222, -17.4866638, 6.2136383, -18.7866554, 18.7278900
25: -11.4234314, 10.1549215, -11.3733091, 10.1689653, -20.5434265, 20.4804001
26: -16.2440052, 9.8217773, -16.2222443, 9.8249817, -24.7559929, 24.7482681
27: -27.6112747, 0.8311434, -27.5417976, 0.8406653, -20.4190369, 20.3073540
28: -16.3833485, 7.3190846, -16.3362961, 7.3220038, -20.8796005, 20.8280869
29: -7.4092999, 10.3898401, -7.3720379, 10.4192257, -16.2174072, 16.1518669
30: -19.2268143, 7.4133778, -19.1615829, 7.4665403, -21.9316330, 21.7786865
31: -13.1566772, 9.5160294, -13.1195993, 9.5146465, -19.0837021, 19.0402184
32: -12.5267000, 9.3651428, -12.5194006, 9.3302832, -18.1483727, 18.1738548
33: -45.4263992, -9.3393326, -45.4439240, -9.3823195, -31.3202362, 31.3914719
34: -41.9600105, -14.0856457, -41.9637108, -14.0504436, -19.8695145, 19.8147316
35: -29.0449181, -2.4518287, -29.0460072, -2.4501092, -21.8294754, 21.8476906
36: -23.7359467, 3.7787085, -23.7386150, 3.7719727, -23.5862656, 23.5945816
37: -43.6385040, -4.5733271, -43.6485939, -4.6298609, -36.1788635, 36.2491913
38: -30.0070763, 1.4013627, -30.0278702, 1.3846381, -29.2993469, 29.3076324
39: -38.9333878, -3.9076786, -38.9660072, -3.9732103, -32.4330597, 32.5454712
40: -44.3114929, -12.3616199, -44.3353729, -12.4091692, -26.2212601, 26.2883072
41: -24.2581806, 5.0904169, -24.2657452, 5.0343418, -23.3352737, 23.3977051
42: -19.4562874, 2.3184528, -19.4457855, 2.2734137, -16.5862656, 16.6445389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: A, layer: 1, pos: 1790
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1747
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1706

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5169271, upper bound: 13.4270233
time: 29.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5194869, upper bound: 13.4600252
time: 50.16 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -29.1040363, 4.0710649, -29.2051239, 4.0742717, -26.7797089, 26.9168701
1: -10.1990356, 6.6753931, -10.2344151, 6.6769080, -13.5784912, 13.6274586
2: -14.3290472, 4.4717722, -14.4184561, 4.4728346, -14.2877121, 14.4033813
3: -21.0143776, 0.8004773, -21.0303612, 0.7975993, -19.6066628, 19.6108437
4: -22.1557751, 3.3624597, -22.2328148, 3.3683853, -19.5450974, 19.6656189
5: -20.5786247, 5.7773905, -20.5828552, 5.7765136, -23.1225739, 23.1120529
6: -22.5242367, 3.2607813, -22.5260696, 3.3011522, -21.3215790, 21.2501602
7: -21.4665108, 4.0004616, -21.4991093, 4.0083971, -21.1405716, 21.1641808
8: -34.1639404, -4.0919914, -34.2350655, -4.0831504, -20.8475189, 20.9583359
9: -12.3165159, 16.6788635, -12.3430672, 16.6525993, -26.4046783, 26.4715881
10: -6.4062166, 20.7175465, -6.4139447, 20.6964722, -23.7130280, 23.7515564
11: -6.9890928, 13.9899158, -6.9626956, 14.0016060, -18.6094246, 18.5562897
12: 0.7041612, 35.3964233, 0.6955409, 35.4284554, -28.7441711, 28.7043381
13: -10.7434788, 24.3345718, -10.7884846, 24.2964096, -30.1010208, 30.1963539
14: -33.1444893, 10.3961506, -33.1709137, 10.4729004, -37.9813538, 37.9594879
15: -20.7472095, 0.3060458, -20.7628174, 0.3141489, -18.6188240, 18.6377144
16: -14.5492439, 7.4776936, -14.5745440, 7.4747944, -22.0240383, 22.0522385
17: -21.3185654, 18.7417431, -21.3466320, 18.7820091, -36.6983948, 36.7167969
18: -14.7415276, 9.5188913, -14.7161160, 9.5283375, -21.1848831, 21.1192551
19: -10.8832798, 6.8163843, -10.8858385, 6.8192263, -14.9882355, 14.9871330
20: -15.1245489, 4.9692488, -15.0907507, 4.9731874, -18.0077248, 17.9607430
21: -11.4152746, 9.6129608, -11.3847628, 9.6372833, -18.7048187, 18.6331329
22: -9.6515408, 7.8758717, -9.6243248, 7.9039078, -15.4594879, 15.3999557
23: -14.1085787, 7.3675075, -14.0916414, 7.3707333, -19.6344147, 19.6105194
24: -17.5355186, 6.2133141, -17.5015717, 6.2224579, -18.7959328, 18.7477989
25: -11.4239817, 10.1560259, -11.3788271, 10.1809492, -20.5520210, 20.4900742
26: -16.2451153, 9.8218212, -16.2367477, 9.8334150, -24.7739868, 24.7573090
27: -27.6130867, 0.8304067, -27.5501328, 0.8468161, -20.4339142, 20.3099480
28: -16.3852425, 7.3187990, -16.3479118, 7.3238134, -20.8859406, 20.8494911
29: -7.4121304, 10.3902531, -7.3821859, 10.4313984, -16.2270813, 16.1628647
30: -19.2310715, 7.4136629, -19.1752377, 7.4981680, -21.9654694, 21.7894592
31: -13.1576405, 9.5179291, -13.1369381, 9.5223389, -19.0914688, 19.0631371
32: -12.5356455, 9.3652735, -12.5441570, 9.3624229, -18.1924438, 18.1907959
33: -45.4339409, -9.3378410, -45.4671021, -9.3410091, -31.3622284, 31.4076996
34: -41.9704285, -14.0852776, -41.9892769, -14.0026398, -19.9307938, 19.8301964
35: -29.0497932, -2.4514406, -29.0601692, -2.4289603, -21.8535690, 21.8580742
36: -23.7393913, 3.7790313, -23.7512474, 3.7813964, -23.5990829, 23.6071434
37: -43.6461258, -4.5720162, -43.6692810, -4.5949583, -36.2201385, 36.2666016
38: -30.0079231, 1.4015899, -30.0357437, 1.3860083, -29.3093109, 29.3155212
39: -38.9334717, -3.9062045, -38.9710388, -3.9681141, -32.4430161, 32.5512619
40: -44.3195381, -12.3612499, -44.3563309, -12.3777885, -26.2625656, 26.3034401
41: -24.2667675, 5.0907755, -24.2884655, 5.0649958, -23.3740311, 23.4148178
42: -19.4678860, 2.3187180, -19.4755936, 2.3179798, -16.6409836, 16.6638756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 1700
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 701
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1605
type: B, layer: 1, pos: 1771
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1676
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1739
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1492
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1706

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5169935, upper bound: 13.4584242
time: 27.47 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5195505, upper bound: 13.4914036
time: 31.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -29.1239376, 4.0770693, -29.0927429, 3.9984293, -26.7509537, 26.8110657
1: -10.2238522, 6.6897507, -10.1964464, 6.6403694, -13.5802841, 13.6167030
2: -14.3340292, 4.4681134, -14.3171034, 4.4106274, -14.2559013, 14.3000832
3: -21.0314808, 0.8457241, -21.0007076, 0.7789447, -19.6071777, 19.6315155
4: -22.1648788, 3.3777776, -22.1525059, 3.3022385, -19.5216370, 19.6364441
5: -20.5824242, 5.8107233, -20.5382729, 5.7528524, -23.1024628, 23.1269150
6: -22.5083389, 3.2907772, -22.4333534, 3.2133141, -21.2199554, 21.2350807
7: -21.4850578, 4.0260358, -21.4339657, 3.9637451, -21.1328888, 21.1562691
8: -34.1765327, -4.0904374, -34.1501160, -4.1520152, -20.8293762, 20.8719521
9: -12.3320379, 16.7192860, -12.3093557, 16.6226883, -26.3847580, 26.5141220
10: -6.4293013, 20.7576981, -6.4101648, 20.6720295, -23.7033768, 23.7979279
11: -7.0563965, 14.0031853, -6.9391322, 13.9898386, -18.6754189, 18.5434914
12: 0.6297402, 35.4254456, 0.7843428, 35.3149033, -28.7519531, 28.6923752
13: -10.8578176, 24.3598003, -10.7368488, 24.2695827, -30.1954956, 30.1657829
14: -33.3398285, 10.4802389, -33.0926743, 10.4392357, -38.1578674, 37.9298401
15: -20.7636719, 0.3560343, -20.7283001, 0.2705538, -18.6161880, 18.6511650
16: -14.5702066, 7.5564756, -14.5427628, 7.4491477, -22.0193539, 22.0992393
17: -21.5004654, 18.8223286, -21.3157673, 18.7703762, -36.8798676, 36.7479630
18: -14.7728662, 9.5436459, -14.7102461, 9.5166168, -21.1915741, 21.1396904
19: -10.9138994, 6.8233857, -10.8429222, 6.7995319, -15.0093689, 14.9513092
20: -15.1509399, 4.9972115, -15.0651855, 4.9689136, -18.0249443, 17.9662056
21: -11.4507389, 9.6234703, -11.3578358, 9.6132030, -18.7110786, 18.6327782
22: -9.6993322, 7.9074450, -9.6030655, 7.8686996, -15.4753456, 15.4166012
23: -14.1339769, 7.3910551, -14.0547943, 7.3517308, -19.6737442, 19.5802994
24: -17.5551796, 6.2323303, -17.4719372, 6.1934905, -18.8142586, 18.7274857
25: -11.4800863, 10.1818647, -11.3664837, 10.1457539, -20.5968094, 20.4964142
26: -16.3163300, 9.8286762, -16.2016335, 9.8055267, -24.8352356, 24.7291336
27: -27.6244602, 0.8811197, -27.5321388, 0.8269553, -20.4163704, 20.3599434
28: -16.4143906, 7.3456573, -16.3181896, 7.3180828, -20.9272499, 20.8177719
29: -7.4730110, 10.4193497, -7.3510284, 10.4057970, -16.2755623, 16.1601715
30: -19.2397003, 7.4439249, -19.1227016, 7.4136353, -21.8875160, 21.7848511
31: -13.1919060, 9.5385952, -13.1109676, 9.5083122, -19.1144371, 19.0547562
32: -12.5348797, 9.4002438, -12.4783792, 9.3085556, -18.1427536, 18.2079048
33: -45.4975815, -9.2302628, -45.4322357, -9.4011517, -31.3753128, 31.4901733
34: -41.9810371, -13.9865847, -41.9345474, -14.0902100, -19.8501167, 19.9059448
35: -29.0733204, -2.4204390, -29.0240211, -2.4627533, -21.8817902, 21.8393974
36: -23.7698612, 3.8208103, -23.7124844, 3.7683749, -23.6519852, 23.5827408
37: -43.7075424, -4.5182405, -43.6410027, -4.6434293, -36.2430878, 36.2503738
38: -30.0519714, 1.4142323, -30.0031586, 1.3753567, -29.3760986, 29.2725372
39: -38.9914665, -3.8374202, -38.9478378, -3.9911630, -32.4906693, 32.5776138
40: -44.3837318, -12.2646542, -44.3358688, -12.4183464, -26.2702484, 26.3913155
41: -24.2939720, 5.1708927, -24.2505379, 5.0273767, -23.3503418, 23.4646988
42: -19.4770565, 2.3458948, -19.4333191, 2.2653570, -16.6195526, 16.6605492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1345
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 768

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1706

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5150323, upper bound: 13.4358457
time: 70.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5175888, upper bound: 13.4671133
time: 51.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -29.1246262, 4.0855923, -29.1394234, 4.0213819, -26.7668457, 26.8664398
1: -10.2240067, 6.6931171, -10.2124672, 6.6509085, -13.5884247, 13.6318970
2: -14.3345900, 4.4771967, -14.3648949, 4.4333892, -14.2708893, 14.3571205
3: -21.0319023, 0.8482337, -21.0219994, 0.7863145, -19.6137276, 19.6538467
4: -22.1648808, 3.3884034, -22.1959438, 3.3290868, -19.5387726, 19.6811600
5: -20.5828171, 5.8161979, -20.5690403, 5.7677026, -23.1133728, 23.1602325
6: -22.5217171, 3.2909698, -22.4663181, 3.2583265, -21.2775803, 21.2568283
7: -21.4859467, 4.0331926, -21.4696026, 3.9826832, -21.1464386, 21.1967163
8: -34.1769905, -4.0806479, -34.1925011, -4.1274962, -20.8433342, 20.9252892
9: -12.3323383, 16.7211685, -12.3204346, 16.6304169, -26.3920288, 26.5261765
10: -6.4288201, 20.7594681, -6.4116597, 20.6811028, -23.7083588, 23.8072128
11: -7.0584221, 14.0031118, -6.9483771, 13.9934616, -18.6813812, 18.5530128
12: 0.6159372, 35.4263000, 0.7499323, 35.3783875, -28.8306198, 28.7140350
13: -10.8581772, 24.3626099, -10.7574501, 24.2787647, -30.2051086, 30.1898346
14: -33.3413773, 10.4901237, -33.1352997, 10.4640656, -38.1807861, 37.9884567
15: -20.7638054, 0.3601251, -20.7427731, 0.2835901, -18.6273003, 18.6691780
16: -14.5710945, 7.5571699, -14.5534592, 7.4558425, -22.0269375, 22.1106300
17: -21.5016842, 18.8287144, -21.3362064, 18.7861347, -36.8942871, 36.7750473
18: -14.7734833, 9.5429688, -14.7142506, 9.5213223, -21.2042160, 21.1363792
19: -10.9150782, 6.8263397, -10.8613987, 6.8074427, -15.0167847, 14.9744148
20: -15.1524410, 4.9968019, -15.0764551, 4.9702816, -18.0304947, 17.9794655
21: -11.4524307, 9.6235657, -11.3670645, 9.6188793, -18.7200317, 18.6423149
22: -9.7001019, 7.9085283, -9.6075573, 7.8788977, -15.4858341, 15.4234276
23: -14.1352835, 7.3925190, -14.0672779, 7.3560104, -19.6808624, 19.5973740
24: -17.5558491, 6.2333422, -17.4866695, 6.2016973, -18.8235130, 18.7465706
25: -11.4806395, 10.1829596, -11.3717823, 10.1572571, -20.6051788, 20.5058327
26: -16.3174076, 9.8280334, -16.2148247, 9.8128767, -24.8523254, 24.7357483
27: -27.6263084, 0.8803387, -27.5404453, 0.8336000, -20.4304047, 20.3619843
28: -16.4162693, 7.3454018, -16.3293285, 7.3194394, -20.9338188, 20.8380928
29: -7.4756293, 10.4197407, -7.3606520, 10.4174786, -16.2828140, 16.1709976
30: -19.2434444, 7.4442320, -19.1349525, 7.4443026, -21.9195251, 21.7949104
31: -13.1928272, 9.5404778, -13.1280565, 9.5155745, -19.1220169, 19.0768852
32: -12.5437756, 9.4004164, -12.5031290, 9.3405647, -18.1848373, 18.2243805
33: -45.5051537, -9.2287846, -45.4551735, -9.3602419, -31.4138260, 31.5061340
34: -41.9914207, -13.9862127, -41.9601288, -14.0425940, -19.9079552, 19.9212914
35: -29.0782127, -2.4200656, -29.0381203, -2.4417281, -21.9033203, 21.8498421
36: -23.7733421, 3.8211467, -23.7252121, 3.7777047, -23.6645126, 23.5953522
37: -43.7151299, -4.5169139, -43.6615486, -4.6088023, -36.2839966, 36.2675018
38: -30.0528259, 1.4144359, -30.0112305, 1.3769178, -29.3860092, 29.2802391
39: -38.9915581, -3.8363333, -38.9526672, -3.9869187, -32.5004578, 32.5830688
40: -44.3917351, -12.2642956, -44.3566322, -12.3877296, -26.3098984, 26.4061050
41: -24.3025284, 5.1712308, -24.2731915, 5.0579472, -23.3888245, 23.4814758
42: -19.4886131, 2.3461332, -19.4620800, 2.3097153, -16.6696854, 16.6795349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1573
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1370
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1706

## Relational analysis of IS_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4710608, upper bound: 13.4584361
time: 47.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5175970, upper bound: 13.4896975
time: 33.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -29.1096153, 4.0989976, -29.2053642, 4.0785460, -26.7908173, 26.9485474
1: -10.2177105, 6.6938677, -10.2339478, 6.6792259, -13.6013718, 13.6526451
2: -14.3235149, 4.4903436, -14.4179153, 4.4760675, -14.2843208, 14.4239769
3: -21.0063381, 0.8408990, -21.0342426, 0.7998424, -19.5886002, 19.6550179
4: -22.1525822, 3.4009819, -22.2323761, 3.3708491, -19.5604324, 19.6993675
5: -20.5417080, 5.7997084, -20.5821304, 5.7784724, -23.0787277, 23.1516342
6: -22.5289383, 3.2431211, -22.5214081, 3.3017464, -21.3349457, 21.2459717
7: -21.4518013, 4.0338883, -21.4983330, 4.0106249, -21.1289902, 21.2166672
8: -34.1613007, -4.0603256, -34.2356644, -4.0783811, -20.8591690, 20.9858055
9: -12.3157959, 16.6877747, -12.3430557, 16.6558762, -26.3972321, 26.5109482
10: -6.4247780, 20.7189026, -6.4162650, 20.6991997, -23.7175140, 23.7844925
11: -7.0163021, 13.9828701, -6.9644566, 13.9963360, -18.6375008, 18.5540771
12: 0.6338983, 35.3178940, 0.7046628, 35.4372406, -28.8614044, 28.6183701
13: -10.8317041, 24.3093681, -10.7820282, 24.3027153, -30.2077332, 30.1650925
14: -33.2848969, 10.4967041, -33.1767807, 10.5105801, -38.1771088, 38.0502548
15: -20.7424641, 0.3563187, -20.7645264, 0.3112526, -18.6267052, 18.6961098
16: -14.5579433, 7.5424070, -14.5733204, 7.4770656, -22.0350094, 22.1157265
17: -21.4926128, 18.8127079, -21.3512993, 18.8138752, -36.9143372, 36.7771454
18: -14.7302151, 9.5238171, -14.7230139, 9.5254688, -21.1866837, 21.1274223
19: -10.8783350, 6.8188863, -10.8882713, 6.8154612, -14.9804611, 14.9963493
20: -15.1020737, 4.9739552, -15.0926399, 4.9678402, -17.9789200, 17.9770432
21: -11.4103651, 9.6067400, -11.3889151, 9.6332607, -18.6935272, 18.6492310
22: -9.6676731, 7.8902025, -9.6267014, 7.8980780, -15.4689293, 15.4253769
23: -14.0837965, 7.3776236, -14.0935898, 7.3663359, -19.6357117, 19.6156120
24: -17.4916286, 6.2095022, -17.5042686, 6.2135143, -18.7641258, 18.7453995
25: -11.4220753, 10.1561337, -11.3814812, 10.1728477, -20.5508575, 20.4958839
26: -16.2779045, 9.8158379, -16.2412701, 9.8282814, -24.8475380, 24.7468262
27: -27.5478745, 0.8514671, -27.5525875, 0.8377118, -20.3643494, 20.3523560
28: -16.3577728, 7.3274727, -16.3495026, 7.3193378, -20.8739090, 20.8559761
29: -7.4424267, 10.4058352, -7.3842182, 10.4342585, -16.2555275, 16.1778374
30: -19.1941643, 7.4125366, -19.1781769, 7.4903116, -21.9163208, 21.7996101
31: -13.1427889, 9.5196075, -13.1407852, 9.5158768, -19.0706367, 19.0760689
32: -12.5279369, 9.3400764, -12.5345068, 9.3645077, -18.2031746, 18.1811066
33: -45.4851379, -9.2861481, -45.4783173, -9.3415546, -31.4158859, 31.4612045
34: -41.9997406, -13.9899702, -41.9991684, -14.0022907, -19.9543877, 19.9350891
35: -29.0730247, -2.4290957, -29.0634346, -2.4303083, -21.9111938, 21.8482018
36: -23.7648849, 3.8105042, -23.7515202, 3.7826257, -23.6625290, 23.6098747
37: -43.6842842, -4.5808849, -43.6771469, -4.5944095, -36.2717133, 36.2097626
38: -30.0470200, 1.3991771, -30.0389309, 1.3863487, -29.3883896, 29.2842216
39: -38.9645996, -3.8965492, -38.9744415, -3.9691973, -32.4887238, 32.5307541
40: -44.3634109, -12.3189030, -44.3697586, -12.3763161, -26.2966843, 26.3604164
41: -24.2787018, 5.1087794, -24.2943115, 5.0658617, -23.3788910, 23.4308472
42: -19.4695587, 2.2912469, -19.4688187, 2.3193679, -16.6665955, 16.6374607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1733
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1706

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4730133, upper bound: 13.4882936
time: 32.20 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.4755655, upper bound: 13.5195499
time: 29.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -29.1262360, 4.1037059, -29.1598549, 4.0558891, -26.7907639, 26.9086227
1: -10.2244415, 6.7018266, -10.2187653, 6.6687465, -13.5989685, 13.6518726
2: -14.3358955, 4.4882212, -14.3698502, 4.4528451, -14.2836914, 14.3731499
3: -21.0328331, 0.8514817, -21.0135498, 0.7927959, -19.6198006, 19.6474762
4: -22.1655025, 3.3953981, -22.1868248, 3.3408833, -19.5482025, 19.6812515
5: -20.5843163, 5.8145876, -20.5511837, 5.7647414, -23.1215973, 23.1419754
6: -22.5366135, 3.2919722, -22.4921551, 3.2570992, -21.2923813, 21.2738304
7: -21.4874382, 4.0393381, -21.4627762, 3.9933012, -21.1551971, 21.1860886
8: -34.1775475, -4.0680833, -34.1932106, -4.1038446, -20.8573837, 20.9398155
9: -12.3345003, 16.7296410, -12.3340931, 16.6485538, -26.4114304, 26.5492325
10: -6.4305105, 20.7630196, -6.4152284, 20.6907997, -23.7223740, 23.8140755
11: -7.0616941, 14.0043325, -6.9559135, 13.9996462, -18.6921768, 18.5627289
12: 0.6027298, 35.4292679, 0.7270207, 35.3716469, -28.8156738, 28.7236633
13: -10.8619738, 24.3697395, -10.7702961, 24.2934799, -30.2232208, 30.2117119
14: -33.3450623, 10.5011253, -33.1350708, 10.4869471, -38.2133789, 38.0055237
15: -20.7656860, 0.3686433, -20.7498188, 0.3015225, -18.6411934, 18.6889648
16: -14.5724049, 7.5633678, -14.5645580, 7.4706869, -22.0430908, 22.1279259
17: -21.5046120, 18.8354874, -21.3315582, 18.7982655, -36.9089508, 36.7869949
18: -14.7761745, 9.5450783, -14.7191029, 9.5260553, -21.2176056, 21.1477585
19: -10.9175873, 6.8297176, -10.8680830, 6.8123016, -15.0220642, 14.9825249
20: -15.1564903, 4.9983587, -15.0818481, 4.9739175, -18.0391464, 17.9849510
21: -11.4594002, 9.6249771, -11.3807735, 9.6329517, -18.7430382, 18.6548691
22: -9.7066288, 7.9095769, -9.6223164, 7.8933229, -15.5049553, 15.4356976
23: -14.1372242, 7.3983784, -14.0805721, 7.3686914, -19.6917572, 19.6169052
24: -17.5578136, 6.2394829, -17.4895020, 6.2146945, -18.8325729, 18.7496681
25: -11.4821091, 10.1850529, -11.3763456, 10.1687632, -20.6146698, 20.5121498
26: -16.3219147, 9.8345528, -16.2270622, 9.8254566, -24.8597565, 24.7575226
27: -27.6296616, 0.8840809, -27.5446529, 0.8424811, -20.4457703, 20.3730812
28: -16.4192486, 7.3475294, -16.3388939, 7.3239689, -20.9432640, 20.8517075
29: -7.4821095, 10.4202995, -7.3743248, 10.4261484, -16.2974815, 16.1809120
30: -19.2588692, 7.4459152, -19.1651611, 7.4691658, -21.9680748, 21.8155022
31: -13.1947803, 9.5404873, -13.1240540, 9.5157909, -19.1279831, 19.0729485
32: -12.5509319, 9.4012632, -12.5178623, 9.3325310, -18.1831512, 18.2343025
33: -45.5033798, -9.2263231, -45.4591599, -9.3814936, -31.4000320, 31.5215073
34: -41.9990463, -13.9844542, -41.9738731, -14.0493870, -19.9063492, 19.9297447
35: -29.0810280, -2.4190938, -29.0494957, -2.4508107, -21.8976059, 21.8640137
36: -23.7758713, 3.8222032, -23.7389297, 3.7734780, -23.6657829, 23.6118088
37: -43.7126160, -4.5143156, -43.6627808, -4.6283412, -36.2656097, 36.2781525
38: -30.0550385, 1.4189446, -30.0310516, 1.3856812, -29.3956680, 29.2997055
39: -38.9936447, -3.8289142, -38.9752693, -3.9731860, -32.5073166, 32.6124420
40: -44.3885880, -12.2621937, -44.3560104, -12.4074059, -26.2878494, 26.4128609
41: -24.3015404, 5.1725459, -24.2773495, 5.0356188, -23.3682251, 23.4901962
42: -19.4819622, 2.3474846, -19.4462833, 2.2752409, -16.6303520, 16.6832428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1361
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1409
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1706

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5169271, upper bound: 13.4568813
time: 31.96 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5194869, upper bound: 13.4881555
time: 33.25 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -29.1269398, 4.1122551, -29.2067318, 4.0790544, -26.8072662, 26.9666977
1: -10.2246218, 6.7051907, -10.2348175, 6.6793785, -13.6072464, 13.6686745
2: -14.3364601, 4.4973106, -14.4185400, 4.4759474, -14.2993279, 14.4332466
3: -21.0332794, 0.8540370, -21.0349312, 0.8001482, -19.6265602, 19.6701965
4: -22.1655312, 3.4067750, -22.2332001, 3.3709307, -19.5663147, 19.7335968
5: -20.5847855, 5.8201590, -20.5827713, 5.7803307, -23.1326904, 23.1758728
6: -22.5500393, 3.2921715, -22.5253716, 3.3021469, -21.3519440, 21.2966652
7: -21.4883690, 4.0464697, -21.4994354, 4.0127811, -21.1691895, 21.2299728
8: -34.1780319, -4.0582519, -34.2362633, -4.0793409, -20.8723450, 20.9957771
9: -12.3348103, 16.7314987, -12.3452702, 16.6565018, -26.4189377, 26.5614090
10: -6.4299946, 20.7647839, -6.4168844, 20.7003651, -23.7281647, 23.8237572
11: -7.0637484, 14.0042686, -6.9656096, 14.0031624, -18.6981163, 18.5725899
12: 0.5888958, 35.4301071, 0.6906972, 35.4378624, -28.8991776, 28.7463837
13: -10.8623877, 24.3725834, -10.7909594, 24.3030624, -30.2331085, 30.2360077
14: -33.3467178, 10.5110865, -33.1779556, 10.5119476, -38.2364807, 38.0645294
15: -20.7658348, 0.3728452, -20.7649727, 0.3155088, -18.6528816, 18.7070770
16: -14.5733738, 7.5640988, -14.5756931, 7.4774866, -22.0508614, 22.1397915
17: -21.5057507, 18.8419132, -21.3525333, 18.8145065, -36.9237823, 36.8145447
18: -14.7767420, 9.5444288, -14.7231827, 9.5308180, -21.2307739, 21.1453209
19: -10.9187908, 6.8326907, -10.8888817, 6.8204904, -15.0294037, 15.0080032
20: -15.1580687, 4.9979620, -15.0933619, 4.9752951, -18.0447044, 17.9983978
21: -11.4611521, 9.6250563, -11.3895817, 9.6386166, -18.7521667, 18.6646538
22: -9.7073631, 7.9107089, -9.6269169, 7.9038157, -15.5160427, 15.4425659
23: -14.1385746, 7.3998713, -14.0942211, 7.3730087, -19.6991692, 19.6345825
24: -17.5585060, 6.2408686, -17.5044212, 6.2234902, -18.8418503, 18.7695732
25: -11.4826517, 10.1861687, -11.3818779, 10.1807499, -20.6232834, 20.5218353
26: -16.3230267, 9.8345747, -16.2415447, 9.8338757, -24.8777046, 24.7665558
27: -27.6314964, 0.8833694, -27.5530033, 0.8486156, -20.4606628, 20.3756866
28: -16.4211769, 7.3472528, -16.3505020, 7.3257384, -20.9496346, 20.8730927
29: -7.4849520, 10.4207277, -7.3845205, 10.4383392, -16.3071518, 16.1919289
30: -19.2631626, 7.4462271, -19.1788406, 7.5008001, -22.0018845, 21.8262901
31: -13.1957188, 9.5424080, -13.1413641, 9.5234737, -19.1357346, 19.0958633
32: -12.5598564, 9.4014359, -12.5426235, 9.3646946, -18.2272453, 18.2512398
33: -45.5109406, -9.2248001, -45.4822731, -9.3401127, -31.4420700, 31.5377235
34: -42.0094490, -13.9840631, -41.9994698, -14.0016127, -19.9676285, 19.9451942
35: -29.0859108, -2.4187231, -29.0636272, -2.4296241, -21.9217148, 21.8744202
36: -23.7793159, 3.8225367, -23.7515583, 3.7828903, -23.6786499, 23.6244011
37: -43.7202606, -4.5130210, -43.6834717, -4.5934210, -36.3069153, 36.2955551
38: -30.0559082, 1.4191599, -30.0389233, 1.3870559, -29.4057007, 29.3075562
39: -38.9937782, -3.8274937, -38.9803391, -3.9680901, -32.5172958, 32.6182175
40: -44.3966217, -12.2618332, -44.3769150, -12.3760843, -26.3292160, 26.4280243
41: -24.3100777, 5.1728611, -24.3000717, 5.0662584, -23.4069519, 23.5073318
42: -19.4935722, 2.3477592, -19.4761124, 2.3198237, -16.6850586, 16.7025852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=235, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1706
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: A, layer: 1, pos: 1703
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1706

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5169935, upper bound: 13.4882936
time: 28.04 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 12, lower bound: -13.5195505, upper bound: 13.5195499
time: 27.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 57.87 seconds
IS_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5150323, upper bound: 13.4059819
IS_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5175888, upper bound: 13.4389741
IS_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.4710608, upper bound: 13.4285707
IS_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5175970, upper bound: 13.4615533
IS_A1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5169271, upper bound: 13.4270233
IS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5194869, upper bound: 13.4600252
IS_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5169935, upper bound: 13.4584242
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5195505, upper bound: 13.4914036
IS_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5150323, upper bound: 13.4358457
IS_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5175888, upper bound: 13.4671133
IS_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.4710608, upper bound: 13.4584361
IS_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5175970, upper bound: 13.4896975
IS_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.4730133, upper bound: 13.4882936
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.4755655, upper bound: 13.5195499
IS_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5169271, upper bound: 13.4568813
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5194869, upper bound: 13.4881555
IS_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5169935, upper bound: 13.4882936
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 57.87
Output dim: 12, lower bound: -13.5195505, upper bound: 13.5195499

## BFS IS instance: IS_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -29.1018848, 4.0599356, -29.1950188, 4.0508976, -26.7558670, 26.8941574
1: -10.1984215, 6.6706362, -10.2258072, 6.6674428, -13.5728645, 13.6073494
2: -14.3277721, 4.4600172, -14.3999004, 4.4464636, -14.2644653, 14.3720551
3: -21.0134411, 0.7965193, -21.0272522, 0.7900286, -19.5962791, 19.6068497
4: -22.1552773, 3.3492975, -22.1931534, 3.3383617, -19.5362930, 19.5979996
5: -20.5772476, 5.7695122, -20.5906582, 5.7609949, -23.1076965, 23.1180420
6: -22.5092354, 3.2585602, -22.4941807, 3.2537270, -21.2580719, 21.2263107
7: -21.4643326, 3.9900236, -21.5024757, 3.9847322, -21.1164017, 21.1607170
8: -34.1631050, -4.1046677, -34.2126045, -4.1121411, -20.8231506, 20.9152374
9: -12.3140173, 16.6763039, -12.3308802, 16.6558552, -26.4075699, 26.4563904
10: -6.4047604, 20.7139912, -6.4115567, 20.7035027, -23.7180328, 23.7365837
11: -6.9858971, 13.9882641, -6.9726319, 13.9965324, -18.5997581, 18.5665092
12: 0.7230105, 35.3949585, 0.7394958, 35.4192657, -28.7194824, 28.6684570
13: -10.7391472, 24.3304672, -10.7627821, 24.2892170, -30.0884857, 30.1645432
14: -33.1418304, 10.3828735, -33.1718445, 10.4436836, -37.9511108, 37.9405975
15: -20.7442741, 0.3007662, -20.7471542, 0.3086846, -18.6200905, 18.6131363
16: -14.5473328, 7.4747949, -14.5710621, 7.4704700, -22.0178032, 22.0458565
17: -21.3165417, 18.7332706, -21.3299675, 18.7644711, -36.6814270, 36.6909485
18: -14.7398844, 9.5170555, -14.7128315, 9.5216331, -21.1685638, 21.1182098
19: -10.8811836, 6.8120232, -10.8859959, 6.8090706, -14.9765816, 14.9835014
20: -15.1219893, 4.9674740, -15.1128254, 4.9685121, -17.9966888, 17.9802170
21: -11.4125910, 9.6117878, -11.3851223, 9.6304741, -18.6852798, 18.6413231
22: -9.6463947, 7.8735266, -9.6162319, 7.9020910, -15.4622040, 15.3889122
23: -14.1064415, 7.3649755, -14.1069536, 7.3669710, -19.6234512, 19.6205521
24: -17.5346680, 6.2098689, -17.4998970, 6.2113562, -18.7829590, 18.7410622
25: -11.4231262, 10.1535902, -11.3784590, 10.1709785, -20.5445786, 20.4852676
26: -16.2416763, 9.8209772, -16.2256088, 9.8388195, -24.7727737, 24.7437668
27: -27.6097984, 0.8288207, -27.5541134, 0.8379502, -20.4115677, 20.3114700
28: -16.3820610, 7.3168697, -16.3629951, 7.3190112, -20.8736725, 20.8550644
29: -7.4061251, 10.3893318, -7.3706312, 10.4199800, -16.2122307, 16.1506691
30: -19.2263775, 7.4115171, -19.1705093, 7.4654408, -21.9273376, 21.7901802
31: -13.1560287, 9.5140810, -13.1430597, 9.5125904, -19.0800095, 19.0637741
32: -12.5238800, 9.3650379, -12.5175018, 9.3350163, -18.1489410, 18.1708794
33: -45.4231415, -9.3408566, -45.4450226, -9.3505325, -31.3508148, 31.3877182
34: -41.9574661, -14.0862494, -41.9600143, -14.0282040, -19.8932152, 19.8045807
35: -29.0412750, -2.4524639, -29.0423470, -2.4290197, -21.8548126, 21.8389931
36: -23.7323875, 3.7784896, -23.7346458, 3.7805700, -23.5906296, 23.5888710
37: -43.6356697, -4.5738835, -43.6481400, -4.5936470, -36.2118835, 36.2452087
38: -30.0051613, 1.3988810, -30.0261574, 1.3816814, -29.2958450, 29.2993927
39: -38.9300880, -3.9089983, -38.9641418, -3.9382806, -32.4645004, 32.5393982
40: -44.3096390, -12.3619003, -44.3364639, -12.3992901, -26.2324142, 26.2840958
41: -24.2553177, 5.0901680, -24.2628536, 5.0448594, -23.3439102, 23.3937988
42: -19.4541054, 2.3177338, -19.4462204, 2.2833109, -16.5974083, 16.6437302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=234, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1481
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1411
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1492
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 685

## Relational analysis of IS_A1_B2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4850487, upper bound: 13.4589728
time: 15.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5187566, upper bound: 13.4592966
time: 27.99 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -29.1025639, 4.0684886, -29.2419205, 4.0740509, -26.7723694, 26.9522247
1: -10.1985912, 6.6739855, -10.2418633, 6.6780815, -13.5811157, 13.6241474
2: -14.3283272, 4.4691057, -14.4485703, 4.4695458, -14.2800903, 14.4321556
3: -21.0139065, 0.7990453, -21.0486565, 0.7974110, -19.6030807, 19.6296005
4: -22.1553078, 3.3606238, -22.2395210, 3.3683958, -19.5544167, 19.6503105
5: -20.5777416, 5.7750192, -20.6222706, 5.7765512, -23.1188354, 23.1519089
6: -22.5226479, 3.2587719, -22.5274334, 3.2987680, -21.3176346, 21.2490959
7: -21.4652557, 3.9971917, -21.5391483, 4.0042119, -21.1304550, 21.2045822
8: -34.1635933, -4.0948381, -34.2556610, -4.0876460, -20.8381081, 20.9711914
9: -12.3143206, 16.6781578, -12.3419895, 16.6637897, -26.4150391, 26.4685593
10: -6.4042568, 20.7157326, -6.4132509, 20.7131100, -23.7238388, 23.7462959
11: -6.9879522, 13.9881878, -6.9823608, 14.0000534, -18.6057014, 18.5763702
12: 0.7091651, 35.3958740, 0.7031879, 35.4854736, -28.8029633, 28.6912003
13: -10.7395430, 24.3333015, -10.7833729, 24.2987766, -30.0983734, 30.1888733
14: -33.1434212, 10.3927174, -33.2148056, 10.4686356, -37.9742584, 37.9995651
15: -20.7444534, 0.3049293, -20.7623062, 0.3226566, -18.6317711, 18.6312485
16: -14.5483027, 7.4755135, -14.5821829, 7.4772725, -22.0255756, 22.0576973
17: -21.3177147, 18.7396984, -21.3509560, 18.7807083, -36.6962128, 36.7184830
18: -14.7405005, 9.5163898, -14.7169132, 9.5263882, -21.1817703, 21.1157646
19: -10.8823719, 6.8149662, -10.9068203, 6.8172536, -14.9839287, 15.0089951
20: -15.1235332, 4.9670830, -15.1243458, 4.9699306, -18.0022469, 17.9936600
21: -11.4143324, 9.6118717, -11.3939438, 9.6361504, -18.6944046, 18.6511497
22: -9.6471615, 7.8746920, -9.6208239, 7.9125671, -15.4732609, 15.3957863
23: -14.1077805, 7.3664694, -14.1206532, 7.3712997, -19.6309052, 19.6382675
24: -17.5353432, 6.2112474, -17.5148087, 6.2201767, -18.7922363, 18.7609482
25: -11.4236603, 10.1547022, -11.3839531, 10.1829548, -20.5531807, 20.4949799
26: -16.2428093, 9.8210049, -16.2401066, 9.8472242, -24.7907181, 24.7527924
27: -27.6116066, 0.8281236, -27.5624447, 0.8441448, -20.4264297, 20.3140678
28: -16.3839455, 7.3165765, -16.3746033, 7.3208156, -20.8799973, 20.8764801
29: -7.4089642, 10.3897400, -7.3807716, 10.4321699, -16.2219200, 16.1616859
30: -19.2306919, 7.4118576, -19.1841679, 7.4971104, -21.9611893, 21.8009796
31: -13.1569672, 9.5159664, -13.1603737, 9.5202751, -19.0877686, 19.0867004
32: -12.5327902, 9.3651562, -12.5422783, 9.3671684, -18.1930389, 18.1878204
33: -45.4307213, -9.3393154, -45.4681625, -9.3092079, -31.3928299, 31.4039459
34: -41.9678497, -14.0858660, -41.9855881, -13.9804106, -19.9544678, 19.8200531
35: -29.0461445, -2.4520755, -29.0565014, -2.4078851, -21.8789330, 21.8493996
36: -23.7359009, 3.7788272, -23.7472878, 3.7899799, -23.6034622, 23.6014557
37: -43.6432724, -4.5725713, -43.6688309, -4.5587301, -36.2532120, 36.2626266
38: -30.0059891, 1.3991418, -30.0340366, 1.3830447, -29.3058319, 29.3072815
39: -38.9301643, -3.9075696, -38.9692078, -3.9331722, -32.4744720, 32.5451660
40: -44.3176842, -12.3615465, -44.3574104, -12.3679113, -26.2737732, 26.2992630
41: -24.2639084, 5.0904942, -24.2855721, 5.0754848, -23.3826294, 23.4109192
42: -19.4657059, 2.3179998, -19.4760323, 2.3279042, -16.6521111, 16.6630516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=234, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: B, layer: 1, pos: 1704
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: B, layer: 1, pos: 1689
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1613
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1706
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1755
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: B, layer: 1, pos: 995
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1732
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1660
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1377
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1301
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1283
type: B, layer: 1, pos: 1317
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1317
type: B, layer: 1, pos: 1390
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 685

## Relational analysis of IS_A1_B2_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4851110, upper bound: 13.4903430
time: 24.26 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5188204, upper bound: 13.4906742
time: 32.17 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -29.1081123, 4.0964303, -29.2422409, 4.0783472, -26.7834473, 26.9839325
1: -10.2172728, 6.6924381, -10.2414017, 6.6804113, -13.6039581, 13.6493053
2: -14.3227615, 4.4876938, -14.4480572, 4.4728103, -14.2766991, 14.4529037
3: -21.0058575, 0.8391285, -21.0517368, 0.7996821, -19.5849876, 19.6730270
4: -22.1520958, 3.3989615, -22.2390766, 3.3709111, -19.5697403, 19.6840782
5: -20.5407505, 5.7974291, -20.6218224, 5.7785215, -23.0747757, 23.1918716
6: -22.5273857, 3.2411060, -22.5227432, 3.2993631, -21.3310013, 21.2446861
7: -21.4505119, 4.0305805, -21.5383530, 4.0064726, -21.1186142, 21.2576447
8: -34.1609764, -4.0632243, -34.2562294, -4.0828171, -20.8497429, 20.9982567
9: -12.3135815, 16.6870899, -12.3420868, 16.6670647, -26.4075775, 26.5079041
10: -6.4228387, 20.7170696, -6.4156132, 20.7157974, -23.7283325, 23.7793427
11: -7.0151358, 13.9811592, -6.9841003, 13.9948025, -18.6337700, 18.5741997
12: 0.6377530, 35.3172951, 0.7122679, 35.4955521, -28.9163818, 28.6052246
13: -10.8276596, 24.3080597, -10.7769909, 24.3050804, -30.2050247, 30.1577148
14: -33.2838211, 10.4932852, -33.2206001, 10.5062628, -38.1700134, 38.0903778
15: -20.7396851, 0.3551812, -20.7640953, 0.3197134, -18.6396713, 18.6896706
16: -14.5569878, 7.5402207, -14.5809336, 7.4795809, -22.0365677, 22.1211548
17: -21.4918079, 18.8106327, -21.3556480, 18.8126030, -36.9121704, 36.7789307
18: -14.7291460, 9.5213757, -14.7238312, 9.5235777, -21.1835518, 21.1239319
19: -10.8774223, 6.8174829, -10.9092484, 6.8135076, -14.9761581, 15.0182304
20: -15.1010494, 4.9718266, -15.1261978, 4.9645848, -17.9734879, 18.0099258
21: -11.4093885, 9.6056557, -11.3981018, 9.6321402, -18.6831093, 18.6673431
22: -9.6632843, 7.8889589, -9.6232691, 7.9067287, -15.4824448, 15.4211960
23: -14.0829687, 7.3768063, -14.1226120, 7.3669100, -19.6322250, 19.6433067
24: -17.4914589, 6.2074213, -17.5175209, 6.2112136, -18.7604599, 18.7588120
25: -11.4217491, 10.1547737, -11.3866186, 10.1748438, -20.5520210, 20.5007515
26: -16.2754326, 9.8150425, -16.2446842, 9.8421211, -24.8644104, 24.7423706
27: -27.5463753, 0.8492575, -27.5650558, 0.8350124, -20.3565636, 20.3557472
28: -16.3564682, 7.3252416, -16.3762264, 7.3163552, -20.8680000, 20.8829422
29: -7.4392643, 10.4053154, -7.3828297, 10.4350224, -16.2503586, 16.1766434
30: -19.1937542, 7.4106855, -19.1871204, 7.4892321, -21.9120483, 21.8111229
31: -13.1421289, 9.5176573, -13.1642208, 9.5138359, -19.0669708, 19.0996094
32: -12.5250778, 9.3399639, -12.5326271, 9.3692789, -18.2037544, 18.1781082
33: -45.4819565, -9.2876453, -45.4794197, -9.3097410, -31.4465256, 31.4568787
34: -41.9971771, -13.9905214, -41.9954910, -13.9800148, -19.9780693, 19.9249611
35: -29.0693436, -2.4297140, -29.0597572, -2.4092391, -21.9364815, 21.8394966
36: -23.7616215, 3.8103073, -23.7477074, 3.7912271, -23.6668854, 23.6041870
37: -43.6814995, -4.5814428, -43.6767159, -4.5582232, -36.3044510, 36.2058334
38: -30.0450783, 1.3966994, -30.0372391, 1.3833795, -29.3849945, 29.2759399
39: -38.9612160, -3.8979082, -38.9724426, -3.9342873, -32.5201645, 32.5246582
40: -44.3615456, -12.3191891, -44.3708344, -12.3664436, -26.3078613, 26.3562317
41: -24.2757931, 5.1084876, -24.2913971, 5.0763230, -23.3874512, 23.4269409
42: -19.4673557, 2.2905121, -19.4692364, 2.3292537, -16.6780472, 16.6366425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=234, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1717
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1713
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1466
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: A, layer: 1, pos: 1741
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1756
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 980
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1556
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 1541
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1541
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1481
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1508
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1596
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1283
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1476
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1390
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 1527
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 685

## Relational analysis of IS_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4411221, upper bound: 13.5184798
time: 34.47 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4748338, upper bound: 13.5188200
time: 35.09 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -29.1247845, 4.1011305, -29.1967258, 4.0556912, -26.7834167, 26.9440079
1: -10.2239904, 6.7004299, -10.2261944, 6.6699276, -13.6015930, 13.6485252
2: -14.3351736, 4.4855585, -14.3999891, 4.4496031, -14.2760887, 14.4020805
3: -21.0323639, 0.8496993, -21.0310669, 0.7926273, -19.6161842, 19.6654892
4: -22.1649952, 3.3934307, -22.1935329, 3.3409562, -19.5575256, 19.6659813
5: -20.5834007, 5.8123064, -20.5908623, 5.7647524, -23.1176453, 23.1821594
6: -22.5350456, 3.2899647, -22.4934940, 3.2547035, -21.2884521, 21.2725983
7: -21.4861412, 4.0360355, -21.5028381, 3.9891310, -21.1448898, 21.2270813
8: -34.1772385, -4.0709629, -34.2137451, -4.1082511, -20.8479576, 20.9522552
9: -12.3323479, 16.7289257, -12.3331089, 16.6597538, -26.4217987, 26.5461884
10: -6.4285631, 20.7612190, -6.4145851, 20.7074490, -23.7332077, 23.8089142
11: -7.0605650, 14.0026379, -6.9755340, 13.9981527, -18.6884651, 18.5828438
12: 0.6065946, 35.4286499, 0.7346263, 35.4299927, -28.8706512, 28.7105637
13: -10.8579731, 24.3684216, -10.7653074, 24.2958717, -30.2205658, 30.2043419
14: -33.3440475, 10.4977131, -33.1789627, 10.4827156, -38.2062836, 38.0456543
15: -20.7629108, 0.3675172, -20.7493973, 0.3099771, -18.6541824, 18.6825180
16: -14.5714550, 7.5612020, -14.5722027, 7.4731665, -22.0446205, 22.1334038
17: -21.5038147, 18.8333569, -21.3359127, 18.7970200, -36.9067993, 36.7887268
18: -14.7751074, 9.5426493, -14.7198896, 9.5241795, -21.2144928, 21.1442490
19: -10.9166927, 6.8282971, -10.8890533, 6.8103404, -15.0177612, 15.0043831
20: -15.1555061, 4.9962244, -15.1153984, 4.9706421, -18.0336800, 18.0178604
21: -11.4584217, 9.6238785, -11.3899364, 9.6318321, -18.7326126, 18.6729317
22: -9.7022610, 7.9083652, -9.6188641, 7.9019747, -15.5185013, 15.4315224
23: -14.1364269, 7.3975515, -14.1095772, 7.3692703, -19.6882553, 19.6446457
24: -17.5576458, 6.2373886, -17.5027447, 6.2124138, -18.8288956, 18.7630997
25: -11.4817677, 10.1837006, -11.3814745, 10.1707706, -20.6158295, 20.5169983
26: -16.3194313, 9.8337498, -16.2304573, 9.8392963, -24.8766060, 24.7530212
27: -27.6281433, 0.8818684, -27.5571423, 0.8398161, -20.4380035, 20.3765259
28: -16.4179573, 7.3453164, -16.3655968, 7.3209777, -20.9373398, 20.8786850
29: -7.4789310, 10.4198132, -7.3729424, 10.4269428, -16.2923241, 16.1797256
30: -19.2584496, 7.4440613, -19.1740856, 7.4680834, -21.9638062, 21.8270302
31: -13.1941538, 9.5385475, -13.1475096, 9.5137386, -19.1243286, 19.0965118
32: -12.5480747, 9.4011593, -12.5159893, 9.3372593, -18.1837540, 18.2313423
33: -45.5000916, -9.2278404, -45.4602470, -9.3496208, -31.4306641, 31.5171814
34: -41.9964981, -13.9850111, -41.9701691, -14.0271311, -19.9300346, 19.9195900
35: -29.0774002, -2.4197016, -29.0458317, -2.4297216, -21.9228706, 21.8553085
36: -23.7725983, 3.8220212, -23.7351303, 3.7820632, -23.6701660, 23.6060982
37: -43.7098160, -4.5148845, -43.6623764, -4.5921335, -36.2983017, 36.2742081
38: -30.0530853, 1.4164710, -30.0293655, 1.3826900, -29.3923416, 29.2914886
39: -38.9902802, -3.8302684, -38.9732285, -3.9382730, -32.5387573, 32.6064224
40: -44.3867149, -12.2624912, -44.3571243, -12.3975439, -26.2990112, 26.4086304
41: -24.2986507, 5.1722684, -24.2743912, 5.0460858, -23.3768158, 23.4863281
42: -19.4797688, 2.3467607, -19.4467335, 2.2851434, -16.6418152, 16.6824379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=234, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: A, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 729
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1724
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1695
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1556
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1010
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 559
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1301
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1409
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1436
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1548
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1436
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1425
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1425
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1492
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1495
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1495
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 1527
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 685

## Relational analysis of IS_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4850488, upper bound: 13.4870954
time: 36.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5187566, upper bound: 13.4874276
time: 31.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -29.1254654, 4.1096759, -29.2435818, 4.0788832, -26.7998962, 27.0020828
1: -10.2241631, 6.7037897, -10.2422533, 6.6805634, -13.6098289, 13.6653328
2: -14.3357306, 4.4946456, -14.4486752, 4.4726934, -14.2916756, 14.4621735
3: -21.0328064, 0.8522141, -21.0524445, 0.8000238, -19.6229553, 19.6881943
4: -22.1650181, 3.4047360, -22.2398834, 3.3709788, -19.5756378, 19.7182961
5: -20.5838203, 5.8178458, -20.6224537, 5.7803431, -23.1287689, 23.2160339
6: -22.5484734, 3.2901955, -22.5267315, 3.2998085, -21.3479996, 21.2953682
7: -21.4871101, 4.0431719, -21.5395069, 4.0086060, -21.1588669, 21.2709579
8: -34.1776695, -4.0611792, -34.2568207, -4.0837731, -20.8629036, 21.0082092
9: -12.3326321, 16.7307968, -12.3442593, 16.6676769, -26.4292603, 26.5583496
10: -6.4280720, 20.7630157, -6.4162536, 20.7170067, -23.7389679, 23.8185997
11: -7.0626135, 14.0025406, -6.9852505, 14.0016670, -18.6943893, 18.5926971
12: 0.5927620, 35.4295120, 0.6983199, 35.4961853, -28.9541550, 28.7332458
13: -10.8584108, 24.3712807, -10.7859201, 24.3054276, -30.2304230, 30.2286606
14: -33.3456268, 10.5076294, -33.2219238, 10.5076504, -38.2293854, 38.1046600
15: -20.7630653, 0.3716881, -20.7645531, 0.3239288, -18.6658554, 18.7006454
16: -14.5724220, 7.5619268, -14.5833330, 7.4799948, -22.0524178, 22.1452599
17: -21.5049400, 18.8397961, -21.3568535, 18.8133106, -36.9215698, 36.8162918
18: -14.7757006, 9.5419903, -14.7240038, 9.5289326, -21.2276802, 21.1417999
19: -10.9178677, 6.8312707, -10.9098768, 6.8185449, -15.0251045, 15.0298996
20: -15.1570549, 4.9958286, -15.1269293, 4.9720387, -18.0392456, 18.0312996
21: -11.4601517, 9.6239643, -11.3987770, 9.6374950, -18.7417336, 18.6827469
22: -9.7030029, 7.9094987, -9.6234570, 7.9124727, -15.5295677, 15.4383831
23: -14.1377659, 7.3990517, -14.1232853, 7.3735666, -19.6956825, 19.6623230
24: -17.5583057, 6.2388177, -17.5176678, 6.2212462, -18.8381691, 18.7829933
25: -11.4823141, 10.1848059, -11.3870058, 10.1827297, -20.6244125, 20.5267029
26: -16.3205509, 9.8337955, -16.2449703, 9.8477039, -24.8945541, 24.7620926
27: -27.6299629, 0.8811412, -27.5654545, 0.8459563, -20.4528694, 20.3791122
28: -16.4198685, 7.3450303, -16.3772106, 7.3227730, -20.9436874, 20.9000893
29: -7.4817848, 10.4202023, -7.3830919, 10.4390974, -16.3020020, 16.1907349
30: -19.2627487, 7.4443588, -19.1877632, 7.4997168, -21.9975586, 21.8378067
31: -13.1950636, 9.5404663, -13.1648455, 9.5214128, -19.1320648, 19.1194305
32: -12.5569897, 9.4013243, -12.5407248, 9.3694201, -18.2278404, 18.2482796
33: -45.5076752, -9.2263327, -45.4833832, -9.3083210, -31.4726562, 31.5334320
34: -42.0069237, -13.9846544, -41.9957504, -13.9793606, -19.9912910, 19.9350357
35: -29.0822716, -2.4193144, -29.0599823, -2.4085650, -21.9470139, 21.8657036
36: -23.7760639, 3.8223395, -23.7477608, 3.7914777, -23.6830139, 23.6187019
37: -43.7174263, -4.5135899, -43.6830444, -4.5572114, -36.3395996, 36.2916260
38: -30.0539207, 1.4166923, -30.0372295, 1.3840740, -29.4023438, 29.2993088
39: -38.9903793, -3.8288503, -38.9782944, -3.9331808, -32.5487061, 32.6121979
40: -44.3948021, -12.2621002, -44.3780136, -12.3662243, -26.3403702, 26.4237976
41: -24.3071861, 5.1725817, -24.2971840, 5.0767736, -23.4155426, 23.5034180
42: -19.4913616, 2.3470163, -19.4765339, 2.3297348, -16.6965218, 16.7017784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=235, inp2_unstable=234, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=208, inp2_unstable=208, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=35, inp2_unstable=35, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 1700
type: B, layer: 1, pos: 1700
type: B, layer: 1, pos: 1704
type: B, layer: 1, pos: 1691
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1704
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 569
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 700
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 700
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1689
type: B, layer: 1, pos: 1757
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 965
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1790
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1717
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 889
type: A, layer: 1, pos: 889
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 701
type: A, layer: 1, pos: 701
type: A, layer: 1, pos: 1462
type: B, layer: 1, pos: 1462
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1654
type: B, layer: 1, pos: 1654
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 853
type: A, layer: 1, pos: 853
type: B, layer: 1, pos: 1789
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 1466
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1613
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1771
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 571
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 570
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1706
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 688
type: A, layer: 1, pos: 688
type: B, layer: 1, pos: 1593
type: A, layer: 1, pos: 1593
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1773
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1703
type: B, layer: 1, pos: 1716
type: A, layer: 1, pos: 1716
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1677
type: B, layer: 1, pos: 1741
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 1677
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1755
type: A, layer: 1, pos: 521
type: B, layer: 1, pos: 521
type: A, layer: 1, pos: 1427
type: B, layer: 1, pos: 1427
type: A, layer: 1, pos: 1703
type: A, layer: 1, pos: 1787
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1676
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1666
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: B, layer: 1, pos: 1565
type: A, layer: 1, pos: 1565
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1740
type: B, layer: 1, pos: 1740
type: A, layer: 1, pos: 1334
type: B, layer: 1, pos: 1334
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1350
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1350
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1451
type: A, layer: 1, pos: 1451
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1728
type: A, layer: 1, pos: 1728
type: B, layer: 1, pos: 1732
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 1732
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1479
type: A, layer: 1, pos: 1479
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1731
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1731
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1712
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1756
type: B, layer: 1, pos: 1712
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 728
type: B, layer: 1, pos: 1727
type: A, layer: 1, pos: 1727
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 980
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 728
type: A, layer: 1, pos: 1573
type: B, layer: 1, pos: 1650
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 1572
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1406
type: B, layer: 1, pos: 1406
type: A, layer: 1, pos: 1724
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 572
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1660
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1660
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1273
type: B, layer: 1, pos: 1273
type: A, layer: 1, pos: 979
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 979
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 703
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 687
type: B, layer: 1, pos: 687
type: A, layer: 1, pos: 1635
type: B, layer: 1, pos: 1635
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1011
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1361
type: A, layer: 1, pos: 1772
type: B, layer: 1, pos: 1772
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1695
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 1739
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1396
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1396
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1304
type: B, layer: 1, pos: 1304
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1550
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 1010
type: A, layer: 1, pos: 559
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1747
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 1644
type: B, layer: 1, pos: 1644
type: A, layer: 1, pos: 1377
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1301
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1301
type: B, layer: 1, pos: 1567
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 802
type: B, layer: 1, pos: 802
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 785
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1697
type: B, layer: 1, pos: 785
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1409
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1409
type: A, layer: 1, pos: 1345
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1558
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1481
type: A, layer: 1, pos: 1481
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1566
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1629
type: A, layer: 1, pos: 1629
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 1423
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 994
type: B, layer: 1, pos: 994
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1324
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1411
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 806
type: A, layer: 1, pos: 1425
type: B, layer: 1, pos: 1283
type: A, layer: 1, pos: 1370
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1317
type: B, layer: 1, pos: 1425
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1283
type: B, layer: 1, pos: 1476
type: A, layer: 1, pos: 1596
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 806
type: A, layer: 1, pos: 1492
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 1317
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1495
type: A, layer: 1, pos: 1495
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1282
type: A, layer: 1, pos: 1393
type: B, layer: 1, pos: 1393
type: B, layer: 1, pos: 1525
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1282
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1527
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1319
type: B, layer: 1, pos: 1527
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 1391
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1319

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 685

## Relational analysis of IS_A2_B2_A2_B2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.4851110, upper bound: 13.5184797
time: 42.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 12, lower bound: -13.5188204, upper bound: 13.5188200
time: 27.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 72.69 seconds
IS_A1_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 72.69
Output dim: 12, lower bound: -13.4850487, upper bound: 13.4589728
IS_A1_B2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 72.69
Output dim: 12, lower bound: -13.5187566, upper bound: 13.4592966
IS_A1_B2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 72.69
Output dim: 12, lower bound: -13.4851110, upper bound: 13.4903430
IS_A1_B2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 72.69
Output dim: 12, lower bound: -13.5188204, upper bound: 13.4906742
IS_A2_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 72.69
Output dim: 12, lower bound: -13.4411221, upper bound: 13.5184798
IS_A2_B2_A1_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 72.69
Output dim: 12, lower bound: -13.4748338, upper bound: 13.5188200
IS_A2_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 72.69
Output dim: 12, lower bound: -13.4850488, upper bound: 13.4870954
IS_A2_B2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 6, time: 72.69
Output dim: 12, lower bound: -13.5187566, upper bound: 13.4874276
IS_A2_B2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 6, time: 72.69
Output dim: 12, lower bound: -13.4851110, upper bound: 13.5184797
IS_A2_B2_A2_B2_B2_B2, status: Status.VERIFIED, split count: 6, time: 72.69
Output dim: 12, lower bound: -13.5188204, upper bound: 13.5188200

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 49.64 + 1737.80 = 1787.43 seconds
