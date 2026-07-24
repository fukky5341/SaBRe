## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 148.163763758
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-194.3560638, 123.7643051, -194.3560638, 123.7643051, -318.1203613, 318.1203613)
1: (-112.6573334, 98.2656555, -112.6573334, 98.2656555, -210.9229889, 210.9229889)
2: (-99.1251068, 90.8807373, -99.1251068, 90.8807373, -190.0058441, 190.0058441)
3: (-115.3852997, 118.0533752, -115.3852997, 118.0533752, -233.4386597, 233.4386597)
4: (-122.4373779, 112.1905746, -122.4373779, 112.1905746, -234.6279297, 234.6279602)
5: (-111.2773132, 118.3461990, -111.2773132, 118.3461990, -229.6235046, 229.6234894)
6: (-137.7839966, 120.7834320, -137.7839966, 120.7834320, -258.5674133, 258.5674438)
7: (-135.4126282, 119.6500015, -135.4126282, 119.6500015, -255.0626221, 255.0626221)
8: (-127.4194870, 132.8383179, -127.4194870, 132.8383179, -260.2578125, 260.2578125)
9: (-119.2357254, 110.2499542, -119.2357254, 110.2499542, -229.4856873, 229.4856873)
10: (-174.5004272, 163.4277802, -174.5004272, 163.4277802, -337.9282227, 337.9281921)
11: (-166.8539734, 129.6512146, -166.8539734, 129.6512146, -296.5051880, 296.5051880)
12: (-156.0662231, 146.7076263, -156.0662231, 146.7076263, -302.7738647, 302.7738342)
13: (-161.5158081, 168.4806824, -161.5158081, 168.4806824, -329.9964905, 329.9964905)
14: (-238.9123383, 147.0448761, -238.9123383, 147.0448761, -385.9572144, 385.9572144)
15: (-139.2399445, 112.0155334, -139.2399445, 112.0155334, -251.2554779, 251.2554779)
16: (-182.3288879, 134.4130859, -182.3288879, 134.4130859, -316.7419739, 316.7419739)
17: (-249.6140747, 196.0419617, -249.6140747, 196.0419617, -445.6560059, 445.6560059)
18: (-152.5888062, 126.4845505, -152.5888062, 126.4845505, -279.0733643, 279.0733643)
19: (-117.1988754, 74.9498978, -117.1988754, 74.9498978, -192.1487732, 192.1487732)
20: (-110.0395050, 92.9520416, -110.0395050, 92.9520416, -202.9915466, 202.9915466)
21: (-149.0595856, 100.4649124, -149.0595856, 100.4649124, -249.5244598, 249.5244598)
22: (-158.1970825, 108.0006943, -158.1970825, 108.0006943, -266.1977539, 266.1977844)
23: (-120.2197723, 95.0907288, -120.2197723, 95.0907288, -215.3104858, 215.3104706)
24: (-147.7882538, 108.8617859, -147.7882538, 108.8617859, -256.6500244, 256.6500244)
25: (-124.3082123, 106.1959991, -124.3082123, 106.1959991, -230.5042114, 230.5042114)
26: (-171.0725708, 148.6687469, -171.0725708, 148.6687469, -319.7413330, 319.7413330)
27: (-155.3504333, 110.9619522, -155.3504333, 110.9619522, -266.3123779, 266.3123779)
28: (-116.1600723, 100.0528336, -116.1600723, 100.0528336, -216.2129059, 216.2129059)
29: (-174.3669586, 115.8534164, -174.3669586, 115.8534164, -290.2203674, 290.2203674)
30: (-146.8119354, 125.6781540, -146.8119354, 125.6781540, -272.4900513, 272.4900513)
31: (-149.9770966, 101.8592148, -149.9770966, 101.8592148, -251.8362885, 251.8363037)
32: (-146.5981750, 117.1200714, -146.5981750, 117.1200714, -263.7182617, 263.7182617)
33: (-192.7234802, 159.2073059, -192.7234802, 159.2073059, -351.9307861, 351.9307861)
34: (-161.5019989, 124.2549896, -161.5019989, 124.2549896, -285.7569885, 285.7569885)
35: (-163.2583618, 126.8004913, -163.2583618, 126.8004913, -290.0588379, 290.0588379)
36: (-154.2919922, 125.1236725, -154.2919922, 125.1236725, -279.4156189, 279.4156494)
37: (-223.7451019, 141.9507141, -223.7451019, 141.9507141, -365.6958008, 365.6958008)
38: (-190.3866577, 154.8811646, -190.3866577, 154.8811646, -345.2678223, 345.2678223)
39: (-226.4278870, 154.2048950, -226.4278870, 154.2048950, -380.6327820, 380.6327820)
40: (-192.2375641, 133.5492554, -192.2375641, 133.5492554, -325.7868042, 325.7868042)
41: (-144.3735046, 111.2714844, -144.3735046, 111.2714844, -255.6449738, 255.6449738)
42: (-111.1141891, 108.9842834, -111.1141891, 108.9842834, -220.0984344, 220.0984497)

## BASE Result
execution time: IAR + LP analysis = 2.76 + 748.15 = 750.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -151.4425369, upper bound: 151.4425371


# Binary Search by BASE starts (time budget: 17249.09 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=233.43865966796875
rel_dist={3: [-146.7491330209824, 146.7491331173323]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=233.43865966796875
rel_dist={3: [-149.40356280758328, 149.4035629176625]}

## Binary search (step 2) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=7, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=233.43865966796875
rel_dist={3: [-147.729598459751, 147.72959856394056]}

## Binary search (step 3) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start
Binary search (step 3): status=Status.UNKNOWN, k_low=8, k_high=8, k_mid=8, eps_mid=0.0312500, abs_max=233.43865966796875
rel_dist={3: [-148.6090126930012, 148.60901277819403]}

## Binary Search Result
Binary search time: 3712.41 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.02734375


# Relational Split (RS_dual_Z) starts
Time budget: 13536.68 seconds

## Binary search (step 0) starts
Candidate k: 10, corresponding eps: 0.0390625


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1175
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1225
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1162
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1178
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 325
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 305
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 309
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 324
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 264
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 307
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1097
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 306
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 336
type: RSZ, layer: 1, pos: 308
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 353
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1792
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 291
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 321
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 256
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 320
type: RSZ, layer: 1, pos: 292
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 352
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 257
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1184

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.0879944, upper bound: 150.1290825
time: 221.16 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.1290824, upper bound: 150.0879945
time: 327.54 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 548.85 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 548.85
Output dim: 3, lower bound: -150.0879944, upper bound: 150.1290825
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 548.85
Output dim: 3, lower bound: -150.1290824, upper bound: 150.0879945

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -194.3560638, 123.7643051, -194.3560638, 123.7643051, -318.1203613, 318.1203613
1: -112.6573334, 98.2656555, -112.6573334, 98.2656555, -210.9229889, 210.9229889
2: -99.1251068, 90.8807373, -99.1251068, 90.8807373, -190.0058441, 190.0058441
3: -115.3852997, 118.0533752, -115.3852997, 118.0533752, -233.4386597, 233.4386597
4: -122.4373779, 112.1905746, -122.4373779, 112.1905746, -234.6279297, 234.6279602
5: -111.2773132, 118.3461990, -111.2773132, 118.3461990, -229.6235046, 229.6234894
6: -137.7839966, 120.7834320, -137.7839966, 120.7834320, -258.5674133, 258.5674438
7: -135.4126282, 119.6500015, -135.4126282, 119.6500015, -255.0626221, 255.0626221
8: -127.4194870, 132.8383179, -127.4194870, 132.8383179, -260.2578125, 260.2578125
9: -119.2357254, 110.2499542, -119.2357254, 110.2499542, -229.4856873, 229.4856873
10: -174.5004272, 163.4277802, -174.5004272, 163.4277802, -337.9282227, 337.9281921
11: -166.8539734, 129.6512146, -166.8539734, 129.6512146, -296.5051880, 296.5051880
12: -156.0662231, 146.7076263, -156.0662231, 146.7076263, -302.7738647, 302.7738342
13: -161.5158081, 168.4806824, -161.5158081, 168.4806824, -329.9964905, 329.9964905
14: -238.9123383, 147.0448761, -238.9123383, 147.0448761, -385.9572144, 385.9572144
15: -139.2399445, 112.0155334, -139.2399445, 112.0155334, -251.2554779, 251.2554779
16: -182.3288879, 134.4130859, -182.3288879, 134.4130859, -316.7419739, 316.7419739
17: -249.6140747, 196.0419617, -249.6140747, 196.0419617, -445.6560059, 445.6560059
18: -152.5888062, 126.4845505, -152.5888062, 126.4845505, -279.0733643, 279.0733643
19: -117.1988754, 74.9498978, -117.1988754, 74.9498978, -192.1487732, 192.1487732
20: -110.0395050, 92.9520416, -110.0395050, 92.9520416, -202.9915466, 202.9915466
21: -149.0595856, 100.4649124, -149.0595856, 100.4649124, -249.5244598, 249.5244598
22: -158.1970825, 108.0006943, -158.1970825, 108.0006943, -266.1977539, 266.1977844
23: -120.2197723, 95.0907288, -120.2197723, 95.0907288, -215.3104858, 215.3104706
24: -147.7882538, 108.8617859, -147.7882538, 108.8617859, -256.6500244, 256.6500244
25: -124.3082123, 106.1959991, -124.3082123, 106.1959991, -230.5042114, 230.5042114
26: -171.0725708, 148.6687469, -171.0725708, 148.6687469, -319.7413330, 319.7413330
27: -155.3504333, 110.9619522, -155.3504333, 110.9619522, -266.3123779, 266.3123779
28: -116.1600723, 100.0528336, -116.1600723, 100.0528336, -216.2129059, 216.2129059
29: -174.3669586, 115.8534164, -174.3669586, 115.8534164, -290.2203674, 290.2203674
30: -146.8119354, 125.6781540, -146.8119354, 125.6781540, -272.4900513, 272.4900513
31: -149.9770966, 101.8592148, -149.9770966, 101.8592148, -251.8362885, 251.8363037
32: -146.5981750, 117.1200714, -146.5981750, 117.1200714, -263.7182617, 263.7182617
33: -192.7234802, 159.2073059, -192.7234802, 159.2073059, -351.9307861, 351.9307861
34: -161.5019989, 124.2549896, -161.5019989, 124.2549896, -285.7569885, 285.7569885
35: -163.2583618, 126.8004913, -163.2583618, 126.8004913, -290.0588379, 290.0588379
36: -154.2919922, 125.1236725, -154.2919922, 125.1236725, -279.4156189, 279.4156494
37: -223.7451019, 141.9507141, -223.7451019, 141.9507141, -365.6958008, 365.6958008
38: -190.3866577, 154.8811646, -190.3866577, 154.8811646, -345.2678223, 345.2678223
39: -226.4278870, 154.2048950, -226.4278870, 154.2048950, -380.6327820, 380.6327820
40: -192.2375641, 133.5492554, -192.2375641, 133.5492554, -325.7868042, 325.7868042
41: -144.3735046, 111.2714844, -144.3735046, 111.2714844, -255.6449738, 255.6449738
42: -111.1141891, 108.9842834, -111.1141891, 108.9842834, -220.0984344, 220.0984497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=698, inp2_unstable=698, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1175
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1225
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1162
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1178
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 325
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 305
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 309
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 324
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 264
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 307
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1097
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 306
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 336
type: RSZ, layer: 1, pos: 308
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 353
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1792
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 291
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 321
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 256
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 320
type: RSZ, layer: 1, pos: 292
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 352
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 257
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1184

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.0819121, upper bound: 150.0060461
time: 403.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -149.9649171, upper bound: 150.1229712
time: 260.39 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -194.3560638, 123.7643051, -194.3560638, 123.7643051, -318.1203613, 318.1203613
1: -112.6573334, 98.2656555, -112.6573334, 98.2656555, -210.9229889, 210.9229889
2: -99.1251068, 90.8807373, -99.1251068, 90.8807373, -190.0058441, 190.0058441
3: -115.3852997, 118.0533752, -115.3852997, 118.0533752, -233.4386597, 233.4386597
4: -122.4373779, 112.1905746, -122.4373779, 112.1905746, -234.6279297, 234.6279602
5: -111.2773132, 118.3461990, -111.2773132, 118.3461990, -229.6235046, 229.6234894
6: -137.7839966, 120.7834320, -137.7839966, 120.7834320, -258.5674133, 258.5674438
7: -135.4126282, 119.6500015, -135.4126282, 119.6500015, -255.0626221, 255.0626221
8: -127.4194870, 132.8383179, -127.4194870, 132.8383179, -260.2578125, 260.2578125
9: -119.2357254, 110.2499542, -119.2357254, 110.2499542, -229.4856873, 229.4856873
10: -174.5004272, 163.4277802, -174.5004272, 163.4277802, -337.9282227, 337.9281921
11: -166.8539734, 129.6512146, -166.8539734, 129.6512146, -296.5051880, 296.5051880
12: -156.0662231, 146.7076263, -156.0662231, 146.7076263, -302.7738647, 302.7738342
13: -161.5158081, 168.4806824, -161.5158081, 168.4806824, -329.9964905, 329.9964905
14: -238.9123383, 147.0448761, -238.9123383, 147.0448761, -385.9572144, 385.9572144
15: -139.2399445, 112.0155334, -139.2399445, 112.0155334, -251.2554779, 251.2554779
16: -182.3288879, 134.4130859, -182.3288879, 134.4130859, -316.7419739, 316.7419739
17: -249.6140747, 196.0419617, -249.6140747, 196.0419617, -445.6560059, 445.6560059
18: -152.5888062, 126.4845505, -152.5888062, 126.4845505, -279.0733643, 279.0733643
19: -117.1988754, 74.9498978, -117.1988754, 74.9498978, -192.1487732, 192.1487732
20: -110.0395050, 92.9520416, -110.0395050, 92.9520416, -202.9915466, 202.9915466
21: -149.0595856, 100.4649124, -149.0595856, 100.4649124, -249.5244598, 249.5244598
22: -158.1970825, 108.0006943, -158.1970825, 108.0006943, -266.1977539, 266.1977844
23: -120.2197723, 95.0907288, -120.2197723, 95.0907288, -215.3104858, 215.3104706
24: -147.7882538, 108.8617859, -147.7882538, 108.8617859, -256.6500244, 256.6500244
25: -124.3082123, 106.1959991, -124.3082123, 106.1959991, -230.5042114, 230.5042114
26: -171.0725708, 148.6687469, -171.0725708, 148.6687469, -319.7413330, 319.7413330
27: -155.3504333, 110.9619522, -155.3504333, 110.9619522, -266.3123779, 266.3123779
28: -116.1600723, 100.0528336, -116.1600723, 100.0528336, -216.2129059, 216.2129059
29: -174.3669586, 115.8534164, -174.3669586, 115.8534164, -290.2203674, 290.2203674
30: -146.8119354, 125.6781540, -146.8119354, 125.6781540, -272.4900513, 272.4900513
31: -149.9770966, 101.8592148, -149.9770966, 101.8592148, -251.8362885, 251.8363037
32: -146.5981750, 117.1200714, -146.5981750, 117.1200714, -263.7182617, 263.7182617
33: -192.7234802, 159.2073059, -192.7234802, 159.2073059, -351.9307861, 351.9307861
34: -161.5019989, 124.2549896, -161.5019989, 124.2549896, -285.7569885, 285.7569885
35: -163.2583618, 126.8004913, -163.2583618, 126.8004913, -290.0588379, 290.0588379
36: -154.2919922, 125.1236725, -154.2919922, 125.1236725, -279.4156189, 279.4156494
37: -223.7451019, 141.9507141, -223.7451019, 141.9507141, -365.6958008, 365.6958008
38: -190.3866577, 154.8811646, -190.3866577, 154.8811646, -345.2678223, 345.2678223
39: -226.4278870, 154.2048950, -226.4278870, 154.2048950, -380.6327820, 380.6327820
40: -192.2375641, 133.5492554, -192.2375641, 133.5492554, -325.7868042, 325.7868042
41: -144.3735046, 111.2714844, -144.3735046, 111.2714844, -255.6449738, 255.6449738
42: -111.1141891, 108.9842834, -111.1141891, 108.9842834, -220.0984344, 220.0984497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=698, inp2_unstable=698, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1175
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1225
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1162
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1178
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 325
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 305
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 309
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 324
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 264
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 307
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1097
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 306
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 336
type: RSZ, layer: 1, pos: 308
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 353
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1792
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 291
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 321
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 256
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 320
type: RSZ, layer: 1, pos: 292
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 352
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 257
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1184

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.1229711, upper bound: 149.9649172
time: 436.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.0060460, upper bound: 150.0819122
time: 875.87 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 1314.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1314.86
Output dim: 3, lower bound: -150.0819121, upper bound: 150.0060461
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1314.86
Output dim: 3, lower bound: -149.9649171, upper bound: 150.1229712
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 1314.86
Output dim: 3, lower bound: -150.1229711, upper bound: 149.9649172
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 1314.86
Output dim: 3, lower bound: -150.0060460, upper bound: 150.0819122

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -194.3560638, 123.7643051, -194.3560638, 123.7643051, -318.1203613, 318.1203613
1: -112.6573334, 98.2656555, -112.6573334, 98.2656555, -210.9229889, 210.9229889
2: -99.1251068, 90.8807373, -99.1251068, 90.8807373, -190.0058441, 190.0058441
3: -115.3852997, 118.0533752, -115.3852997, 118.0533752, -233.4386597, 233.4386597
4: -122.4373779, 112.1905746, -122.4373779, 112.1905746, -234.6279297, 234.6279602
5: -111.2773132, 118.3461990, -111.2773132, 118.3461990, -229.6235046, 229.6234894
6: -137.7839966, 120.7834320, -137.7839966, 120.7834320, -258.5674133, 258.5674438
7: -135.4126282, 119.6500015, -135.4126282, 119.6500015, -255.0626221, 255.0626221
8: -127.4194870, 132.8383179, -127.4194870, 132.8383179, -260.2578125, 260.2578125
9: -119.2357254, 110.2499542, -119.2357254, 110.2499542, -229.4856873, 229.4856873
10: -174.5004272, 163.4277802, -174.5004272, 163.4277802, -337.9282227, 337.9281921
11: -166.8539734, 129.6512146, -166.8539734, 129.6512146, -296.5051880, 296.5051880
12: -156.0662231, 146.7076263, -156.0662231, 146.7076263, -302.7738647, 302.7738342
13: -161.5158081, 168.4806824, -161.5158081, 168.4806824, -329.9964905, 329.9964905
14: -238.9123383, 147.0448761, -238.9123383, 147.0448761, -385.9572144, 385.9572144
15: -139.2399445, 112.0155334, -139.2399445, 112.0155334, -251.2554779, 251.2554779
16: -182.3288879, 134.4130859, -182.3288879, 134.4130859, -316.7419739, 316.7419739
17: -249.6140747, 196.0419617, -249.6140747, 196.0419617, -445.6560059, 445.6560059
18: -152.5888062, 126.4845505, -152.5888062, 126.4845505, -279.0733643, 279.0733643
19: -117.1988754, 74.9498978, -117.1988754, 74.9498978, -192.1487732, 192.1487732
20: -110.0395050, 92.9520416, -110.0395050, 92.9520416, -202.9915466, 202.9915466
21: -149.0595856, 100.4649124, -149.0595856, 100.4649124, -249.5244598, 249.5244598
22: -158.1970825, 108.0006943, -158.1970825, 108.0006943, -266.1977539, 266.1977844
23: -120.2197723, 95.0907288, -120.2197723, 95.0907288, -215.3104858, 215.3104706
24: -147.7882538, 108.8617859, -147.7882538, 108.8617859, -256.6500244, 256.6500244
25: -124.3082123, 106.1959991, -124.3082123, 106.1959991, -230.5042114, 230.5042114
26: -171.0725708, 148.6687469, -171.0725708, 148.6687469, -319.7413330, 319.7413330
27: -155.3504333, 110.9619522, -155.3504333, 110.9619522, -266.3123779, 266.3123779
28: -116.1600723, 100.0528336, -116.1600723, 100.0528336, -216.2129059, 216.2129059
29: -174.3669586, 115.8534164, -174.3669586, 115.8534164, -290.2203674, 290.2203674
30: -146.8119354, 125.6781540, -146.8119354, 125.6781540, -272.4900513, 272.4900513
31: -149.9770966, 101.8592148, -149.9770966, 101.8592148, -251.8362885, 251.8363037
32: -146.5981750, 117.1200714, -146.5981750, 117.1200714, -263.7182617, 263.7182617
33: -192.7234802, 159.2073059, -192.7234802, 159.2073059, -351.9307861, 351.9307861
34: -161.5019989, 124.2549896, -161.5019989, 124.2549896, -285.7569885, 285.7569885
35: -163.2583618, 126.8004913, -163.2583618, 126.8004913, -290.0588379, 290.0588379
36: -154.2919922, 125.1236725, -154.2919922, 125.1236725, -279.4156189, 279.4156494
37: -223.7451019, 141.9507141, -223.7451019, 141.9507141, -365.6958008, 365.6958008
38: -190.3866577, 154.8811646, -190.3866577, 154.8811646, -345.2678223, 345.2678223
39: -226.4278870, 154.2048950, -226.4278870, 154.2048950, -380.6327820, 380.6327820
40: -192.2375641, 133.5492554, -192.2375641, 133.5492554, -325.7868042, 325.7868042
41: -144.3735046, 111.2714844, -144.3735046, 111.2714844, -255.6449738, 255.6449738
42: -111.1141891, 108.9842834, -111.1141891, 108.9842834, -220.0984344, 220.0984497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=698, inp2_unstable=698, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1175
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1225
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1162
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1178
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 325
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 305
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 309
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 324
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 264
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 307
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1097
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 306
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 336
type: RSZ, layer: 1, pos: 308
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 353
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1792
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 291
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 321
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 256
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 320
type: RSZ, layer: 1, pos: 292
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 352
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 257
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1184

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.0384041, upper bound: 150.0013508
time: 739.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -150.0753720, upper bound: 149.9433539
time: 210.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 952.52 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 952.52
Output dim: 3, lower bound: -150.0384041, upper bound: 150.0013508
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 952.52
Output dim: 3, lower bound: -150.0753720, upper bound: 149.9433539
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 952.52
Output dim: 3, lower bound: -149.9649171, upper bound: 150.1229712
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 952.52
Output dim: 3, lower bound: -150.1229711, upper bound: 149.9649172
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 952.52
Output dim: 3, lower bound: -150.0060460, upper bound: 150.0819122
Binary search (step 0): status=Status.UNKNOWN, k_low=8, k_high=12, k_mid=10, eps_mid=0.0390625, abs_max=233.43865966796875
rel_dist={3: [-150.13758384939092, 150.13758388525406]}

## Binary search (step 1) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1175
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1225
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1162
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1178
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 325
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 305
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 309
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 324
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 264
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 307
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1097
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 306
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 336
type: RSZ, layer: 1, pos: 308
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 353
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1792
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 291
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 321
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 256
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 320
type: RSZ, layer: 1, pos: 292
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 352
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 257
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1184

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -148.5611855, upper bound: 148.6007015
time: 1104.11 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -148.6007014, upper bound: 148.5611856
time: 715.68 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1819.94 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1819.94
Output dim: 3, lower bound: -148.5611855, upper bound: 148.6007015
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1819.94
Output dim: 3, lower bound: -148.6007014, upper bound: 148.5611856

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -194.3560638, 123.7643051, -194.3560638, 123.7643051, -318.1203613, 318.1203613
1: -112.6573334, 98.2656555, -112.6573334, 98.2656555, -210.9229889, 210.9229889
2: -99.1251068, 90.8807373, -99.1251068, 90.8807373, -190.0058441, 190.0058441
3: -115.3852997, 118.0533752, -115.3852997, 118.0533752, -233.4386597, 233.4386597
4: -122.4373779, 112.1905746, -122.4373779, 112.1905746, -234.6279297, 234.6279602
5: -111.2773132, 118.3461990, -111.2773132, 118.3461990, -229.6235046, 229.6234894
6: -137.7839966, 120.7834320, -137.7839966, 120.7834320, -258.5674133, 258.5674438
7: -135.4126282, 119.6500015, -135.4126282, 119.6500015, -255.0626221, 255.0626221
8: -127.4194870, 132.8383179, -127.4194870, 132.8383179, -260.2578125, 260.2578125
9: -119.2357254, 110.2499542, -119.2357254, 110.2499542, -229.4856873, 229.4856873
10: -174.5004272, 163.4277802, -174.5004272, 163.4277802, -337.9282227, 337.9281921
11: -166.8539734, 129.6512146, -166.8539734, 129.6512146, -296.5051880, 296.5051880
12: -156.0662231, 146.7076263, -156.0662231, 146.7076263, -302.7738647, 302.7738342
13: -161.5158081, 168.4806824, -161.5158081, 168.4806824, -329.9964905, 329.9964905
14: -238.9123383, 147.0448761, -238.9123383, 147.0448761, -385.9572144, 385.9572144
15: -139.2399445, 112.0155334, -139.2399445, 112.0155334, -251.2554779, 251.2554779
16: -182.3288879, 134.4130859, -182.3288879, 134.4130859, -316.7419739, 316.7419739
17: -249.6140747, 196.0419617, -249.6140747, 196.0419617, -445.6560059, 445.6560059
18: -152.5888062, 126.4845505, -152.5888062, 126.4845505, -279.0733643, 279.0733643
19: -117.1988754, 74.9498978, -117.1988754, 74.9498978, -192.1487732, 192.1487732
20: -110.0395050, 92.9520416, -110.0395050, 92.9520416, -202.9915466, 202.9915466
21: -149.0595856, 100.4649124, -149.0595856, 100.4649124, -249.5244598, 249.5244598
22: -158.1970825, 108.0006943, -158.1970825, 108.0006943, -266.1977539, 266.1977844
23: -120.2197723, 95.0907288, -120.2197723, 95.0907288, -215.3104858, 215.3104706
24: -147.7882538, 108.8617859, -147.7882538, 108.8617859, -256.6500244, 256.6500244
25: -124.3082123, 106.1959991, -124.3082123, 106.1959991, -230.5042114, 230.5042114
26: -171.0725708, 148.6687469, -171.0725708, 148.6687469, -319.7413330, 319.7413330
27: -155.3504333, 110.9619522, -155.3504333, 110.9619522, -266.3123779, 266.3123779
28: -116.1600723, 100.0528336, -116.1600723, 100.0528336, -216.2129059, 216.2129059
29: -174.3669586, 115.8534164, -174.3669586, 115.8534164, -290.2203674, 290.2203674
30: -146.8119354, 125.6781540, -146.8119354, 125.6781540, -272.4900513, 272.4900513
31: -149.9770966, 101.8592148, -149.9770966, 101.8592148, -251.8362885, 251.8363037
32: -146.5981750, 117.1200714, -146.5981750, 117.1200714, -263.7182617, 263.7182617
33: -192.7234802, 159.2073059, -192.7234802, 159.2073059, -351.9307861, 351.9307861
34: -161.5019989, 124.2549896, -161.5019989, 124.2549896, -285.7569885, 285.7569885
35: -163.2583618, 126.8004913, -163.2583618, 126.8004913, -290.0588379, 290.0588379
36: -154.2919922, 125.1236725, -154.2919922, 125.1236725, -279.4156189, 279.4156494
37: -223.7451019, 141.9507141, -223.7451019, 141.9507141, -365.6958008, 365.6958008
38: -190.3866577, 154.8811646, -190.3866577, 154.8811646, -345.2678223, 345.2678223
39: -226.4278870, 154.2048950, -226.4278870, 154.2048950, -380.6327820, 380.6327820
40: -192.2375641, 133.5492554, -192.2375641, 133.5492554, -325.7868042, 325.7868042
41: -144.3735046, 111.2714844, -144.3735046, 111.2714844, -255.6449738, 255.6449738
42: -111.1141891, 108.9842834, -111.1141891, 108.9842834, -220.0984344, 220.0984497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=698, inp2_unstable=698, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1175
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1225
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1162
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1178
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 325
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 305
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 309
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 324
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 264
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 307
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1097
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 306
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 336
type: RSZ, layer: 1, pos: 308
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 353
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1792
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 291
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 321
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 256
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 320
type: RSZ, layer: 1, pos: 292
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 352
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 257
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1184

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -148.5570243, upper bound: 148.4807855
time: 281.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -148.4407533, upper bound: 148.5965557
time: 594.28 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -194.3560638, 123.7643051, -194.3560638, 123.7643051, -318.1203613, 318.1203613
1: -112.6573334, 98.2656555, -112.6573334, 98.2656555, -210.9229889, 210.9229889
2: -99.1251068, 90.8807373, -99.1251068, 90.8807373, -190.0058441, 190.0058441
3: -115.3852997, 118.0533752, -115.3852997, 118.0533752, -233.4386597, 233.4386597
4: -122.4373779, 112.1905746, -122.4373779, 112.1905746, -234.6279297, 234.6279602
5: -111.2773132, 118.3461990, -111.2773132, 118.3461990, -229.6235046, 229.6234894
6: -137.7839966, 120.7834320, -137.7839966, 120.7834320, -258.5674133, 258.5674438
7: -135.4126282, 119.6500015, -135.4126282, 119.6500015, -255.0626221, 255.0626221
8: -127.4194870, 132.8383179, -127.4194870, 132.8383179, -260.2578125, 260.2578125
9: -119.2357254, 110.2499542, -119.2357254, 110.2499542, -229.4856873, 229.4856873
10: -174.5004272, 163.4277802, -174.5004272, 163.4277802, -337.9282227, 337.9281921
11: -166.8539734, 129.6512146, -166.8539734, 129.6512146, -296.5051880, 296.5051880
12: -156.0662231, 146.7076263, -156.0662231, 146.7076263, -302.7738647, 302.7738342
13: -161.5158081, 168.4806824, -161.5158081, 168.4806824, -329.9964905, 329.9964905
14: -238.9123383, 147.0448761, -238.9123383, 147.0448761, -385.9572144, 385.9572144
15: -139.2399445, 112.0155334, -139.2399445, 112.0155334, -251.2554779, 251.2554779
16: -182.3288879, 134.4130859, -182.3288879, 134.4130859, -316.7419739, 316.7419739
17: -249.6140747, 196.0419617, -249.6140747, 196.0419617, -445.6560059, 445.6560059
18: -152.5888062, 126.4845505, -152.5888062, 126.4845505, -279.0733643, 279.0733643
19: -117.1988754, 74.9498978, -117.1988754, 74.9498978, -192.1487732, 192.1487732
20: -110.0395050, 92.9520416, -110.0395050, 92.9520416, -202.9915466, 202.9915466
21: -149.0595856, 100.4649124, -149.0595856, 100.4649124, -249.5244598, 249.5244598
22: -158.1970825, 108.0006943, -158.1970825, 108.0006943, -266.1977539, 266.1977844
23: -120.2197723, 95.0907288, -120.2197723, 95.0907288, -215.3104858, 215.3104706
24: -147.7882538, 108.8617859, -147.7882538, 108.8617859, -256.6500244, 256.6500244
25: -124.3082123, 106.1959991, -124.3082123, 106.1959991, -230.5042114, 230.5042114
26: -171.0725708, 148.6687469, -171.0725708, 148.6687469, -319.7413330, 319.7413330
27: -155.3504333, 110.9619522, -155.3504333, 110.9619522, -266.3123779, 266.3123779
28: -116.1600723, 100.0528336, -116.1600723, 100.0528336, -216.2129059, 216.2129059
29: -174.3669586, 115.8534164, -174.3669586, 115.8534164, -290.2203674, 290.2203674
30: -146.8119354, 125.6781540, -146.8119354, 125.6781540, -272.4900513, 272.4900513
31: -149.9770966, 101.8592148, -149.9770966, 101.8592148, -251.8362885, 251.8363037
32: -146.5981750, 117.1200714, -146.5981750, 117.1200714, -263.7182617, 263.7182617
33: -192.7234802, 159.2073059, -192.7234802, 159.2073059, -351.9307861, 351.9307861
34: -161.5019989, 124.2549896, -161.5019989, 124.2549896, -285.7569885, 285.7569885
35: -163.2583618, 126.8004913, -163.2583618, 126.8004913, -290.0588379, 290.0588379
36: -154.2919922, 125.1236725, -154.2919922, 125.1236725, -279.4156189, 279.4156494
37: -223.7451019, 141.9507141, -223.7451019, 141.9507141, -365.6958008, 365.6958008
38: -190.3866577, 154.8811646, -190.3866577, 154.8811646, -345.2678223, 345.2678223
39: -226.4278870, 154.2048950, -226.4278870, 154.2048950, -380.6327820, 380.6327820
40: -192.2375641, 133.5492554, -192.2375641, 133.5492554, -325.7868042, 325.7868042
41: -144.3735046, 111.2714844, -144.3735046, 111.2714844, -255.6449738, 255.6449738
42: -111.1141891, 108.9842834, -111.1141891, 108.9842834, -220.0984344, 220.0984497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=698, inp2_unstable=698, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=874, inp2_unstable=874, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=38, inp2_unstable=38, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1159
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 627
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1239
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1175
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1205
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1149
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1241
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1240
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1139
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1171
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1223
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1245
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1133
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1238
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1150
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1225
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1235
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 659
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1140
type: RSZ, layer: 1, pos: 1233
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1116
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 612
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1165
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1181
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1234
type: RSZ, layer: 1, pos: 1155
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1224
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1166
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1222
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1115
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1257
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1256
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1273
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1138
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1109
type: RSZ, layer: 1, pos: 1169
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1237
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1217
type: RSZ, layer: 1, pos: 1218
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1108
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1154
type: RSZ, layer: 1, pos: 1100
type: RSZ, layer: 1, pos: 1182
type: RSZ, layer: 1, pos: 1107
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1080
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1219
type: RSZ, layer: 1, pos: 1162
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1272
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1170
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1079
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1123
type: RSZ, layer: 1, pos: 1153
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1137
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 1214
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 1251
type: RSZ, layer: 1, pos: 660
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 1187
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1124
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1236
type: RSZ, layer: 1, pos: 1213
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1226
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1267
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1183
type: RSZ, layer: 1, pos: 1211
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1053
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1099
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1221
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1084
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 263
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1204
type: RSZ, layer: 1, pos: 1202
type: RSZ, layer: 1, pos: 1064
type: RSZ, layer: 1, pos: 1201
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 891
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1178
type: RSZ, layer: 1, pos: 1252
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1250
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1093
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1110
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 325
type: RSZ, layer: 1, pos: 691
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 879
type: RSZ, layer: 1, pos: 1046
type: RSZ, layer: 1, pos: 261
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 295
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1258
type: RSZ, layer: 1, pos: 1203
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1083
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1249
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1029
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1254
type: RSZ, layer: 1, pos: 1073
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1244
type: RSZ, layer: 1, pos: 1268
type: RSZ, layer: 1, pos: 1266
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1278
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1030
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1081
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 676
type: RSZ, layer: 1, pos: 1265
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1220
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1092
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 1274
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 262
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1227
type: RSZ, layer: 1, pos: 1122
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 1045
type: RSZ, layer: 1, pos: 1094
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1065
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1279
type: RSZ, layer: 1, pos: 1074
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1089
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1076
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 305
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 309
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 275
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 324
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 260
type: RSZ, layer: 1, pos: 264
type: RSZ, layer: 1, pos: 1028
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 277
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1082
type: RSZ, layer: 1, pos: 1077
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 846
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1104
type: RSZ, layer: 1, pos: 307
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1097
type: RSZ, layer: 1, pos: 1091
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 323
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 294
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1105
type: RSZ, layer: 1, pos: 1062
type: RSZ, layer: 1, pos: 306
type: RSZ, layer: 1, pos: 1027
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1232
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1121
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1216
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1098
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 847
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 1090
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1228
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1088
type: RSZ, layer: 1, pos: 1042
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1106
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 259
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1269
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1185
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1120
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1095
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1072
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1200
type: RSZ, layer: 1, pos: 1136
type: RSZ, layer: 1, pos: 1041
type: RSZ, layer: 1, pos: 1186
type: RSZ, layer: 1, pos: 1056
type: RSZ, layer: 1, pos: 293
type: RSZ, layer: 1, pos: 336
type: RSZ, layer: 1, pos: 308
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 112
type: RSZ, layer: 1, pos: 1044
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 2
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 353
type: RSZ, layer: 1, pos: 1025
type: RSZ, layer: 1, pos: 1792
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 291
type: RSZ, layer: 1, pos: 1060
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1253
type: RSZ, layer: 1, pos: 321
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1196
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 276
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 1075
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 1026
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 1059
type: RSZ, layer: 1, pos: 675
type: RSZ, layer: 1, pos: 1061
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1152
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1168
type: RSZ, layer: 1, pos: 1066
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1058
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1040
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1197
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1057
type: RSZ, layer: 1, pos: 256
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1043
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 258
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 320
type: RSZ, layer: 1, pos: 292
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 352
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 337
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 322
type: RSZ, layer: 1, pos: 1248
type: RSZ, layer: 1, pos: 1212
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1024
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1078
type: RSZ, layer: 1, pos: 1264
type: RSZ, layer: 1, pos: 257
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1184

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1719

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -148.5965556, upper bound: 148.4407534
time: 1646.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -148.4807854, upper bound: 148.5570244
time: 775.82 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 2425.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2425.08
Output dim: 3, lower bound: -148.5570243, upper bound: 148.4807855
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2425.08
Output dim: 3, lower bound: -148.4407533, upper bound: 148.5965557
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 2425.08
Output dim: 3, lower bound: -148.5965556, upper bound: 148.4407534
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 2425.08
Output dim: 3, lower bound: -148.4807854, upper bound: 148.5570244
Binary search (step 1): status=Status.UNKNOWN, k_low=8, k_high=9, k_mid=8, eps_mid=0.0312500, abs_max=233.43865966796875
rel_dist={3: [-148.6090126930012, 148.60901277819403]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.02734375
execution time: 9507.49 seconds
