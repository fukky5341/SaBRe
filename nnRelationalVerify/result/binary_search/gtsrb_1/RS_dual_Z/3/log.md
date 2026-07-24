## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 109.86210131
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490)
1: (-84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591)
2: (-73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443)
3: (-82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425)
4: (-85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136)
5: (-84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891)
6: (-104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552)
7: (-102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182)
8: (-98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266)
9: (-82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299)
10: (-117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769)
11: (-122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353)
12: (-113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700)
13: (-115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660)
14: (-172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130)
15: (-97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083)
16: (-131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306)
17: (-178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383)
18: (-114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708)
19: (-88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800)
20: (-78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332)
21: (-108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175)
22: (-112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821)
23: (-89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540)
24: (-107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015)
25: (-91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116)
26: (-127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797)
27: (-111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618)
28: (-86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463)
29: (-120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066)
30: (-107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661)
31: (-113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204)
32: (-101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892)
33: (-142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755)
34: (-121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955)
35: (-118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390)
36: (-111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718)
37: (-162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274)
38: (-145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403)
39: (-166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481)
40: (-141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322)
41: (-103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708)
42: (-79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822)

## BASE Result
execution time: IAR + LP analysis = 2.82 + 141.21 = 144.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -117.8238352, upper bound: 117.8238352


# Binary Search by BASE starts (time budget: 17855.98 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=170.32391357421875
rel_dist={4: [-113.45889199656516, 113.45889199845624]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=170.32391357421875
rel_dist={4: [-109.87288027651347, 109.87288028306807]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=170.32391357421875
rel_dist={4: [-106.78651155613684, 106.78651156283166]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=170.32391357421875
rel_dist={4: [-108.4037440570051, 108.40374405376852]}

## Binary Search Result
Binary search time: 2813.03 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 15042.94 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.3936823, upper bound: 114.4118220
time: 747.02 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4118220, upper bound: 114.3936823
time: 147.92 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 895.09 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 895.09
Output dim: 4, lower bound: -114.3936823, upper bound: 114.4118220
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 895.09
Output dim: 4, lower bound: -114.4118220, upper bound: 114.3936823

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.3571387, upper bound: 114.4064129
time: 122.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.3881215, upper bound: 114.3754901
time: 181.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.3754901, upper bound: 114.3881215
time: 294.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.4064129, upper bound: 114.3571387
time: 176.16 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 472.84 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 472.84
Output dim: 4, lower bound: -114.3571387, upper bound: 114.4064129
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 472.84
Output dim: 4, lower bound: -114.3881215, upper bound: 114.3754901
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 472.84
Output dim: 4, lower bound: -114.3754901, upper bound: 114.3881215
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 472.84
Output dim: 4, lower bound: -114.4064129, upper bound: 114.3571387

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.3126134, upper bound: 114.3922419
time: 1767.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.3410638, upper bound: 114.3530280
time: 120.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.3438646, upper bound: 114.3610522
time: 127.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -114.3720594, upper bound: 114.3219727
time: 222.16 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 351.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 351.81
Output dim: 4, lower bound: -114.3126134, upper bound: 114.3922419
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 351.81
Output dim: 4, lower bound: -114.3410638, upper bound: 114.3530280
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 351.81
Output dim: 4, lower bound: -114.3438646, upper bound: 114.3610522
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 351.81
Output dim: 4, lower bound: -114.3720594, upper bound: 114.3219727
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 351.81
Output dim: 4, lower bound: -114.3754901, upper bound: 114.3881215
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 351.81
Output dim: 4, lower bound: -114.4064129, upper bound: 114.3571387
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=170.32391357421875
rel_dist={4: [-114.42300729727798, 114.42300729596855]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1704933, upper bound: 111.1868268
time: 146.04 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1868268, upper bound: 111.1704933
time: 174.63 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 320.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 320.83
Output dim: 4, lower bound: -111.1704933, upper bound: 111.1868268
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 320.83
Output dim: 4, lower bound: -111.1868268, upper bound: 111.1704933

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1421731, upper bound: 111.1842587
time: 140.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1680774, upper bound: 111.1587816
time: 171.98 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1587816, upper bound: 111.1680774
time: 126.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1842587, upper bound: 111.1421731
time: 172.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 301.29 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 301.29
Output dim: 4, lower bound: -111.1421731, upper bound: 111.1842587
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 301.29
Output dim: 4, lower bound: -111.1680774, upper bound: 111.1587816
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 301.29
Output dim: 4, lower bound: -111.1587816, upper bound: 111.1680774
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 301.29
Output dim: 4, lower bound: -111.1842587, upper bound: 111.1421731

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1141962, upper bound: 111.1752537
time: 156.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1320508, upper bound: 111.1499648
time: 159.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1401695, upper bound: 111.1497517
time: 177.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1579253, upper bound: 111.1243101
time: 1894.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1243101, upper bound: 111.1579253
time: 3406.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -111.1497517, upper bound: 111.1401695
time: 147.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3556.34 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3556.34
Output dim: 4, lower bound: -111.1141962, upper bound: 111.1752537
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3556.34
Output dim: 4, lower bound: -111.1320508, upper bound: 111.1499648
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3556.34
Output dim: 4, lower bound: -111.1401695, upper bound: 111.1497517
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3556.34
Output dim: 4, lower bound: -111.1579253, upper bound: 111.1243101
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3556.34
Output dim: 4, lower bound: -111.1243101, upper bound: 111.1579253
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3556.34
Output dim: 4, lower bound: -111.1497517, upper bound: 111.1401695
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3556.34
Output dim: 4, lower bound: -111.1842587, upper bound: 111.1421731
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=170.32391357421875
rel_dist={4: [-111.19460966977735, 111.19460966338143]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -109.8532787, upper bound: 109.8666963
time: 204.10 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -109.8666963, upper bound: 109.8532787
time: 125.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 329.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 329.77
Output dim: 4, lower bound: -109.8532787, upper bound: 109.8666963
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 329.77
Output dim: 4, lower bound: -109.8666963, upper bound: 109.8532787

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -109.8256476, upper bound: 109.8622038
time: 155.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -109.8488586, upper bound: 109.8395729
time: 127.95 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1657

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -109.8395729, upper bound: 109.8488586
time: 243.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -109.8622038, upper bound: 109.8256476
time: 169.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 415.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 415.57
Output dim: 4, lower bound: -109.8256476, upper bound: 109.8622038
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 415.57
Output dim: 4, lower bound: -109.8488586, upper bound: 109.8395729
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 415.57
Output dim: 4, lower bound: -109.8395729, upper bound: 109.8488586
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 415.57
Output dim: 4, lower bound: -109.8622038, upper bound: 109.8256476

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -109.8034606, upper bound: 109.8551014
time: 114.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -109.8173923, upper bound: 109.8354061
time: 125.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -138.9447327, 94.5717163, -138.9447327, 94.5717163, -233.5164490, 233.5164490
1: -84.2806854, 82.2811661, -84.2806854, 82.2811661, -166.5618591, 166.5618591
2: -73.7048492, 72.8923950, -73.7048492, 72.8923950, -146.5972443, 146.5972443
3: -82.0299301, 88.7317123, -82.0299301, 88.7317123, -170.7616425, 170.7616425
4: -85.0996704, 85.2242432, -85.0996704, 85.2242432, -170.3239136, 170.3239136
5: -84.9262543, 90.0769348, -84.9262543, 90.0769348, -175.0031891, 175.0031891
6: -104.5175018, 79.1390533, -104.5175018, 79.1390533, -183.6565552, 183.6565552
7: -102.0870590, 92.8603592, -102.0870590, 92.8603592, -194.9474182, 194.9474182
8: -98.1902008, 102.6377258, -98.1902008, 102.6377258, -200.8278961, 200.8279266
9: -82.6638184, 85.2262115, -82.6638184, 85.2262115, -167.8900299, 167.8900299
10: -117.0789948, 111.7922897, -117.0789948, 111.7922897, -228.8712769, 228.8712769
11: -122.8685532, 93.4825897, -122.8685532, 93.4825897, -216.3511353, 216.3511353
12: -113.0648117, 103.8744659, -113.0648117, 103.8744659, -216.9392700, 216.9392700
13: -115.3176193, 126.3525543, -115.3176193, 126.3525543, -241.6701660, 241.6701660
14: -172.2850952, 94.2245178, -172.2850952, 94.2245178, -266.5096130, 266.5096130
15: -97.1777878, 83.8445358, -97.1777878, 83.8445358, -181.0223083, 181.0223083
16: -131.8961792, 99.9564438, -131.8961792, 99.9564438, -231.8526306, 231.8526306
17: -178.4643250, 130.2437134, -178.4643250, 130.2437134, -308.7080078, 308.7080383
18: -114.7727280, 96.7998428, -114.7727280, 96.7998428, -211.5725708, 211.5725708
19: -88.5097122, 56.3835678, -88.5097122, 56.3835678, -144.8932800, 144.8932800
20: -78.5058136, 66.1282196, -78.5058136, 66.1282196, -144.6340332, 144.6340332
21: -108.7656555, 69.9066772, -108.7656555, 69.9066772, -178.6723175, 178.6723175
22: -112.6518707, 73.7141190, -112.6518707, 73.7141190, -186.3659821, 186.3659821
23: -89.6281815, 67.4846802, -89.6281815, 67.4846802, -157.1128540, 157.1128540
24: -107.8633347, 62.8519592, -107.8633347, 62.8519592, -170.7153015, 170.7153015
25: -91.0894318, 74.6255798, -91.0894318, 74.6255798, -165.7149963, 165.7150116
26: -127.6246490, 109.9039383, -127.6246490, 109.9039383, -237.5285950, 237.5285797
27: -111.3309402, 74.0468369, -111.3309402, 74.0468369, -185.3777466, 185.3777618
28: -86.4348373, 76.2489014, -86.4348373, 76.2489014, -162.6837463, 162.6837463
29: -120.1831512, 73.7367554, -120.1831512, 73.7367554, -193.9199066, 193.9199066
30: -107.9367218, 83.4610443, -107.9367218, 83.4610443, -191.3977661, 191.3977661
31: -113.8610535, 69.1560669, -113.8610535, 69.1560669, -183.0171204, 183.0171204
32: -101.2306137, 78.8014755, -101.2306137, 78.8014755, -180.0320892, 180.0320892
33: -142.0742035, 112.5369644, -142.0742035, 112.5369644, -254.6111755, 254.6111755
34: -121.6920776, 89.1620331, -121.6920776, 89.1620331, -210.8540802, 210.8540955
35: -118.6921310, 88.5410309, -118.6921310, 88.5410309, -207.2331543, 207.2331390
36: -111.4387131, 89.7848740, -111.4387131, 89.7848740, -201.2235870, 201.2235718
37: -162.9339142, 98.5217285, -162.9339142, 98.5217285, -261.4556274, 261.4556274
38: -145.7049561, 114.4813995, -145.7049561, 114.4813995, -260.1863403, 260.1863403
39: -166.0494995, 110.9698639, -166.0494995, 110.9698639, -277.0193481, 277.0193481
40: -141.4682465, 92.5259857, -141.4682465, 92.5259857, -233.9942322, 233.9942322
41: -103.5714188, 76.5524673, -103.5714188, 76.5524673, -180.1238861, 180.1238708
42: -79.8955536, 74.0618286, -79.8955536, 74.0618286, -153.9573822, 153.9573822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=575, inp2_unstable=575, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=737, inp2_unstable=737, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=33, inp2_unstable=33, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1753
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1673
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1688
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 637
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1719
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 1677
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 666
type: RSZ, layer: 1, pos: 1621
type: RSZ, layer: 1, pos: 669
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 1693
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1651
type: RSZ, layer: 1, pos: 729
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1415
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1690
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 636
type: RSZ, layer: 1, pos: 1720
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 661
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1737
type: RSZ, layer: 1, pos: 1691
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 628
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 728
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1271
type: RSZ, layer: 1, pos: 1255
type: RSZ, layer: 1, pos: 1627
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1717
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 651
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1692
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 694
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1246
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1624
type: RSZ, layer: 1, pos: 1748
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1049
type: RSZ, layer: 1, pos: 1047
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 678
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1230
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1229
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1609
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 619
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 713
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 644
type: RSZ, layer: 1, pos: 1151
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1644
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1607
type: RSZ, layer: 1, pos: 1731
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 711
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1048
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1699
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1736
type: RSZ, layer: 1, pos: 1247
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 712
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1242
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1167
type: RSZ, layer: 1, pos: 1277
type: RSZ, layer: 1, pos: 645
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1489
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1708
type: RSZ, layer: 1, pos: 1231
type: RSZ, layer: 1, pos: 1031
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 701
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1134
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1365
type: RSZ, layer: 1, pos: 1086
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 1261
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 668
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 652
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 934
type: RSZ, layer: 1, pos: 677
type: RSZ, layer: 1, pos: 698
type: RSZ, layer: 1, pos: 1457
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1709
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1426
type: RSZ, layer: 1, pos: 1703
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1215
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1427
type: RSZ, layer: 1, pos: 682
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1033
type: RSZ, layer: 1, pos: 699
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1087
type: RSZ, layer: 1, pos: 1270
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1262
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1032
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1070
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 635
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1410
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1333
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1276
type: RSZ, layer: 1, pos: 1330
type: RSZ, layer: 1, pos: 716
type: RSZ, layer: 1, pos: 1394
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1626
type: RSZ, layer: 1, pos: 1085
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1135
type: RSZ, layer: 1, pos: 1050
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 697
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 1349
type: RSZ, layer: 1, pos: 1522
type: RSZ, layer: 1, pos: 1260
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1054
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1263
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1425
type: RSZ, layer: 1, pos: 1071
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 1103
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1450
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1051
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1393
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1700
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 1684
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1067
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1069
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 714
type: RSZ, layer: 1, pos: 683
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1068
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1314
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1346
type: RSZ, layer: 1, pos: 1055
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 684
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1315
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1362
type: RSZ, layer: 1, pos: 1521
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1063
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1035
type: RSZ, layer: 1, pos: 1034
type: RSZ, layer: 1, pos: 1243
type: RSZ, layer: 1, pos: 1706
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 786
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 817
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 710
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1038
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 1118
type: RSZ, layer: 1, pos: 1037
type: RSZ, layer: 1, pos: 1715
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 1275
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1039
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1036
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1259
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1198
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1716
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 700
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 771
type: RSZ, layer: 1, pos: 1119
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1704
type: RSZ, layer: 1, pos: 1199
type: RSZ, layer: 1, pos: 715
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 802
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1301
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 1347
type: RSZ, layer: 1, pos: 1052
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -109.8354061, upper bound: 109.8173923
time: 283.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -109.8551014, upper bound: 109.8034606
time: 276.73 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 563.13 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 563.13
Output dim: 4, lower bound: -109.8034606, upper bound: 109.8551014
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 563.13
Output dim: 4, lower bound: -109.8173923, upper bound: 109.8354061
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 563.13
Output dim: 4, lower bound: -109.8354061, upper bound: 109.8173923
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 563.13
Output dim: 4, lower bound: -109.8551014, upper bound: 109.8034606
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=170.32391357421875
rel_dist={4: [-109.87288027651347, 109.87288028306807]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 13455.27 seconds
