\ Model Second_Stage_Problem_Scenario_1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  15 x2_1 + theta3_1
Subject To
 Balance_1: x2_1 = -0.5
 initial_cut: 5 x2_1 + theta3_1 >= 765
Bounds
End
