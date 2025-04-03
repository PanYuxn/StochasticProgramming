import gurobipy as gp
from gurobipy import GRB

def solve_deterministic_equivalent():
    """使用确定性等价方法求解三阶段随机线性规划问题"""
    
    # 初始化参数
    # 第一阶段参数
    c1 = 10  # 投资成本系数
    
    # 第二阶段参数 (2个情景)
    c2 = {1: 15, 2: 15}  # 情景1: 高增长, 情景2: 低增长
    b2 = {1: 120, 2: 100}
    prob_stage2 = {1: 0.5, 2: 0.5}
    
    # 第三阶段参数 (每个第二阶段情景下3个情景，共6个情景)
    c3 = {
        (1, 1): 5, (1, 2): 5, (1, 3): 5,  # 情景1下的三个子情景
        (2, 1): 5, (2, 2): 5, (2, 3): 5   # 情景2下的三个子情景
    }
    b3 = {
        (1, 1): 180, (1, 2): 150, (1, 3): 130,  # 高增长情景下的需求
        (2, 1): 150, (2, 2): 130, (2, 3): 110   # 低增长情景下的需求
    }
    prob_stage3 = {
        (1, 1): 0.3, (1, 2): 0.4, (1, 3): 0.3,
        (2, 1): 0.3, (2, 2): 0.4, (2, 3): 0.3
    }
    
    # 创建模型
    model = gp.Model("Deterministic_Equivalent")
    
    # 第一阶段决策变量
    x1 = model.addVar(lb=0.0, name="x1")
    
    # 第二阶段决策变量 (对每个情景)
    x2 = {}
    for s in [1, 2]:
        x2[s] = model.addVar(lb=0.0, name=f"x2_{s}")
    
    # 第三阶段决策变量 (对每个情景组合)
    x3 = {}
    for s in [1, 2]:
        for r in [1, 2, 3]:
            x3[(s, r)] = model.addVar(lb=0.0, name=f"x3_{s}_{r}")
    
    # 第二阶段约束 (对每个情景)
    for s in [1, 2]:
        model.addConstr(x2[s] + x1 == b2[s], name=f"balance_stage2_{s}")
    
    # 第三阶段约束 (对每个情景组合)
    for s in [1, 2]:
        for r in [1, 2, 3]:
            model.addConstr(x3[(s, r)] + x2[s] == b3[(s, r)], name=f"balance_stage3_{s}_{r}")
    
    # 目标函数 (预期总成本)
    objective = c1 * x1
    
    # 添加第二阶段和第三阶段的期望成本
    for s in [1, 2]:
        stage2_cost = c2[s] * x2[s]
        stage3_cost = 0
        for r in [1, 2, 3]:
            stage3_cost += prob_stage3[(s, r)] * c3[(s, r)] * x3[(s, r)]
        objective += prob_stage2[s] * (stage2_cost + stage3_cost)
    
    model.setObjective(objective, GRB.MINIMIZE)
    
    # 求解模型
    model.optimize()
    
    # 输出结果
    print("\n====== 确定性等价模型求解结果 ======")
    
    if model.status == GRB.OPTIMAL:
        print(f"最优目标值: {model.objVal:.2f}")
        print(f"第一阶段决策: x1 = {x1.x:.2f}")
        
        print("\n第二阶段决策:")
        for s in [1, 2]:
            print(f"  情景 {s} (概率 = {prob_stage2[s]}): x2 = {x2[s].x:.2f}")
        
        print("\n第三阶段决策:")
        for s in [1, 2]:
            for r in [1, 2, 3]:
                print(f"  情景 ({s},{r}) (概率 = {prob_stage3[(s,r)]}): x3 = {x3[(s,r)].x:.2f}")
        
        # 验证结果
        print("\n验证目标函数:")
        first_stage_cost = c1 * x1.x
        print(f"  第一阶段成本: {first_stage_cost:.2f}")
        
        expected_later_stages_cost = 0
        for s in [1, 2]:
            stage2_cost = c2[s] * x2[s].x
            stage3_cost = 0
            for r in [1, 2, 3]:
                stage3_cost += prob_stage3[(s, r)] * c3[(s, r)] * x3[(s, r)].x
            scenario_cost = stage2_cost + stage3_cost
            expected_later_stages_cost += prob_stage2[s] * scenario_cost
            print(f"  情景 {s} 第二阶段成本: {stage2_cost:.2f}, 期望第三阶段成本: {stage3_cost:.2f}")
        
        print(f"  期望后续阶段成本: {expected_later_stages_cost:.2f}")
        print(f"  总期望成本: {first_stage_cost + expected_later_stages_cost:.2f}")
    else:
        print("未找到最优解")
    
    return model, x1, x2, x3

# 运行求解器
if __name__ == "__main__":
    model, x1, x2, x3 = solve_deterministic_equivalent()