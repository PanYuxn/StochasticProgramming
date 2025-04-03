import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import matplotlib.pyplot as plt

class AirConditionerProductionProblem:
    """
    空调生产规划问题的输入类
    """
    
    def __init__(self):
        # 生产参数
        self.regular_production_capacity = 200  # 每月正常产能
        self.regular_production_cost = 100      # 每单位正常生产成本
        self.overtime_production_cost = 300     # 每单位加班生产成本
        self.storage_cost = 50                  # 每单位每月存储成本
        
        # 需求场景
        # 第一阶段：确定性需求
        self.first_stage_demand = 100
        
        # 第二阶段：两种可能的需求情况（低需求和高需求）
        self.second_stage_demands = [100, 300]
        self.second_stage_probabilities = [0.5, 0.5]
        
        # 第三阶段：两种可能的需求情况（低需求和高需求）
        self.third_stage_demands = [100, 300]
        self.third_stage_probabilities = [0.5, 0.5]
        
        # 构建场景树
        self.scenario_tree = self._build_scenario_tree()
    
    def _build_scenario_tree(self):
        """构建场景树"""
        scenario_tree = {
            1: {  # 第一阶段
                0: {
                    'probability': 1.0,
                    'children': [0, 1],
                    'data': {'demand': self.first_stage_demand}
                }
            },
            2: {  # 第二阶段
                0: {
                    'probability': self.second_stage_probabilities[0],
                    'parent': 0,
                    'children': [0, 1],
                    'data': {'demand': self.second_stage_demands[0]}
                },
                1: {
                    'probability': self.second_stage_probabilities[1],
                    'parent': 0,
                    'children': [2, 3],
                    'data': {'demand': self.second_stage_demands[1]}
                }
            },
            3: {  # 第三阶段
                0: {
                    'probability': self.second_stage_probabilities[0] * self.third_stage_probabilities[0],
                    'parent': 0,
                    'data': {'demand': self.third_stage_demands[0]}
                },
                1: {
                    'probability': self.second_stage_probabilities[0] * self.third_stage_probabilities[1],
                    'parent': 0,
                    'data': {'demand': self.third_stage_demands[1]}
                },
                2: {
                    'probability': self.second_stage_probabilities[1] * self.third_stage_probabilities[0],
                    'parent': 1,
                    'data': {'demand': self.third_stage_demands[0]}
                },
                3: {
                    'probability': self.second_stage_probabilities[1] * self.third_stage_probabilities[1],
                    'parent': 1,
                    'data': {'demand': self.third_stage_demands[1]}
                }
            }
        }
        
        return scenario_tree
    
    def build_model(self, stage, scenario_data):
        """
        为特定阶段和场景构建优化模型
        
        参数:
        -----
        stage : int
            阶段（1, 2, 或 3）
        scenario_data : dict
            场景数据，包含需求信息
            
        返回:
        -----
        model : gurobipy.Model
            阶段模型
        """
        model = gp.Model(f"Stage_{stage}")
        
        # 获取当前阶段的需求
        demand = scenario_data['data']['demand']
        
        # 决策变量
        x = model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{stage}")  # 正常生产
        w = model.addVar(vtype=GRB.CONTINUOUS, name=f"w_{stage}")  # 加班生产
        
        # 库存变量
        if stage == 1:
            # 第一阶段，没有初始库存
            y_in = 0
        else:
            # 链接前一阶段的库存
            y_in = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_in_{stage}")
        
        y_out = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_out_{stage}")
        
        # 约束条件
        # 1. 正常生产能力限制
        model.addConstr(x <= self.regular_production_capacity, name=f"capacity_{stage}")
        
        # 2. 库存平衡约束
        model.addConstr(y_in + x + w - y_out == demand, name=f"balance_{stage}")
        
        # 3. 非负约束
        model.addConstr(x >= 0, name=f"non_neg_x_{stage}")
        model.addConstr(w >= 0, name=f"non_neg_w_{stage}")
        model.addConstr(y_out >= 0, name=f"non_neg_y_out_{stage}")
        
        # 目标函数
        obj = self.regular_production_cost * x + self.overtime_production_cost * w
        
        # 如果不是最后一个阶段，添加库存成本和指向下一阶段的状态变量
        if stage < 3:
            obj += self.storage_cost * y_out
            
        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        
        return model


class NestedLShapedMethod:
    """
    用于多阶段随机规划问题的Nested L-shaped方法实现
    """
    
    def __init__(self, num_stages, scenario_tree, model_builder_func, max_iterations=100, tolerance=1e-6):
        """
        初始化Nested L-shaped方法
        
        参数:
        -----
        num_stages : int
            随机规划问题的阶段数
        scenario_tree : dict
            表示场景树结构的字典
        model_builder_func : function
            构建阶段模型的函数
        max_iterations : int
            最大迭代次数
        tolerance : float
            收敛容差
        """
        self.num_stages = num_stages
        self.scenario_tree = scenario_tree
        self.model_builder_func = model_builder_func
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # 存储阶段模型、切割平面和解
        self.stage_models = {}
        self.optimality_cuts = {stage: {} for stage in range(1, num_stages)}
        self.feasibility_cuts = {stage: {} for stage in range(1, num_stages)}
        self.stage_solutions = {}
        
        # 上下界
        self.UB = float('inf')
        self.LB = float('-inf')
        
        # 收敛历史
        self.UB_history = []
        self.LB_history = []
        
    def build_stage_models(self):
        """构建所有阶段的所有场景模型"""
        for stage in range(1, self.num_stages + 1):
            self.stage_models[stage] = {}
            for node in self.scenario_tree[stage]:
                scenario_data = self.scenario_tree[stage][node]
                self.stage_models[stage][node] = self.model_builder_func(stage, scenario_data)
                
                # 对于阶段 > 1，添加future cost变量
                if stage < self.num_stages:
                    model = self.stage_models[stage][node]
                    model._costToGo = model.addVar(lb=-GRB.INFINITY, name="cost_to_go")
                    model.setObjective(model.getObjective() + model._costToGo)
                    model.update()
    
    def forward_pass(self):
        """
        Nested L-shaped方法的前向传递
        
        返回:
        -----
        dict: 所有阶段和场景的解
        """
        solutions = {}
        
        # 解决第一阶段
        first_stage_model = self.stage_models[1][0]  # 假设第一阶段的节点为0
        first_stage_model.optimize()
        
        if first_stage_model.status != GRB.OPTIMAL:
            raise ValueError("第一阶段问题无可行解或无界")
        
        solutions[1] = {0: self._extract_solution(first_stage_model)}
        
        # 后续阶段的前向传递
        for stage in range(2, self.num_stages + 1):
            solutions[stage] = {}
            
            for node in self.scenario_tree[stage]:
                parent_node = self.scenario_tree[stage][node]['parent']
                parent_solution = solutions[stage-1][parent_node]
                
                # 获取当前节点的模型
                model = self.stage_models[stage][node]
                
                # 根据父节点的解固定状态变量
                self._fix_state_variables(model, parent_solution)
                
                # 求解模型
                model.optimize()
                
                if model.status != GRB.OPTIMAL:
                    # 处理不可行情况
                    print(f"警告: 阶段 {stage}, 节点 {node} 不可行.")
                    # 稍后生成可行性切割
                    solutions[stage][node] = None
                else:
                    solutions[stage][node] = self._extract_solution(model)
        
        self.stage_solutions = solutions
        return solutions
    
    def backward_pass(self):
        """
        Nested L-shaped方法的后向传递，用于生成切割平面
        
        返回:
        -----
        bool: 如果达到收敛则返回True，否则返回False
        """
        # 从倒数第二阶段开始
        for stage in range(self.num_stages - 1, 0, -1):
            for node in self.scenario_tree[stage]:
                # 获取当前节点的模型
                model = self.stage_models[stage][node]
                
                # 获取子节点
                children = self.scenario_tree[stage][node].get('children', [])
                if not children:
                    continue
                
                # 计算期望回报函数值
                expected_recourse = 0
                for child in children:
                    probability = self.scenario_tree[stage+1][child]['probability']
                    child_solution = self.stage_solutions[stage+1][child]
                    
                    if child_solution is None:
                        # 生成可行性切割
                        self._add_feasibility_cut(stage, node, child)
                    else:
                        # 计算此子节点的future cost
                        obj_value = self.stage_models[stage+1][child].objVal
                        expected_recourse += probability * obj_value
                        
                        # 如果需要，生成最优性切割
                        self._add_optimality_cut(stage, node, child, child_solution)
                
                # 更新父模型中的future cost变量系数
                if stage > 1:
                    parent_node = self.scenario_tree[stage][node]['parent']
                    parent_model = self.stage_models[stage-1][parent_node]
                    # 更新父模型的future cost变量
                    parent_model._costToGo.obj = expected_recourse
                
                # 更新上下界
                if stage == 1:
                    obj_value = model.objVal
                    self.UB = min(self.UB, obj_value)
                    self.LB = max(self.LB, obj_value - model._costToGo.X)
                    
                    self.UB_history.append(self.UB)
                    self.LB_history.append(self.LB)
        
        # 检查收敛
        gap = abs(self.UB - self.LB) / (1e-10 + abs(self.UB))
        return gap <= self.tolerance
    
    def _extract_solution(self, model):
        """从模型中提取解值"""
        solution = {}
        for var in model.getVars():
            if not var.name.startswith('cost_to_go'):
                solution[var.name] = var.X
        return solution
    
    def _fix_state_variables(self, model, parent_solution):
        """根据父节点的解固定状态变量"""
        for var_name, value in parent_solution.items():
            # 仅固定状态变量（链接各阶段的变量）
            if var_name.startswith('y_out_'):
                # 从父阶段的输出库存获取当前阶段的输入库存
                stage = int(var_name.split('_')[2])
                current_stage = stage + 1
                var = model.getVarByName(f"y_in_{current_stage}")
                if var is not None:
                    var.lb = value
                    var.ub = value
        model.update()
    
    def _add_optimality_cut(self, stage, node, child_node, child_solution):
        """向阶段模型添加最优性切割"""
        model = self.stage_models[stage][node]
        child_model = self.stage_models[stage+1][child_node]
        
        # 获取链接约束的对偶值
        # 这里简化处理，实际问题中需要根据具体问题结构谨慎地制定切割
        
        # 创建最优性切割
        cut_expr = child_model.objVal
        
        # 添加切割
        if cut_expr != 0:
            cut_name = f"opt_cut_s{stage}_n{node}_c{child_node}_{len(self.optimality_cuts[stage])}"
            model.addConstr(model._costToGo >= cut_expr, name=cut_name)
            self.optimality_cuts[stage][cut_name] = cut_expr
            model.update()
    
    def _add_feasibility_cut(self, stage, node, child_node):
        """向阶段模型添加可行性切割"""
        # 这是可行性切割生成的占位符
        # 实际中，需要解决一个不可行性问题，并基于对偶解生成可行性切割
        print(f"为阶段 {stage}, 节点 {node}, 子节点 {child_node} 添加可行性切割")
        
        # 一种简化的方法是添加约束，防止导致不可行的相同第一阶段解
        # 这是问题相关的，需要谨慎实现
        
        cut_name = f"feas_cut_s{stage}_n{node}_c{child_node}_{len(self.feasibility_cuts[stage])}"
        self.feasibility_cuts[stage][cut_name] = None  # 存储切割
    
    def solve(self):
        """
        使用Nested L-shaped方法求解多阶段随机规划问题
        
        返回:
        -----
        dict: 最优解
        float: 最优目标值
        """
        # 构建所有阶段模型
        print("构建阶段模型...")
        self.build_stage_models()
        
        print("开始Nested L-shaped方法...")
        start_time = time.time()
        
        for iter_num in range(self.max_iterations):
            print(f"\n迭代 {iter_num+1}")
            
            # 前向传递
            print("  前向传递...")
            self.forward_pass()
            
            # 后向传递
            print("  后向传递...")
            converged = self.backward_pass()
            
            print(f"  UB: {self.UB:.6f}, LB: {self.LB:.6f}, Gap: {abs(self.UB - self.LB) / (1e-10 + abs(self.UB)):.6f}")
            
            if converged:
                print(f"\n在 {iter_num+1} 次迭代后收敛!")
                break
        
        else:
            print(f"\n达到最大迭代次数 ({self.max_iterations}).")
        
        elapsed_time = time.time() - start_time
        print(f"总求解时间: {elapsed_time:.2f} 秒")
        
        # 绘制收敛图
        self._plot_convergence()
        
        return self.stage_solutions, self.UB
    
    def _plot_convergence(self):
        """绘制上下界的收敛情况"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.UB_history) + 1), self.UB_history, 'r-', label='上界')
        plt.plot(range(1, len(self.LB_history) + 1), self.LB_history, 'b-', label='下界')
        plt.xlabel('迭代次数')
        plt.ylabel('边界值')
        plt.title('Nested L-shaped方法的收敛情况')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('nested_lshaped_convergence.png')
        plt.close()


# 主函数：解决空调生产规划问题
def solve_ac_production_planning():
    """
    解决空调生产规划问题
    """
    # 创建问题实例
    problem = AirConditionerProductionProblem()
    
    # 创建并求解问题
    nested_lshaped = NestedLShapedMethod(
        num_stages=3,
        scenario_tree=problem.scenario_tree,
        model_builder_func=problem.build_model,
        max_iterations=20,
        tolerance=1e-4
    )
    
    optimal_solutions, optimal_value = nested_lshaped.solve()
    
    # 打印结果
    print("\n最优解:")
    for stage in range(1, 4):
        print(f"\n第 {stage} 阶段:")
        for node in optimal_solutions[stage]:
            print(f"  场景 {node}:")
            for var_name, value in optimal_solutions[stage][node].items():
                print(f"    {var_name}: {value:.4f}")
    
    print(f"\n最优目标值 (期望总成本): {optimal_value:.4f}")
    
    # 打印第一阶段决策建议
    first_stage_sol = optimal_solutions[1][0]
    print("\n第一阶段决策建议:")
    print(f"  正常生产量: {first_stage_sol.get('x_1', 0):.0f} 台")
    print(f"  加班生产量: {first_stage_sol.get('w_1', 0):.0f} 台")
    print(f"  库存量: {first_stage_sol.get('y_out_1', 0):.0f} 台")


if __name__ == "__main__":
    solve_ac_production_planning()