import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from typing import Tuple, List, Dict, Any, Optional
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体，用于图表显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

class VariableFactorProblem:
    """

    这个类实现了基于Generalized Benders Decomposition (GBD)算法求解
    变量因子规划问题的全过程。GBD是一种分解方法，将原问题分解为主问题和子问题，
    通过迭代求解逐步收敛到全局最优解。
    
    问题的形式为：
        min  x₁(-y₁-1) + x₂(-0.8y₂-1.2)
        s.t. x₁y₁ + x₂y₂ ≤ 0.75
             x₁ + x₂ = 1
             x₁, x₂ ≥ 0
             0 ≤ y₁, y₂ ≤ 10
    """
    def __init__(self, max_iterations: int = 10, tolerance: float = 1e-6, verbose: bool = True):
        """
        初始化
        
        参数:
            max_iterations: 最大迭代次数，控制算法的最大运行次数
            tolerance: 收敛容差，当上下界相对间隙小于此值时停止迭代
            verbose: 是否输出详细信息，用于调试和教学展示
        """
        # 问题数据参数
        self.c1 = np.array([-1.0, -1.2])      # x部分常数系数，对应于目标函数中的常数项
        self.c2 = np.array([-1.0, -0.8])      # y系数，对应于目标函数中y的系数
        self.A1 = np.array([[1.0, 1.0]])      # 第一阶段约束矩阵，表示x₁ + x₂ = 1的约束
        self.b1 = np.array([1.0])             # 第一阶段约束右侧值
        self.resource_limit = 0.75            # 资源限制，表示x₁y₁ + x₂y₂ ≤ 0.75中的右侧值
        self.y_upper = 10.0                   # y变量上界
        self.y_lower = 0.0                    # y变量下界
        
        # 算法控制参数
        self.max_iterations = max_iterations  # 最大迭代次数
        self.tolerance = tolerance            # 收敛容差
        self.verbose = verbose                # 是否输出详细信息
        
        # 初始解设置
        self.initial_x = np.array([1.0, 0.0])  # 初始x解，从笔记中的例子取值
        
        # 结果存储变量
        self.best_x = None                     # 存储最优的x解
        self.best_y = None                     # 存储最优的y解
        self.best_obj = float('inf')           # 存储最优目标函数值，初始设为无穷大
        self.lower_bound = float('-inf')       # 问题下界，初始设为负无穷
        self.upper_bound = float('inf')        # 问题上界，初始设为正无穷
        self.history = []                      # 存储迭代历史
        self.cuts = []                         # 存储生成的割平面集合
    
    def solve_subproblem(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        针对给定的x值，求解子问题以获取最优y和对偶变量π
        
        在GBD算法中，子问题是针对固定的x值求解原问题，获得对应的最优y值和对偶变量π。
        这些值将用于构建主问题中的割平面。
        
        子问题的形式为：
            min  x₁(-y₁-1) + x₂(-0.8y₂-1.2)
            s.t. x₁y₁ + x₂y₂ ≤ 0.75
                 0 ≤ y₁, y₂ ≤ 10
        
        参数:
            x: 当前的x值，一个二维数组[x₁, x₂]
            
        返回:
            y_sol: 最优的y解
            pi_sol: 对偶变量π的值
            obj_value: 子问题的目标函数值
        """
        try:
            # 创建Gurobi优化模型
            model = gp.Model("VFPSubproblem")
            model.setParam('OutputFlag', 0)  # 关闭Gurobi的输出信息
            
            # 添加决策变量y₁和y₂，设置其上下界
            y1 = model.addVar(lb=self.y_lower, ub=self.y_upper, name="y1")
            y2 = model.addVar(lb=self.y_lower, ub=self.y_upper, name="y2")
            
            # 设置目标函数：x₁(-y₁-1) + x₂(-0.8y₂-1.2)
            # 注意这里x是固定的参数，而y是决策变量
            obj = x[0] * (self.c2[0] * y1 + self.c1[0]) + x[1] * (self.c2[1] * y2 + self.c1[1])
            model.setObjective(obj, GRB.MINIMIZE)
            
            # 添加资源约束：x₁y₁ + x₂y₂ ≤ 0.75
            # 这个约束对应的对偶变量将用于生成割平面
            resource_constr = model.addConstr(x[0] * y1 + x[1] * y2 <= self.resource_limit, name="resource")
            
            # 更新模型并求解
            model.update()
            model.optimize()
            
            # 检查求解状态并提取结果
            if model.Status == GRB.OPTIMAL:
                # 获取最优y值
                y_sol = np.array([y1.X, y2.X])
                
                # 获取资源约束对应的对偶变量π
                # 对偶变量表示资源的边际价值，是构建割平面的关键
                pi_sol = np.array([resource_constr.Pi])
                
                # 计算目标函数值
                obj_value = model.ObjVal
                
                return y_sol, pi_sol, obj_value
            else:
                # 如果无最优解，抛出异常
                raise ValueError(f"子问题无最优解，状态: {model.Status}")
                
        except gp.GurobiError as e:
            print(f"子问题Gurobi错误: {str(e)}")
            raise
    
    def create_cut(self, x: np.ndarray, pi: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        创建Benders割平面
        
        在GBD算法中，割平面是通过拉格朗日函数L(x,π)创建的，用于在主问题中逐步
        逼近真实的最优解。每次迭代生成一个割平面，形式为 θ ≥ gradient·x + rhs。
        
        拉格朗日函数L的形式为：
            L(x,π) = min_y [x₁(-y₁-1) + x₂(-0.8y₂-1.2) + π(0.75 - x₁y₁ - x₂y₂)]
        
        步骤：
        1. 首先整理L(x,π)为关于x的表达式
           L(x,π) = x₁(-y₁-1-π·y₁) + x₂(-0.8y₂-1.2-π·y₂) + π·0.75
        
        2. 对于每个y分量，根据其系数决定取上界还是下界：
           - 如果某个y的系数为负，则取y的上界以最小化L
           - 如果某个y的系数为正，则取y的下界以最小化L
           - 如果某个y的系数为0，则y可取任意值
           
        3. 计算L函数关于x的梯度，作为割平面的系数
        
        4. 计算切割平面的右侧常数项
        
        参数:
            x: 当前的x值
            pi: 子问题的对偶变量
            y: 子问题的最优解
            
        返回:
            cut_coeffs: 割平面系数，即梯度
            cut_rhs: 割平面右侧常数项
        """
        # 提取对偶值
        pi_val = pi[0]
        
        # 计算当前点L(x,π)的函数值
        # L(x,π) = x₁(-y₁-1) + x₂(-0.8y₂-1.2) + π(0.75 - x₁y₁ - x₂y₂)
        current_value = (x[0] * (self.c2[0] * y[0] + self.c1[0]) + 
                        x[1] * (self.c2[1] * y[1] + self.c1[1]) + 
                        pi_val * (self.resource_limit - x[0] * y[0] - x[1] * y[1]))
        
        # 计算L(x,π)函数对x的梯度，这将成为割平面的系数
        # 需要根据y的系数符号确定取y的上界还是下界
        gradient = np.zeros(2)
        
        # 针对x₁分量的梯度计算
        # 系数为self.c2[0]-pi_val = -1-π，表示y₁在L函数中的系数
        if self.c2[0]-pi_val < 0:
            # 如果系数为负，y₁取上界以最小化L
            gradient[0] = (self.c2[0]-pi_val) * self.y_upper + self.c1[0]
        else:
            # 如果系数为正或零，y₁取下界以最小化L
            gradient[0] = (self.c2[0]-pi_val) * self.y_lower + self.c1[0]
            
        # 针对x₂分量的梯度计算
        # 系数为self.c2[1]-pi_val = -0.8-π，表示y₂在L函数中的系数
        if self.c2[1]-pi_val < 0:
            # 如果系数为负，y₂取上界以最小化L
            gradient[1] = (self.c2[1]-pi_val) * self.y_upper + self.c1[1]
        else:
            # 如果系数为正或零，y₂取下界以最小化L
            gradient[1] = (self.c2[1]-pi_val) * self.y_lower + self.c1[1]

        # 将计算出的梯度作为割平面系数
        cut_coeffs = gradient
        
        # 计算割平面右侧常数项
        # 根据割平面方程：θ ≥ cut_coeffs·x + cut_rhs
        # 由于我们已知L(x,π) = cut_coeffs·x + cut_rhs，且已计算出current_value = L(x,π)
        # 因此：cut_rhs = current_value - cut_coeffs·x
        cut_rhs = current_value - gradient @ x
        
        # 输出本轮生成的割平面信息
        print(f"本轮生成割平面为:\n theta >= {cut_rhs} + {gradient[0]}*x1 + {gradient[1]}*x2 ")
        
        return cut_coeffs, cut_rhs

    
    def solve_master_problem(self, cuts: List[Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, float]:
        """
        求解受限主问题
        
        在GBD算法中，主问题是一个线性规划问题，目标是最小化θ，
        其中θ是递归函数的估计值。通过不断添加割平面约束，主问题逐步
        逼近原问题的真实解。
        
        主问题的形式为：
            min  θ
            s.t. x₁ + x₂ = 1
                 θ ≥ cut_coeffs·x + cut_rhs (对每个割平面)
                 x₁, x₂ ≥ 0
        
        参数:
            cuts: 割平面列表，每个元素是(系数, 右侧常数项)的元组
            
        返回:
            x_sol: 主问题的最优x解
            theta_val: 主问题的最优θ值，作为原问题的下界
        """
        try:
            # 创建Gurobi优化模型
            rm_model = gp.Model("VFPMaster")
            rm_model.setParam('OutputFlag', 0)  # 关闭Gurobi的输出信息
            
            # 添加决策变量
            x1 = rm_model.addVar(lb=0, name="x1")  # x₁变量，非负
            x2 = rm_model.addVar(lb=0, name="x2")  # x₂变量，非负
            theta = rm_model.addVar(lb=-GRB.INFINITY, name="theta")  # θ变量，表示递归函数的估计值
            
            # 设置目标函数：最小化θ
            rm_model.setObjective(theta, GRB.MINIMIZE)
            
            # 添加第一阶段约束：x₁ + x₂ = 1
            rm_model.addConstr(x1 + x2 == 1, name="sum_to_one")
            
            # 添加所有生成的Benders割平面约束
            # 每个割平面约束形式为：θ ≥ cut_rhs + cut_coeffs[0]*x₁ + cut_coeffs[1]*x₂
            for i, (cut_coeffs, cut_rhs) in enumerate(cuts):
                rm_model.addConstr(theta >= cut_rhs + cut_coeffs[0] * x1 + cut_coeffs[1] * x2, name=f"cut_{i}")
            
            # 更新模型并求解
            rm_model.update()
            rm_model.optimize()
            
            # 检查求解状态并提取结果
            if rm_model.Status == GRB.OPTIMAL:
                # 获取最优解
                x_sol = np.array([x1.X, x2.X])
                theta_val = theta.X
                
                return x_sol, theta_val
            else:
                # 如果无最优解，抛出异常
                raise ValueError(f"主问题无最优解，状态: {rm_model.Status}")
                
        except gp.GurobiError as e:
            print(f"主问题Gurobi错误: {str(e)}")
            raise
    
    def solve(self) -> Dict[str, Any]:
        """
        求解变量因子规划问题
        
        这是GBD算法的主要实现，按照以下步骤进行：
        1. 初始化一个可行的x解
        2. 对于每次迭代：
           a. 求解子问题，获取y和对偶变量π
           b. 如果找到更好的可行解，更新上界
           c. 生成并添加一个新的割平面
           d. 求解更新后的主问题，获取新的x和下界θ
           e. 检查收敛条件，如果满足则停止迭代
        
        返回:
            包含求解结果的字典，包括最优解、目标值、收敛信息等
        """
        # 首先初始化一个x，保证整个迭代的正常运行
        x = self.initial_x.copy()
        
        # 迭代计数器
        iteration = 0
        
        # 主循环：开始GBD算法迭代
        while iteration < self.max_iterations:
            iteration += 1
            if self.verbose:
                print('-'*100)
                print(f"开始第{iteration}迭代 ")
            
            try:
                # 步骤1：求解子问题
                # 对于当前的x值，求解子问题获取最优y、对偶变量π和目标值
                y, pi, obj_value = self.solve_subproblem(x)
                
                # 步骤2：更新上界
                # 如果找到更好的可行解，更新最优解和上界
                if obj_value < self.best_obj:
                    self.best_obj = obj_value
                    self.best_x = x.copy()
                    self.best_y = y.copy() 
                    self.upper_bound = obj_value
                    if self.verbose:
                        print(f"更新可行解，可行解为:\n x1 = {self.best_x[0]:.6f}, x2 = {self.best_x[1]:.6f}, y1 = {self.best_y[0]:.6f}, y2 = {self.best_y[1]:.6f}，当前UB为: {self.best_obj:.6f}")
                
                # 步骤3：生成并添加割平面
                # 使用当前的x、π和y生成一个新的Benders割平面
                cut_coeffs, cut_rhs = self.create_cut(x, pi, y)
                self.cuts.append((cut_coeffs, cut_rhs))
                
                # 步骤4：求解主问题
                # 利用所有已生成的割平面求解主问题，获取新的x和θ
                x, theta = self.solve_master_problem(self.cuts)
                self.lower_bound = theta  # 更新下界
                
                # 步骤5：记录历史和检查收敛
                # 计算当前迭代的上下界间隙
                gap = self.upper_bound - self.lower_bound
                rel_gap = gap / max(1e-10, abs(self.upper_bound))  # 相对间隙
                
                # 记录本次迭代的结果
                self.history.append({
                    'iteration': iteration,
                    'lower_bound': self.lower_bound,
                    'upper_bound': self.upper_bound,
                    'gap': gap,
                    'relative_gap': rel_gap
                })
                
                # 输出迭代信息
                if self.verbose:
                    print(f"LB: {self.lower_bound:.6f}, UB: {self.upper_bound:.6f}, Gap: {gap:.6f}, Rel Gap: {rel_gap:.6f}")
                    print(f'theta = {obj_value:.6f}')
                
                # 检查收敛条件：如果相对间隙小于容差，则停止迭代
                if rel_gap <= self.tolerance:
                    if self.verbose:
                        print('-'*100)
                        print(f"算法收敛于第{iteration}次迭代 .")
                    break
                
            except Exception as e:
                print(f"迭代{iteration}中发生错误: {str(e)}")
                break

        # 返回完整的求解结果
        return {
            'x': self.best_x,                  # 最优x解
            'y': self.best_y,                  # 最优y解
            'objective': self.best_obj,        # 最优目标函数值
            'lower_bound': self.lower_bound,   # 最终下界
            'upper_bound': self.upper_bound,   # 最终上界
            'gap': self.upper_bound - self.lower_bound,  # 最终间隙
            'relative_gap': (self.upper_bound - self.lower_bound) / max(1e-10, abs(self.upper_bound)),  # 最终相对间隙
            'iterations': iteration,           # 迭代次数
            'converged': iteration < self.max_iterations,  # 是否收敛
            'history': self.history            # 迭代历史
        }


# 示例函数: 求解VFP问题并可视化结果
def solve_and_visualize_vfp():
    """
    求解变量因子规划问题并可视化收敛过程
    
    这个函数创建变量因子规划问题实例，调用GBD算法求解问题，
    然后输出结果并绘制算法的收敛过程图。
    
    返回:
        求解结果字典
    """
    
    # 创建问题实例
    problem = VariableFactorProblem(
        max_iterations=10,    # 最大迭代次数设为10
        tolerance=1e-6,       # 收敛容差设为1e-6
        verbose=True          # 输出详细信息
    )
    
    # 求解问题
    print("开始求解VFP问题...")
    result = problem.solve()
    
    # 显示求解结果
    print(f"最优解: x = {result['x']}")
    print(f"最优值: {result['objective']:.6f}")
    print(f"迭代次数: {result['iterations']}")
    print(f"相对间隙: {result['relative_gap']:.6e}")
    
    # 绘制收敛过程图
    plt.figure(figsize=(10, 6))
    
    # 提取迭代历史数据
    iterations = [h['iteration'] for h in result['history']]
    lb = [h['lower_bound'] for h in result['history']]  # 下界历史
    ub = [h['upper_bound'] for h in result['history']]  # 上界历史
    
    # 绘制上下界变化曲线
    plt.plot(iterations, lb, 'b-', marker='o', label='下界')
    plt.plot(iterations, ub, 'r-', marker='s', label='上界')
    
    # 设置图表属性
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值')
    plt.title('GBD算法收敛过程')
    plt.legend()
    plt.grid(True)
    
    # 显示图表
    plt.show()
    
    return result

if __name__ == "__main__":
    # 求解变量因子规划问题并可视化结果
    vfp_result = solve_and_visualize_vfp()