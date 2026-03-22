"""
画图配置模块
提供统一的配色方案和画图风格配置
"""

# numuav=20/30/40/50的统一配色方案
# 可修改此处来测试不同的配色方案
NUMUAV_COLORS = {
    50: '#E63946',  # 红色
    40: '#457B9D',  # 蓝色
    30: '#2A9D8F',  # 青绿色
    20: '#E9C46A',  # 黄色
}

# baseline的配色方案
BASELINE_COLORS = {
    'random': '#B0B0B0',        # 灰色
    'simple_greedy': '#5589F2', # 深蓝
    'local_greedy': '#8BBCFF',  # 中蓝
    'greedy_spatial': '#C4D8F1', # 浅蓝
    'spatial_dl': '#FFAA89',    # 中红
    'fplinq': '#f7D2C3',        # 浅红
    'proposed GAT': '#FF6950',          # 深红 (我们的方法)
}

# 画图风格设置 - 适配论文格式
# 论文常用字号：正文10-12pt，标题12-14pt，图例9-10pt

# 论文双栏宽度适配（约7英寸，适合单栏）
PAPER_SINGLE_COLUMN_WIDTH = 7.0  # inches
PAPER_DOUBLE_COLUMN_WIDTH = 14.0  # inches (full width)

PLOT_STYLE = {
    'font.family': 'sans-serif',
    'font.size': 10,           # 正文字号
    'axes.titlesize': 12,      # 标题字号 (正文+2)
    'axes.labelsize': 10,      # 坐标轴标签字号 (正文)
    'legend.fontsize': 10,     # 图例字号 (正文)
    'xtick.labelsize': 10,      # x轴刻度字号 (正文-1)
    'ytick.labelsize': 10,      # y轴刻度字号 (正文-1)
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.format': 'eps',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 1.5,    # 线条宽度
    'axes.linewidth': 0.8,     # 坐标轴线宽
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
}

# 平滑曲线参数 - 改进版本：避免过度平滑
SMOOTH_PARAMS = {
    'window': 5,           # 减小窗口大小 (原为10)
    'method': 'savgol',    # 改用Savitzky-Golay滤波器 (保留峰值)
    'ema_alpha': 0.2,      # EMA系数 (越小越平滑)
    'shadow_alpha': 0.15,  # 减小阴影透明度 (原为0.3)
    'raw_line_alpha': 0.25, # 原始曲线透明度
}

# 分段平滑参数（前10个epoch用更小的窗口）
SMOOTH_PARAMS_EARLY = {
    'window': 3,           # 早期用更小的窗口
    'method': 'savgol',
    'ema_alpha': 0.15,
    'shadow_alpha': 0.1,
    'raw_line_alpha': 0.3,
}

def get_numuav_color(numuav: int) -> str:
    """获取numuav对应的颜色"""
    return NUMUAV_COLORS.get(numuav, '#333333')

def get_all_numuav_colors() -> dict:
    """获取所有numuav的配色"""
    return NUMUAV_COLORS.copy()

def update_colors(new_colors: dict):
    """更新配色方案
    
    Args:
        new_colors: 新的配色字典，例如 {'20': '#FF0000', '30': '#00FF00'}
    """
    global NUMUAV_COLORS
    for key, value in new_colors.items():
        if isinstance(key, str) and key.isdigit():
            key = int(key)
        NUMUAV_COLORS[key] = value
