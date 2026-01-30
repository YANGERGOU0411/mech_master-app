import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from math import pi, sqrt, ceil, floor, cos, sin, radians, tan
import os

# ==============================================================================
# 0. 全局系统配置 (System Config)
# ==============================================================================
st.set_page_config(
    page_title="冶金与机械设计综合计算平台 (Pro)",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

# --- 样式注入 ---
st.markdown("""
<style>
    .main-header {font-size: 24px; font-weight: bold; color: #1E3A8A; border-bottom: 2px solid #1E3A8A; padding-bottom: 10px; margin-bottom: 20px;}
    .sub-header {font-size: 18px; font-weight: bold; color: #4B5563; margin-top: 15px;}
    .info-box {background-color: #EFF6FF; padding: 15px; border-radius: 8px; border-left: 5px solid #3B82F6;}
    .warning-box {background-color: #FEF2F2; padding: 15px; border-radius: 8px; border-left: 5px solid #EF4444;}
    .success-box {background-color: #ECFDF5; padding: 15px; border-radius: 8px; border-left: 5px solid #10B981;}
</style>
""", unsafe_allow_html=True)

# --- 字体加载 ---
@st.cache_resource
def configure_fonts():
    # 优先加载上传的 SimHei，否则尝试系统字体
    font_candidates = ["SimHei.ttf", "simhei.ttf", "msyh.ttc", "simsun.ttc"]
    found_font = None
    for f in font_candidates:
        if os.path.exists(f):
            found_font = f
            break
    
    if found_font:
        fm.fontManager.addfont(found_font)
        prop = fm.FontProperties(fname=found_font)
        return prop.get_name(), True
    else:
        return "sans-serif", False

font_family, is_font_success = configure_fonts()
plt.rcParams['font.sans-serif'] = [font_family, 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 核心数据库 (The "Brain" - Digested from your 5 Books & Excels)
# ==============================================================================

# [矿热炉] 经验系数库 (源自您的Excel)
FURNACE_DB = {
    "硅锰 (SiMn)":     {"Ke": 6.3,  "J": 5.5, "Ky": 2.7,  "Ki": 6.4,  "Kh": 2.5, "rho": 1658},
    "高碳铬铁 (FeCr)": {"Ke": 6.8,  "J": 5.7, "Ky": 2.65, "Ki": 6.3,  "Kh": 2.6, "rho": 2156},
    "镍铁 (FeNi-RKEF)":{"Ke": 12.0, "J": 4.0, "Ky": 3.6,  "Ki": 10.0, "Kh": 2.9, "rho": 2500},
    "硅铁75 (FeSi75)": {"Ke": 6.8,  "J": 6.5, "Ky": 2.25, "Ki": 5.8,  "Kh": 2.2, "rho": 1200},
    "电石 (CaC2)":     {"Ke": 6.5,  "J": 7.0, "Ky": 2.7,  "Ki": 6.4,  "Kh": 2.2, "rho": 1800},
    "工业硅 (Si)":     {"Ke": 7.5,  "J": 6.0, "Ky": 2.4,  "Ki": 6.0,  "Kh": 2.3, "rho": 1000},
    "自定义":          {"Ke": 6.5,  "J": 5.5, "Ky": 2.7,  "Ki": 6.5,  "Kh": 2.5, "rho": 2000}
}

# [手册卷1] 常用材料力学性能 (GB/T 699, GB/T 3077)
MATERIAL_DB = pd.DataFrame({
    "材料牌号": ["Q235-A", "45钢 (调质)", "40Cr (调质)", "35SiMn (调质)", "20CrMnTi (渗碳)", "42CrMo (调质)"],
    "抗拉强度 σb (MPa)": [370, 600, 785, 885, 1080, 1080],
    "屈服强度 σs (MPa)": [235, 355, 540, 735, 835, 930],
    "硬度 (HB)": [140, 240, 260, 270, 600, 290],
    "轴设计系数 A0": [130, 118, 110, 105, 100, 100]
}).set_index("材料牌号")

# [手册卷2] 螺纹标准 (GB/T 196) - 部分常用数据
THREAD_DB = pd.DataFrame({
    "规格": [6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 48, 56, 64],
    "螺距 P": [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
    "中径 d2": [5.35, 7.18, 9.02, 10.86, 14.70, 18.37, 22.05, 27.72, 33.40, 39.07, 44.75, 52.42, 60.10],
    "小径 d1": [4.91, 6.64, 8.37, 10.10, 13.83, 17.29, 20.75, 26.21, 31.67, 37.12, 42.58, 50.04, 57.50],
    "应力截面 As": [20.1, 36.6, 58.0, 84.3, 157, 245, 353, 561, 817, 1120, 1470, 2030, 2676]
}).set_index("规格")

# [手册卷4] Y2系列电机简表 (同步转速1500rpm, 4极)
MOTOR_DB = pd.DataFrame({
    "功率 (kW)": [0.75, 1.1, 1.5, 2.2, 3, 4, 5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45, 55],
    "型号": ["Y2-80M2-4", "Y2-90S-4", "Y2-90L-4", "Y2-100L1-4", "Y2-100L2-4", "Y2-112M-4", "Y2-132S-4", 
             "Y2-132M-4", "Y2-160M-4", "Y2-160L-4", "Y2-180M-4", "Y2-180L-4", "Y2-200L-4", "Y2-225S-4", "Y2-225M-4", "Y2-250M-4"],
    "轴伸直径 D (mm)": [19, 24, 24, 28, 28, 38, 38, 38, 42, 42, 48, 48, 55, 60, 60, 65]
})

# ==============================================================================
# 2. 辅助计算函数 (Logic Layer)
# ==============================================================================

def recommend_key(d):
    """[手册卷2] 键槽GB/T 1096推荐"""
    if d <= 12: return 4, 4
    elif d <= 17: return 5, 5
    elif d <= 22: return 6, 6
    elif d <= 30: return 8, 7
    elif d <= 38: return 10, 8
    elif d <= 44: return 12, 8
    elif d <= 50: return 14, 9
    elif d <= 58: return 16, 10
    elif d <= 65: return 18, 11
    elif d <= 75: return 20, 12
    elif d <= 85: return 22, 14
    elif d <= 95: return 25, 14
    elif d <= 110: return 28, 16
    else: return 32, 18

def calc_gear_module(T, z1, K=1.3, phi_d=1.0, sigma_H=600):
    """[手册卷3] 齿轮模数估算 (基于接触强度)"""
    # 简化经验公式: m >= K * (T / z1)^(1/3) 
    # 实际上更复杂的公式可以通过 T 和 sigma_H 反推 d1，再求 m
    # 这里用工程常用的快速估算法
    # d1 >= 76.6 * ((T*K*(u+1))/(phi_d * u * sigma_H^2))^(1/3)
    # 此处仅做演示级算法
    m_calc = 1.6 * (T / z1) ** (1/3)
    return m_calc

# ==============================================================================
# 3. 界面逻辑：主侧边栏导航
# ==============================================================================

with st.sidebar:
    st.title("🏭 综合设计平台")
    st.markdown("---")
    
    app_mode = st.radio("请选择设计系统:", [
        "🔥 矿热电炉设计系统 (Excel核心)",
        "🏭 铁水包/渣罐设计 (几何核心)",
        "📘 机械设计手册 (Vol.1-5)"
    ])
    
    st.markdown("---")
    if is_font_success:
        st.success(f"✅ 字体就绪: {font_family}")
    else:
        st.error("❌ 字体缺失 (SimHei.ttf)")
        
    st.info("数据来源：\n1. 企业内部Excel计算表\n2. 《机械设计手册》第六版")

# ==============================================================================
# 🔴 系统一：矿热电炉设计 (Deep Furnace Logic)
# ==============================================================================
if app_mode == "🔥 矿热电炉设计系统 (Excel核心)":
    
    st.markdown("<div class='main-header'>🔥 矿热电炉全参数计算与选型平台</div>", unsafe_allow_html=True)
    
    # --- 状态管理 ---
    if 'f_recalc' not in st.session_state: st.session_state.f_recalc = True
    def trigger_f(): st.session_state.f_recalc = True

    # --- 输入区 ---
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.markdown("<div class='sub-header'>1. 基础工况</div>", unsafe_allow_html=True)
        alloy = st.selectbox("冶炼品种", list(FURNACE_DB.keys()), on_change=trigger_f)
        
        col_in1, col_in2 = st.columns(2)
        cap_mva = col_in1.number_input("变压器容量 (MVA)", 1.0, 100.0, 33.0, 0.5, on_change=trigger_f)
        u1_kv = col_in2.selectbox("一次电压 (kV)", [110, 35, 10, 6, 220], index=1, on_change=trigger_f)
        
        st.markdown("<div class='sub-header'>2. 导电系统 (铜瓦/铜管)</div>", unsafe_allow_html=True)
        tile_n = st.number_input("铜瓦数量 (块/相)", 4, 16, 8)
        tube_d = st.selectbox("铜管外径 (mm)", [50,60,70,80,90,100], index=2)
        tube_t = st.selectbox("铜管壁厚 (mm)", [10,12.5,15,20], index=1)
        tube_n = tile_n * 2
        st.caption(f"📐 自动匹配：铜管数量 = {tube_n} 根/相 (2:1)")

        st.markdown("<div class='sub-header'>3. 经验系数 (Expert)</div>", unsafe_allow_html=True)
        defs = FURNACE_DB[alloy]
        ke = st.slider("电压系数 Ke", 1.0, 15.0, defs['Ke'], 0.1, on_change=trigger_f)
        j_val = st.slider("电流密度 J", 1.0, 10.0, defs['J'], 0.1, on_change=trigger_f)
        ky = st.number_input("极心圆系数 Ky", value=defs['Ky'], step=0.05, on_change=trigger_f)
        ki = st.number_input("炉膛内径系数 Ki", value=defs['Ki'], step=0.1, on_change=trigger_f)
        kh = st.number_input("炉膛深度系数 Kh", value=defs['Kh'], step=0.1, on_change=trigger_f)
        lining = st.number_input("炉衬厚度 (mm)", value=1200, step=100, on_change=trigger_f)

    # --- 计算逻辑 ---
    p_kva = cap_mva * 1000
    i1_th = p_kva * 1000 / (1.732 * u1_kv * 1000)
    u2_th = ke * (p_kva ** (1/3))
    i2_th = p_kva * 1000 / (1.732 * u2_th)
    
    de_th = sqrt(i2_th / j_val / 0.7854) * 10
    dc_th = ky * de_th
    di_th = ki * de_th
    hh_th = kh * de_th
    shell_id_th = di_th + 2 * lining
    shell_h_th = hh_th + 2000

    # --- 圆整初始化 ---
    if st.session_state.f_recalc:
        st.session_state.r_u2 = round(u2_th)
        st.session_state.r_de = round(de_th/50)*50
        st.session_state.r_dc = round((st.session_state.r_de * ky)/50)*50
        st.session_state.r_di = round((st.session_state.r_de * ki)/100)*100
        st.session_state.r_hh = round((st.session_state.r_de * kh)/100)*100
        st.session_state.r_shell_id = st.session_state.r_di + 2 * lining
        st.session_state.r_shell_h = st.session_state.r_hh + 2000
        st.session_state.f_recalc = False

    def update_furnace_dims():
        d = st.session_state.in_de_val
        st.session_state.r_de = d
        st.session_state.r_dc = round((d * ky)/50)*50
        st.session_state.r_di = round((d * ki)/100)*100
        st.session_state.r_hh = round((d * kh)/100)*100
        st.session_state.r_shell_id = st.session_state.r_di + 2 * lining
        st.session_state.r_shell_h = st.session_state.r_hh + 2000

    with c2:
        st.markdown("<div class='sub-header'>4. 结果分析与工程修正</div>", unsafe_allow_html=True)
        
        # 结果表
        res_cols = st.columns([2, 2, 2])
        res_cols[0].markdown("**参数**")
        res_cols[1].markdown("**理论值**")
        res_cols[2].markdown("**圆整值 (可改)**")
        
        # U2
        res_cols[0].write("二次电压 U₂ (V)")
        res_cols[1].write(f"{u2_th:.1f}")
        fin_u2 = res_cols[2].number_input("U2", value=int(st.session_state.r_u2), label_visibility="collapsed")
        
        # I2
        fin_i2 = p_kva*1000 / (1.732*fin_u2)
        res_cols[0].write("二次电流 I₂ (A)")
        res_cols[1].write(f"{i2_th:.0f}")
        res_cols[2].info(f"{fin_i2:.0f}")
        
        # De
        res_cols[0].write("电极直径 De (mm)")
        res_cols[1].write(f"{de_th:.0f}")
        fin_de = res_cols[2].number_input("De", value=float(st.session_state.r_de), step=10.0, key="in_de_val", on_change=update_furnace_dims, label_visibility="collapsed")
        
        # Di
        res_cols[0].write("炉膛内径 Di (mm)")
        res_cols[1].write(f"{di_th:.0f}")
        fin_di = res_cols[2].number_input("Di", value=float(st.session_state.r_di), step=100.0, key="in_di_val", label_visibility="collapsed")
        
        # Shell
        res_cols[0].write("炉壳内径 (mm)")
        res_cols[1].write(f"{shell_id_th:.0f}")
        fin_shell = res_cols[2].number_input("Shell", value=float(st.session_state.r_shell_id), step=100.0, key="in_shell_val", label_visibility="collapsed")

        # 绘图
        st.markdown("---")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        # Shell
        ax.add_patch(patches.Rectangle((-fin_shell/2, 0), fin_shell, st.session_state.r_shell_h, lw=3, ec='#333', fc='none', label='炉壳'))
        # Hearth
        ax.add_patch(patches.Rectangle((-fin_di/2, 1500), fin_di, st.session_state.r_hh, lw=2, ec='red', fc='#FEF3C7', alpha=0.5, label='熔池'))
        # Electrode
        dc = st.session_state.r_dc
        ew = fin_de
        eh = st.session_state.r_shell_h * 0.7
        ax.add_patch(patches.Rectangle((-dc/2 - ew/2, st.session_state.r_shell_h/2), ew, eh, color='#4B5563', label='电极'))
        ax.add_patch(patches.Rectangle((dc/2 - ew/2, st.session_state.r_shell_h/2), ew, eh, color='#4B5563'))
        
        # Annotations
        ax.plot([-dc/2, dc/2], [st.session_state.r_shell_h+200, st.session_state.r_shell_h+200], color='blue', marker='|')
        ax.text(0, st.session_state.r_shell_h+400, f"极心圆 {dc:.0f}", ha='center', color='blue')
        
        ax.set_xlim(-fin_shell/1.5, fin_shell/1.5)
        ax.set_ylim(-1000, st.session_state.r_shell_h + 2000)
        ax.axis('off')
        ax.set_title(f"{alloy} {cap_mva}MVA 矿热炉结构示意", fontsize=12)
        ax.legend(loc='upper right')
        st.pyplot(fig)
        
        # 下载
        exp_data = pd.DataFrame([
            ["变压器容量", cap_mva, "MVA"],
            ["一次电压", u1_kv, "kV"],
            ["一次电流", i1_th, "A"],
            ["二次电压 (圆整)", fin_u2, "V"],
            ["二次电流 (圆整)", fin_i2, "A"],
            ["电极直径", fin_de, "mm"],
            ["极心圆直径", st.session_state.r_dc, "mm"],
            ["炉膛内径", fin_di, "mm"],
            ["炉壳内径", fin_shell, "mm"],
            ["铜瓦数量", tile_n, "块/相"],
            ["铜管配置", f"{tube_n}根 Φ{tube_d}×{tube_t}", "-"]
        ], columns=["项目", "数值", "单位"])
        csv = exp_data.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 导出计算书", csv, f"Furnace_{cap_mva}MVA.csv")

# ==============================================================================
# 🔵 系统二：铁水包设计 (Ladle Design)
# ==============================================================================
elif app_mode == "🏭 铁水包/渣罐设计 (几何核心)":
    
    st.markdown("<div class='main-header'>🏭 铁水包/渣罐 智能设计系统</div>", unsafe_allow_html=True)
    
    if 'ar' not in st.session_state: st.session_state.ar = 1.05
    def up_ar_s(): st.session_state.ar = st.session_state.ar_slide
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='sub-header'>1. 几何参数</div>", unsafe_allow_html=True)
        vol = st.number_input("有效容积 (m³)", 0.5, 50.0, 4.5, 0.1)
        rho = st.number_input("介质密度 (t/m³)", 1.0, 8.0, 7.0)
        freeboard = st.number_input("净空高度 (mm)", 100, 1000, 300)
        
        st.markdown("---")
        st.write("**径高比 (D/H)**")
        st.slider("粗调", 0.5, 2.0, 1.05, 0.01, key='ar_slide', on_change=up_ar_s)
        st.number_input("精调", 0.5, 2.0, st.session_state.ar, 0.01, key='ar')
        
        st.markdown("---")
        angle = st.number_input("侧壁倾角 (°)", 0.0, 15.0, 5.0)
        t_wall = st.number_input("壁厚 (mm)", 50, 500, 160)
        t_bot = st.number_input("底厚 (mm)", 50, 500, 230)

    # 迭代求解 H
    ar = st.session_state.ar
    tan_a = tan(radians(angle))
    
    def calc_vol(h):
        # 简化圆台计算
        h_liq = h - t_bot/1000 - freeboard/1000
        if h_liq <= 0: return 0
        r_top = (ar * h)/2
        r_bot = r_top - h * tan_a
        if r_bot <= 0: return 0
        
        # 液体部分近似
        r_liq_top = r_top - t_wall/1000
        r_liq_bot = (r_bot + t_bot/1000 * tan_a) - t_wall/1000
        if r_liq_bot <= 0: return 0
        
        return (1/3) * pi * h_liq * (r_liq_bot**2 + r_liq_top**2 + r_liq_bot*r_liq_top)

    # 二分查找
    low, high = 0.5, 10.0
    for _ in range(50):
        mid = (low+high)/2
        if calc_vol(mid) < vol: low = mid
        else: high = mid
    
    H_final = high
    H_mm = H_final * 1000
    D_top_mm = H_mm * ar
    D_bot_mm = D_top_mm - 2 * H_mm * tan_a
    Cap_ton = vol * rho

    with col2:
        st.markdown("<div class='sub-header'>2. 设计图纸</div>", unsafe_allow_html=True)
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("总高度 H", f"{H_mm:.0f} mm")
        k2.metric("上口外径", f"{D_top_mm:.0f} mm")
        k3.metric("计算载重", f"{Cap_ton:.1f} t")
        k4.metric("液面深度", f"{H_mm - t_bot - freeboard:.0f} mm")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Shell
        x = [0, D_bot_mm/2, D_top_mm/2, 0]
        y = [0, 0, H_mm, H_mm]
        ax.add_patch(patches.Polygon(list(zip(x, y)), closed=True, fc='none', ec='black', lw=2))
        ax.add_patch(patches.Polygon(list(zip([-i for i in x], y)), closed=True, fc='none', ec='black', lw=2))
        
        # Liquid
        h_liq = H_mm - t_bot - freeboard
        liq_y = [t_bot, t_bot, t_bot+h_liq, t_bot+h_liq]
        r_l_b = (D_bot_mm/2) - t_wall + (t_bot * tan_a)
        r_l_t = (D_top_mm/2) - t_wall - (freeboard * tan_a)
        liq_x = [0, r_l_b, r_l_t, 0]
        ax.add_patch(patches.Polygon(list(zip(liq_x, liq_y)), closed=True, fc='orange', alpha=0.5))
        ax.add_patch(patches.Polygon(list(zip([-i for i in liq_x], liq_y)), closed=True, fc='orange', alpha=0.5))
        
        # Dimensions
        ax.annotate(f"H={H_mm:.0f}", xy=(-D_top_mm/1.5, H_mm/2), ha='center')
        ax.plot([-D_top_mm/2, D_top_mm/2], [H_mm, H_mm], 'k--')
        
        ax.set_xlim(-D_top_mm, D_top_mm)
        ax.set_ylim(-500, H_mm+500)
        ax.axis('off')
        st.pyplot(fig)

# ==============================================================================
# 📚 系统三：机械设计手册 (Mechanical Design Handbook System)
# ==============================================================================
elif app_mode == "📘 机械设计手册 (Vol.1-5)":
    
    st.markdown("<div class='main-header'>📘 机械设计手册数字化专家系统</div>", unsafe_allow_html=True)
    
    # 使用 Tabs 分割5卷内容
    tabs = st.tabs([
        "Vol.1 常用材料", 
        "Vol.2 连接与轴系", 
        "Vol.3 齿轮传动", 
        "Vol.4 电机选型", 
        "Vol.5 液压传动"
    ])
    
    # --- Tab 1: 材料 ---
    with tabs[0]:
        st.markdown("#### 🧪 常用工程材料库")
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            search_text = st.text_input("🔍 搜索材料 (如: 45, Q235)", "")
        with col_m2:
            df_view = MATERIAL_DB
            if search_text:
                df_view = MATERIAL_DB[MATERIAL_DB.index.str.contains(search_text)]
            st.dataframe(df_view, use_container_width=True)
            st.caption("注：数据基于《机械设计手册》第1卷 常用材料篇")

    # --- Tab 2: 轴与连接 ---
    with tabs[1]:
        st.markdown("#### 🔩 轴系设计向导")
        c1, c2 = st.columns(2)
        with c1:
            st.info("步骤1: 轴径估算")
            P_shaft = st.number_input("传递功率 P (kW)", 1.0, 5000.0, 15.0)
            n_shaft = st.number_input("转速 n (r/min)", 1.0, 10000.0, 960.0)
            mat_shaft = st.selectbox("轴材料", MATERIAL_DB.index.tolist())
            
            A0 = MATERIAL_DB.loc[mat_shaft, "轴设计系数 A0"]
            d_min = A0 * (P_shaft/n_shaft)**(1/3)
            d_design = ceil(d_min * 1.05 / 5) * 5 # 圆整到5
            
            st.metric("估算最小轴径 (含键槽)", f"{d_design} mm", help=f"A0={A0}")
            
        with c2:
            st.info("步骤2: 键槽选择 (GB/T 1096)")
            d_final = st.number_input("确定轴径 d (mm)", value=int(d_design))
            b_key, h_key = recommend_key(d_final)
            t1 = h_key/2 + 0.2
            
            col_k1, col_k2 = st.columns(2)
            col_k1.metric("键宽 b", f"{b_key} mm")
            col_k2.metric("键高 h", f"{h_key} mm")
            st.caption(f"轴上槽深 t1 ≈ {t1:.1f} mm")
            
        st.divider()
        st.markdown("#### 🔗 螺纹连接强度")
        load_F = st.number_input("轴向拉力 F (N)", 1000.0, 100000.0, 5000.0)
        spec = st.selectbox("螺纹规格", THREAD_DB.index.tolist(), index=4) # M16
        grade = st.selectbox("性能等级", ["4.8", "8.8", "10.9", "12.9"], index=1)
        
        As = THREAD_DB.loc[spec, "应力截面 As"]
        sigma_s = float(grade.split('.')[0]) * 100 * (float(grade.split('.')[1])/10)
        sigma_cal = (load_F * 1.3) / As # 预紧系数1.3
        safe = sigma_s / sigma_cal
        
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("螺栓应力", f"{sigma_cal:.1f} MPa")
        cc2.metric("屈服极限", f"{sigma_s:.0f} MPa")
        cc3.metric("安全系数", f"{safe:.2f}", delta="合格" if safe>1.5 else "不合格", delta_color="normal")

    # --- Tab 3: 齿轮 ---
    with tabs[2]:
        st.markdown("#### ⚙️ 齿轮传动设计 (接触强度法)")
        gc1, gc2 = st.columns(2)
        with gc1:
            T_gear = st.number_input("小齿轮扭矩 T1 (N.m)", 100.0, 50000.0, 500.0)
            u_ratio = st.number_input("传动比 u", 1.0, 10.0, 4.0)
            hard = st.radio("齿面硬度", ["软齿面", "硬齿面"])
        
        with gc2:
            z1 = 20 # 默认
            z2 = int(z1 * u_ratio)
            # 估算模数
            m_min = calc_gear_module(T_gear, z1)
            # 标准模数序列
            std_m = [1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16]
            m_final = min([x for x in std_m if x >= m_min], default=20)
            
            a_center = m_final * (z1 + z2) / 2
            
            st.success(f"建议模数 m = {m_final} mm")
            st.info(f"中心距 a = {a_center} mm")
            st.json({"小齿轮齿数": z1, "大齿轮齿数": z2, "分度圆 d1": m_final*z1, "分度圆 d2": m_final*z2})

    # --- Tab 4: 电机 ---
    with tabs[3]:
        st.markdown("#### 🔌 电机自动选型 (Y2系列)")
        req_power = st.number_input("负载功率 (kW)", 0.1, 100.0, 4.5)
        
        # 查找刚好大于需求的电机
        valid_motors = MOTOR_DB[MOTOR_DB["功率 (kW)"] >= req_power]
        
        if not valid_motors.empty:
            rec_motor = valid_motors.iloc[0]
            st.success(f"推荐型号: **{rec_motor['型号']}**")
            
            mc1, mc2 = st.columns(2)
            mc1.metric("额定功率", f"{rec_motor['功率 (kW)']} kW")
            mc2.metric("轴伸直径 D", f"{rec_motor['轴伸直径 D (mm)']} mm")
            
            st.table(valid_motors.head(3))
        else:
            st.warning("未找到匹配电机，请检查功率范围。")

    # --- Tab 5: 液压 ---
    with tabs[4]:
        st.markdown("#### 💧 液压缸推力计算")
        hc1, hc2 = st.columns(2)
        with hc1:
            pressure = st.slider("系统压力 P (MPa)", 1.0, 31.5, 16.0)
            diameter = st.selectbox("缸径 D (mm)", [40, 50, 63, 80, 100, 125, 160, 200, 250])
        
        with hc2:
            area = pi * (diameter/2)**2
            force_kn = pressure * area / 1000
            st.metric("理论推力 F", f"{force_kn:.1f} kN")
            st.caption(f"有效作用面积: {area:.0f} mm²")