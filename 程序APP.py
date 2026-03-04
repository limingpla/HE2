import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# 设置页面配置
st.set_page_config(page_title="Cognitive Impairment Prediction", layout="wide")

# 添加错误处理的模型加载
@st.cache_resource
def load_model():
    """加载模型并获取特征信息"""
    try:
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'RF.pkl')
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            alternative_paths = ['RF.pkl', './RF.pkl', os.path.join(os.getcwd(), 'RF.pkl')]
            for path in alternative_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            else:
                st.error(f"找不到模型文件。当前目录: {os.getcwd()}")
                st.write("目录中的文件:", os.listdir('.'))
                st.stop()
        
        # 加载模型
        model = joblib.load(model_path)
        
        # 获取模型的特征信息
        feature_info = {}
        
        # 方法1：从模型获取特征名称
        if hasattr(model, 'feature_names_in_'):
            feature_info['names'] = list(model.feature_names_in_)
            feature_info['source'] = 'model.feature_names_in_'
        # 方法2：尝试从训练数据获取
        elif hasattr(model, 'n_features_in_'):
            feature_info['count'] = model.n_features_in_
            feature_info['source'] = 'model.n_features_in_'
        else:
            feature_info['count'] = 8  # 假设8个特征
            feature_info['source'] = 'assumed'
        
        return model, feature_info
        
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

# 加载模型和特征信息
model, feature_info = load_model()

# 显示模型信息
st.sidebar.header("🔧 Model Information")
st.sidebar.write(f"模型类型: {type(model).__name__}")
st.sidebar.write(f"特征信息来源: {feature_info.get('source', '未知')}")

if 'names' in feature_info:
    st.sidebar.write("模型期望的特征:")
    for i, name in enumerate(feature_info['names']):
        st.sidebar.write(f"  {i+1}. {name}")
else:
    st.sidebar.write(f"模型期望的特征数量: {feature_info.get('count', '未知')}")

# 定义特征范围（使用你的原始特征名称）
feature_ranges = {
    "D_D": {"type": "numerical", "min": 0.18, "max": 47.17, "default": 2.0},
    "CRP": {"type": "numerical", "min": 3.2, "max": 360, "default": 10.0},
    "INR": {"type": "numerical", "min": 0.91, "max": 8.99, "default": 2.0},
    "WBC": {"type": "numerical", "min": 0.1, "max": 133.67, "default": 10.0},
    "FDP": {"type": "numerical", "min": 0.23, "max": 195.8, "default": 12.0},
    "Lactate": {"type": "numerical", "min": 0.3, "max": 20, "default": 4.0},
    "PTA": {"type": "numerical", "min": 9, "max": 143, "default": 40.0},
    "APTT": {"type": "numerical", "min": 20, "max": 180, "default": 80.0},
}

# Streamlit 界面
st.title("🧠 Cognitive Impairment Prediction Model")
st.markdown("---")

# 侧边栏输入
with st.sidebar:
    st.header("📊 Input Features")
    st.markdown("Enter the following feature values:")
    
    # 收集用户输入
    input_dict = {}
    for feature, properties in feature_ranges.items():
        value = st.number_input(
            label=f"**{feature}**",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            format="%.2f",
            help=f"Range: {properties['min']} - {properties['max']}"
        )
        input_dict[feature] = value

# 主界面
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Summary")
    input_df_display = pd.DataFrame([input_dict])
    st.dataframe(input_df_display, use_container_width=True)

# 预测按钮
if st.sidebar.button("🔍 Predict", type="primary", use_container_width=True):
    with st.spinner("Calculating prediction and SHAP values..."):
        try:
            # 方法1：如果模型有特征名称，使用它们
            if 'names' in feature_info:
                model_features = feature_info['names']
                
                # 创建与模型期望完全匹配的DataFrame
                feature_values = []
                for feat in model_features:
                    if feat in input_dict:
                        feature_values.append(input_dict[feat])
                    else:
                        # 如果模型期望的特征不在输入中，使用默认值
                        st.warning(f"特征 '{feat}' 不在输入中，使用0作为默认值")
                        feature_values.append(0.0)
                
                # 创建带正确列名的DataFrame
                feature_df = pd.DataFrame([feature_values], columns=model_features)
                
            else:
                # 方法2：如果没有特征名称，使用numpy数组
                feature_values = [input_dict[feat] for feat in feature_ranges.keys()]
                feature_df = np.array([feature_values])
            
            # 显示调试信息
            with st.expander("🔧 Debug Info"):
                st.write("模型类型:", type(model).__name__)
                if 'names' in feature_info:
                    st.write("模型特征名称:", feature_info['names'])
                    st.write("输入数据列名:", list(feature_df.columns))
                else:
                    st.write("输入数据形状:", feature_df.shape)
            
            # 模型预测
            predicted_class = model.predict(feature_df)[0]
            predicted_proba = model.predict_proba(feature_df)[0]
            
            # 获取类别标签
            if hasattr(model, 'classes_'):
                class_labels = [str(cls) for cls in model.classes_]
            else:
                class_labels = ["No Impairment", "Cognitive Impairment"]
            
            predicted_label = class_labels[predicted_class]
            probability = predicted_proba[predicted_class] * 100
            
            with col2:
                st.subheader("🎯 Prediction Result")
                
                # 显示预测结果
                if predicted_class == 1:
                    st.error(f"### ⚠️ {predicted_label}")
                else:
                    st.success(f"### ✅ {predicted_label}")
                
                # 显示概率
                st.metric("Probability", f"{probability:.1f}%")
                
                # 显示所有类别的概率
                st.write("**Class Probabilities:**")
                prob_df = pd.DataFrame({
                    'Class': class_labels,
                    'Probability': [f"{p*100:.1f}%" for p in predicted_proba]
                })
                st.dataframe(prob_df, use_container_width=True)
            
            # SHAP分析
            try:
                explainer = shap.TreeExplainer(model)
                
                if isinstance(feature_df, pd.DataFrame):
                    shap_values = explainer.shap_values(feature_df)
                else:
                    # 如果是numpy数组，转换为DataFrame以便显示
                    shap_values = explainer.shap_values(feature_df)
                    feature_df_display = pd.DataFrame(feature_df, columns=list(feature_ranges.keys()))
                
                # 处理SHAP值
                if isinstance(shap_values, list):
                    shap_plot_values = shap_values[predicted_class]
                    if isinstance(explainer.expected_value, list):
                        expected_value = explainer.expected_value[predicted_class]
                    else:
                        expected_value = explainer.expected_value
                else:
                    shap_plot_values = shap_values
                    expected_value = explainer.expected_value
                
                # 显示特征重要性
                st.subheader("📊 Feature Importance")
                
                if isinstance(shap_plot_values, np.ndarray):
                    # 创建特征重要性DataFrame
                    if isinstance(feature_df, pd.DataFrame):
                        feature_names = feature_df.columns
                    else:
                        feature_names = list(feature_ranges.keys())
                    
                    # 确保长度匹配
                    if len(shap_plot_values[0]) == len(feature_names):
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'SHAP Value': shap_plot_values[0],
                            '|SHAP|': np.abs(shap_plot_values[0])
                        }).sort_values('|SHAP|', ascending=True)
                        
                        # 创建图表
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['red' if x > 0 else 'blue' for x in importance_df['SHAP Value']]
                        bars = ax.barh(importance_df['Feature'], importance_df['|SHAP|'], color=colors, alpha=0.7)
                        ax.set_xlabel('|SHAP Value| (Impact on Prediction)')
                        ax.set_title('SHAP Feature Impact')
                        
                        # 添加数值标签
                        for bar, val in zip(bars, importance_df['SHAP Value']):
                            width = bar.get_width()
                            ax.text(width, bar.get_y() + bar.get_height()/2, 
                                   f'{val:.3f}', ha='left', va='center', fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # 显示详细SHAP值
                        with st.expander("📋 View Detailed SHAP Values"):
                            st.dataframe(importance_df[['Feature', 'SHAP Value', '|SHAP|']], use_container_width=True)
                    else:
                        st.warning(f"特征数量不匹配: SHAP值数量={len(shap_plot_values[0])}, 特征数量={len(feature_names)}")
                
            except Exception as e:
                st.warning(f"SHAP分析不可用: {str(e)}")
                
                # 备用：显示模型特征重要性
                if hasattr(model, 'feature_importances_'):
                    st.subheader("📊 Model Feature Importance")
                    
                    if isinstance(feature_df, pd.DataFrame):
                        feature_names = feature_df.columns
                    else:
                        feature_names = list(feature_ranges.keys())
                    
                    # 确保长度匹配
                    if len(model.feature_importances_) == len(feature_names):
                        imp_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=True)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.barh(imp_df['Feature'], imp_df['Importance'])
                        ax.set_xlabel('Feature Importance')
                        ax.set_title('Random Forest Feature Importance')
                        
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width, bar.get_y() + bar.get_height()/2, 
                                   f'{width:.3f}', ha='left', va='center', fontsize=10)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        with st.expander("📋 View Detailed Feature Importance"):
                            st.dataframe(imp_df, use_container_width=True)
                    else:
                        st.write(f"特征重要性数量: {len(model.feature_importances_)}")
                        st.write(f"特征名称数量: {len(feature_names)}")
                    
        except Exception as e:
            st.error(f"预测失败: {str(e)}")
            st.exception(e)

# 添加页脚
st.markdown("---")
st.markdown("📌 *This is a machine learning prediction tool. Please consult healthcare professionals for clinical decisions.*")
