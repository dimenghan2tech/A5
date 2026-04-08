# 篇章分析与指代消解系统

这是一个基于 `Streamlit` 的课程实验项目，用来演示三个核心篇章处理任务：

- 话语分割（EDU segmentation）
- 浅层篇章关系提取（PDTB-style relations）
- 指代消解（Coreference Resolution）

应用同时整合了：

- `spaCy`：分句、基础语言处理
- `requests`：获取网络文本样本
- `fastcoref`：神经指代消解

## 1. 推荐环境

根据题目中的调试经验，建议统一使用 `Python 3.10`，避免 `Python 3.13` 与 `SciPy`、`PyTorch` 的兼容问题。

### Conda 环境创建

```powershell
conda create -n discourse_lab python=3.10 -y
conda activate discourse_lab
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## 2. 运行方式

```powershell
streamlit run app.py
```

## 3. 常见问题与处理

### 依赖冲突

统一使用下面这种方式安装：

```powershell
python -m pip install 包名
```

不要混用 `conda install` 和 `pip install` 到不同解释器路径。

### PyTorch `_C` 模块加载失败

可先卸载再安装稳定版：

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Streamlit 报 `set_page_config()` 位置错误

项目已经把 `st.set_page_config()` 放在脚本最前面的首个 Streamlit 调用位置。

## 4. 功能说明

### 模块一：话语分割

- 页面会用 `requests` 拉取 `NeuralEDUSeg` 仓库 `data/rst/` 下的公开样本
- 支持解析多种可能的公开样本格式：`token-label`、`edu-per-line`、内联边界标记
- 使用 `spaCy` 构建规则基线，依据标点、从属连词和部分依存关系猜测 EDU 边界
- 使用左右两栏对比“规则基线切分结果”和“NeuralEDUSeg 真实标注结果”
- 在单词级别高亮边界词，并给出边界 Precision / Recall / F1 作为课堂展示指标

### 模块二：浅层篇章关系

- 提供单句输入框，默认使用课件中的 `although` 示例句
- 扫描常见显式连接词，并映射到 `Temporal / Contingency / Comparison / Expansion`
- 在页面中高亮连接词并展示其粗粒度语义类别
- 以连接词为界做简化版 `Arg1 / Arg2` 切分，并使用不同背景框对比展示
- 提供 `since` 的时间义 / 因果义观察入口，用于体验显式连接词消歧问题

### 模块三：指代消解

- 提供独立的多行英文段落输入框，方便测试跨句回指
- 优先使用 `fastcoref` 进行端到端指代消解，并提取 coreference clusters
- 将同一实体的不同 mentions 用同一种背景色直接高亮回渲到原始段落中
- 在高亮文本下方输出结构化的簇列表，例如 `Entity 1: [mention1, mention2, ...]`
- 如果 `fastcoref` 环境未成功安装，则自动回退到启发式结果，保证页面仍可展示

## 5. 数据来源设计

- 本地课程样例：位于 `data/sample_discourse.json`
- 网络样例：运行时通过 `requests` 抓取公开 JSON 文本，失败时自动回退到本地

## 6. 展示建议

课堂展示时，可以按下面顺序演示：

1. 选择一段包含连接词和代词的英文文本
2. 观察 EDU 如何被切分
3. 对比连接词如何触发不同篇章关系
4. 查看同一实体的不同 mention 如何被聚类

## 7. 后续可扩展方向

- 接入更真实的 PDTB / RST 标注数据
- 增加中文篇章分析支持
- 为关系图谱增加可视化连线
- 接入更强的 Transformer 篇章模型
