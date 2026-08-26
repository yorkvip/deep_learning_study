import json

fn = r'd:\python\深度学习！\19 丢弃法.ipynb'
nb = json.load(open(fn, encoding='utf-8'))

note = '\n'.join([
"## 怎么理解 `class Net(nn.Module)`",
"",
"带两个隐藏层（各 256 单元）、自带丢弃法的多层感知机。",
"",
"**① 继承基类**：`nn.Module` 是 PyTorch 所有网络的基类，继承它就能自动管理参数、自动求导、能被 `.to('cuda')` 搬运。",
"",
"**② `__init__`（搭积木）**：定义网络包含哪些层，创建对象时自动执行：",
"",
"```python",
"def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):",
"    super().__init__()",
"    self.lin1 = nn.Linear(num_inputs, num_hiddens1)    # 784 → 256",
"    self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)  # 256 → 256",
"    self.lin3 = nn.Linear(num_hiddens2, num_outputs)   # 256 → 10",
"    self.relu = nn.ReLU()",
"```",
"",
"**③ `forward`（流水线）**：定义数据怎么流动，每次 `net(X)` 都执行：",
"",
"```python",
"def forward(self, X):",
"    H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))",
"    if self.training == True:",
"        H1 = dropout_layer(H1, dropout1)   # 第一隐藏层丢弃 0.2",
"    H2 = self.relu(self.lin2(H1))",
"    if self.training == True:",
"        H2 = dropout_layer(H2, dropout2)   # 第二隐藏层丢弃 0.5",
"    out = self.lin3(H2)",
"    return out",
"```",
"",
"路径：展平(784) → lin1 → relu → [训练时丢弃0.2] → lin2 → relu → [训练时丢弃0.5] → lin3 → 输出 10 个数",
"",
"丢弃法只在 `self.training == True`（训练时）生效；测试时所有神经元都参与，输出才稳定。",
])

md_cell = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [line + '\n' for line in note.split('\n')],
}

nb['cells'][0] = md_cell
json.dump(nb, open(fn, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('cell 0 已精简为只讲 Net 类，总 cell 数:', len(nb['cells']))