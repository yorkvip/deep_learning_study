import io

path = r"D:\python\深度学习！\19 丢弃法.ipynb"
with io.open(path, "r", encoding="utf-8") as f:
    text = f.read()

bug = "torch.tensor(X.shape).uniform_(0,1)"
fixed = "torch.rand(X.shape)"

cnt = text.count(bug)
print("找到 bug 写法次数:", cnt)
if cnt == 0:
    raise SystemExit("未找到目标字符串，可能已修复或写法不同")

text = text.replace(bug, fixed)

with io.open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("已替换为:", fixed)
