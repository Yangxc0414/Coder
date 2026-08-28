import ast
import os
from collections import defaultdict

deps = {}
deferred = []
for root, dirs, files in os.walk("coder_agent"):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if not f.endswith(".py"):
            continue
        path = os.path.join(root, f).replace(os.sep, "/")
        mod = path[:-3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[:-9]
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("coder_agent"):
                imports.add(node.module)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("coder_agent"):
                        imports.add(a.name)
        deps[mod] = imports
        # deferred imports inside functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(node):
                    hit = None
                    if isinstance(sub, ast.ImportFrom) and sub.module and sub.module.startswith("coder_agent"):
                        hit = sub.module
                    elif isinstance(sub, ast.Import):
                        for a in sub.names:
                            if a.name.startswith("coder_agent"):
                                hit = a.name
                    if hit:
                        deferred.append(f"{path}:{node.name}() -> {hit}")
                        break

counted = defaultdict(int)
for m, imps in deps.items():
    for i in imps:
        counted[i] += 1

print("=== 被依赖最多的模块（架构核心度 TOP10）===")
for mod, n in sorted(counted.items(), key=lambda x: -x[1])[:10]:
    print(f"  {n}x  {mod}")

print()
print(f"=== 函数内延迟导入（共 {len(deferred)} 处，循环依赖打破点）===")
for d in deferred:
    print(f"  {d}")
