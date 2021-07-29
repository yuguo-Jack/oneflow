# python3 -m pip install astpretty pandas
# requires python3.9 to run
import ast
import os
import argparse
import ast
import subprocess
import multiprocessing
from pathlib import Path
import astpretty
import sys
from astpretty import pprint
from collections import Counter
import pandas as pd
from typed_ast.ast27 import Name, alias

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out_dir", type=str, default="python",
)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--ast", action="store_true")
ONEFLOW_TEST_PYTORCH_VISION_DIR = os.getenv("ONEFLOW_TEST_PYTORCH_VISION_DIR")
parser.add_argument(
    "--pytorch_vision_dir", type=str, default=ONEFLOW_TEST_PYTORCH_VISION_DIR,
)
args = parser.parse_args()

ONEFLOW_TEST_PYTORCH_VISION_PATH = Path(args.pytorch_vision_dir)
SHOULD_SAVE_ASTSHOULD_SAVE_AST = args.ast


class CompatibilityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.module_num = Counter()
        self.attribute_num = Counter()
        self.current_module_in_call = []
        self.ids_tracked = set()
        self.id2full_path = {}

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            if node.module == "torch" or "torch." in node.module:
                for a in node.names:
                    assert isinstance(a, ast.alias)
                    asname = a.asname or a.name
                    self.ids_tracked.add(asname)
                    self.id2full_path[asname] = ".".join([node.module, a.name])
                    # pprint(a)
                self.module_num.update([node.module])

    def visit_Import(self, node: ast.Import):
        modules = [
            a.name
            for a in node.names
            if a.name.startswith("torch.") and a.name != "torch"
        ]
        self.module_num.update(modules)

    def visit_Name(self, node: Name) -> bool:
        self.current_module_in_call.insert(0, node.id)

    def visit_Attribute(self, node: ast.Attribute) -> bool:
        self.current_module_in_call.insert(0, node.attr)

    def record_attr(self, node: ast.Call):
        if self.current_module_in_call:
            attr_full = ".".join(
                [self.id2full_path[self.current_module_in_call[0]],]
                + self.current_module_in_call[1::]
                + [node.func.attr]
            )
        else:
            attr_full = self.id2full_path[node.func.id]
        self.attribute_num.update([attr_full])

    def visit_Call(self, node: ast.Call):
        func = node.func
        self.current_module_in_call = []
        if isinstance(func, ast.Attribute):
            self.visit(func.value)
            if self.current_module_in_call:
                if self.current_module_in_call[0] in self.ids_tracked:
                    self.record_attr(node)
        elif isinstance(func, ast.Name):
            if func.id in self.ids_tracked:
                self.record_attr(node)


def analyze_py(args):
    src: Path = args["src"]
    tree = ast.parse(src.read_text())
    v = CompatibilityVisitor()
    v.visit(tree)
    if SHOULD_SAVE_ASTSHOULD_SAVE_AST:
        ast_path = src.with_suffix(".ast.py")
        if not ast_path.exists():
            ast_path.write_text(
                f"""from ast import *
{astpretty.pformat(tree)}
"""
            )
    return v


if __name__ == "__main__":
    print(ONEFLOW_TEST_PYTORCH_VISION_PATH)
    py_srcs = ONEFLOW_TEST_PYTORCH_VISION_PATH.glob("**/*.py")
    pool = multiprocessing.Pool()
    results = pool.map(
        analyze_py,
        [{"src": src,} for src in py_srcs if not src.name.endswith(".ast.py")],
    )
    pool.close()
    module_num = Counter()
    attribute_num = Counter()
    for r in results:
        module_num.update(r.module_num)
        attribute_num.update(r.attribute_num)
    print(pd.DataFrame(module_num.most_common()).to_markdown())
    print(pd.DataFrame(attribute_num.most_common()).to_markdown())
