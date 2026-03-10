#!/usr/bin/env bash
# Install VectorDBBench and register the DeepData client plugin.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VDBB_DIR="/tmp/VectorDBBench"

echo "=== VectorDBBench + DeepData Integration Setup ==="

# 1. Clone or update VectorDBBench
if [ -d "$VDBB_DIR" ]; then
    echo "VectorDBBench already cloned at $VDBB_DIR"
    cd "$VDBB_DIR" && git pull --ff-only 2>/dev/null || true
else
    echo "Cloning VectorDBBench..."
    git clone https://github.com/zilliztech/VectorDBBench.git "$VDBB_DIR"
fi

# 2. Install VectorDBBench
echo "Installing VectorDBBench..."
cd "$VDBB_DIR"
pip install -e ".[chroma,qdrant]" 2>&1 | tail -5

# 3. Symlink DeepData client into VectorDBBench
CLIENTS_DIR="$VDBB_DIR/vectordb_bench/backend/clients"
DEEPDATA_LINK="$CLIENTS_DIR/deepdata"

if [ -L "$DEEPDATA_LINK" ] || [ -d "$DEEPDATA_LINK" ]; then
    rm -rf "$DEEPDATA_LINK"
fi
ln -sf "$SCRIPT_DIR/deepdata" "$DEEPDATA_LINK"
echo "Symlinked DeepData client: $DEEPDATA_LINK -> $SCRIPT_DIR/deepdata"

# 4. Register DeepData in the DB enum (patch __init__.py)
INIT_FILE="$CLIENTS_DIR/__init__.py"
if ! grep -q "DeepData" "$INIT_FILE" 2>/dev/null; then
    echo "Registering DeepData in DB enum..."
    python3 << 'PYEOF'
import re

init_path = "/tmp/VectorDBBench/vectordb_bench/backend/clients/__init__.py"
with open(init_path, "r") as f:
    content = f.read()

# Add DeepData to the DB enum
# Find the last enum entry and add after it
enum_pattern = r'(class DB\(Enum\):.*?)(    @property)'
match = re.search(enum_pattern, content, re.DOTALL)
if match and "DeepData" not in content:
    enum_body = match.group(1)
    # Find last enum assignment line
    lines = enum_body.split('\n')
    insert_idx = -1
    for i, line in enumerate(lines):
        if '=' in line and '"' in line and 'class' not in line and '@' not in line and 'def' not in line:
            insert_idx = i
    if insert_idx >= 0:
        lines.insert(insert_idx + 1, '    DeepData = "DeepData"')
        new_enum_body = '\n'.join(lines)
        content = content.replace(enum_body, new_enum_body)

    # Add init_cls branch
    init_cls_pattern = r'(def init_cls\(self\).*?)(        raise)'
    match2 = re.search(init_cls_pattern, content, re.DOTALL)
    if match2:
        inject = '''        if self == DB.DeepData:
            from .deepdata.deepdata import DeepData
            return DeepData
'''
        content = content[:match2.start(2)] + inject + content[match2.start(2):]

    # Add config_cls branch
    config_cls_pattern = r'(def config_cls\(self\).*?)(        raise)'
    match3 = re.search(config_cls_pattern, content, re.DOTALL)
    if match3:
        inject = '''        if self == DB.DeepData:
            from .deepdata.config import DeepDataConfig
            return DeepDataConfig
'''
        content = content[:match3.start(2)] + inject + content[match3.start(2):]

    # Add case_config_cls branch
    case_config_pattern = r'(def case_config_cls\(.*?\).*?)(        raise)'
    match4 = re.search(case_config_pattern, content, re.DOTALL)
    if match4:
        inject = '''        if self == DB.DeepData:
            from .deepdata.config import DeepDataIndexConfig
            return DeepDataIndexConfig
'''
        content = content[:match4.start(2)] + inject + content[match4.start(2):]

    with open(init_path, "w") as f:
        f.write(content)
    print("  DB enum patched successfully")
else:
    print("  DeepData already registered or enum pattern not found")

PYEOF
else
    echo "DeepData already registered in DB enum"
fi

# 5. Register CLI command
CLI_FILE="$VDBB_DIR/vectordb_bench/cli/vectordbbench.py"
if ! grep -q "deepdata" "$CLI_FILE" 2>/dev/null; then
    echo "Registering DeepData CLI command..."
    python3 << 'PYEOF'
cli_path = "/tmp/VectorDBBench/vectordb_bench/cli/vectordbbench.py"
with open(cli_path, "r") as f:
    content = f.read()

if "deepdata" not in content.lower():
    # Add import and registration at the end of imports
    import_line = "from vectordb_bench.backend.clients.deepdata.cli import DeepData as DeepDataCmd\n"
    register_line = "cli.add_command(DeepDataCmd)\n"

    # Find last import line
    lines = content.split('\n')
    last_import = 0
    for i, line in enumerate(lines):
        if line.startswith('from ') or line.startswith('import '):
            last_import = i

    lines.insert(last_import + 1, import_line)

    # Find where commands are added
    last_add = 0
    for i, line in enumerate(lines):
        if 'add_command' in line:
            last_add = i

    if last_add > 0:
        lines.insert(last_add + 1, register_line)
    else:
        lines.append(register_line)

    with open(cli_path, "w") as f:
        f.write('\n'.join(lines))
    print("  CLI command registered")

PYEOF
else
    echo "DeepData CLI already registered"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  # Run standard SIFT benchmark (128d, 100K vectors):"
echo "  vectordbbench DeepData --url http://localhost:8080"
echo ""
echo "  # Run with custom HNSW params:"
echo "  vectordbbench DeepData --url http://localhost:8080 --m 32 --ef-construction 400 --ef-search 256"
echo ""
echo "  # Compare against Qdrant:"
echo "  vectordbbench QdrantCloud --url http://localhost:6333"
echo ""
