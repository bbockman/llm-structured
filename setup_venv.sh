#!/usr/bin/env bash
set -e

echo "=== Step 1: Installing prerequisite build tools ==="
sudo apt update
sudo apt install -y \
    build-essential \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libffi-dev \
    libncurses5-dev \
    libncursesw5-dev \
    tk-dev

export CFLAGS="-I/usr/include -I/usr/include/x86_64-linux-gnu"
export LDFLAGS="-L/usr/lib -L/usr/lib/x86_64-linux-gnu"

export FLASH_ATTENTION_FORCE_SM=100
export FLASH_ATTENTION_DISABLE_FP8=1
# export FLASH_ATTENTION_DISABLE_FP32=1
export FLASH_ATTENTION_DISABLE_INT8=1
export FLASH_ATTENTION_DISABLE_GQA=1
export FLASH_ATTENTION_DISABLE_MQA=1
export FLASH_ATTENTION_DISABLE_SPLIT_KV=1
export FLASH_ATTENTION_DISABLE_DIST=1
# export FLASH_ATTENTION_DISABLE_TRITON=1
# export FLASH_ATTENTION_BUILD_PREFILL=1
# export FLASH_ATTENTION_BUILD_DECODE=0

echo "=== Step 2: Installing pyenv (no bashrc modifications) ==="
if [ ! -d "$HOME/.pyenv" ]; then
    curl https://pyenv.run | bash
fi

# Ensure your custom pyenv init script exists
if [ ! -f "$HOME/pyenv-init.sh" ]; then
cat << 'EOF' > "$HOME/pyenv-init.sh"
# Custom pyenv activation script
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
EOF
fi

echo "=== Step 3: Activating pyenv for this script only ==="
source "$HOME/pyenv-init.sh"

echo "=== Step 4: Installing Python 3.11.9 via pyenv ==="
pyenv install -s 3.11.9

echo "=== Step 5: Creating pyenv virtualenv named 'fused' ==="
pyenv virtualenv 3.11.9 fused || true

echo "=== Step 6: Creating your preferred venv symlink at ~/venvs/fused ==="
TARGET="$HOME/.pyenv/versions/fused"
LINK="$HOME/venvs/fused"

# Create directory if needed
mkdir -p "$HOME/venvs"

# Only update symlink if missing OR pointing to the wrong place
if [ ! -L "$LINK" ] || [ "$(readlink "$LINK")" != "$TARGET" ]; then
    ln -sfn "$TARGET" "$LINK"
    echo "Updated symlink: $LINK → $TARGET"
else
    echo "Symlink already correct, no write needed."
fi

echo "=== Step 7: Activating pyenv environment 'fused' ==="
pyenv activate fused

echo "=== Step 8: Upgrading pip + wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== Step 9: Installing PyTorch (CUDA 13.0) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

pip install psutil


echo "=== Step 10: Installing FlashAttention2, Triton, xFormers, TransformerEngine ==="
pip install flash-attn --no-build-isolation --no-cache-dir
pip install triton
pip install xformers

echo "=== Step 11: Installing APEX (optional fused LN/Adam kernels) ==="
pip install ninja
pip install git+https://github.com/NVIDIA/apex.git --no-build-isolation

echo "=== Step 12: Verifying CUDA, FA2, Triton, TE ==="
python - << 'EOF'
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Torch Version:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

try:
    import flash_attn
    print("FlashAttention2 OK")
except:
    print("FlashAttention2 FAILED")

try:
    import xformers
    print("xFormers OK")
except:
    print("xFormers FAILED")

try:
    import transformer_engine
    print("TransformerEngine OK")
except:
    print("TransformerEngine FAILED")

EOF

echo "=== Environment 'fused' successfully created ==="

