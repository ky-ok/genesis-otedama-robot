import sys

# stable-baselines3 -> torch.utils.tensorboard -> tensorboard が TensorFlow を import する。
# TensorFlowが内蔵するLLVMとGenesis(quadrants)のLLVMが1プロセスで衝突すると
# "LLVM ERROR: inconsistency in registered CommandLine options" で異常終了するため、
# 本パッケージを使うプロセスではTensorFlowを読み込ませない(tensorboardは自前のスタブにフォールバックする)。
# 必ず stable_baselines3 / genesis より先に import すること。
sys.modules.setdefault("tensorflow", None)
