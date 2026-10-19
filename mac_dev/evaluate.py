"""
学習済み方策を評価する(中身は genesis_juggling/evaluation.py)。

    python mac_dev/evaluate.py --model checkpoints/stage2_funnel_idle_130M.zip --n-balls 2
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

print("起動中です(初回はファイルの読み込みで最大1分ほどかかります。止まっていません)...", flush=True)

import genesis_juggling  # noqa: F401,E402  (最初にimportする: TensorFlow抑止)
from genesis_juggling.evaluation import main  # noqa: E402

if __name__ == "__main__":
    main()
