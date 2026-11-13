# Зона 1. Усі експерименти по черзі. Запуск з кореня проєкту:
#   python experiments/run_all.py
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
# run_ga.py іде одразу після класифікатора: він обирає параметри і архітектуру
# перцептрона, з якими потім рахуються всі таблиці
order = ["run_classifier.py", "run_ga.py", "run_dp_vs_greedy.py", "run_copies.py", "run_scenarios.py",
         "run_compress_order.py", "run_attack.py", "run_ablation.py", "run_final.py",
         "run_checkpoints.py", "run_adaptive_huffman.py", "run_prediction.py", "run_c1.py"]
for name in order:
    print("=== " + name)
    subprocess.run([sys.executable, os.path.join(here, name)], check=True)
