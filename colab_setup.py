"""
Google Colab Setup — TrustFed-IoT Benchmark Experiments
=========================================================

How to use:
  1. Open Google Colab (https://colab.research.google.com)
  2. Runtime → Change runtime type → T4 GPU
  3. Copy-paste each cell below into a Colab cell and run it

OR upload this file and run:
  !python colab_setup.py
"""

# ============================================================
# CELL 1: Install dependencies
# ============================================================
# !pip install torch torchvision optuna requests matplotlib numpy

# ============================================================
# CELL 2: Clone project from GitHub
# ============================================================
# !git clone https://github.com/YOUR_USERNAME/trustfed-iot.git
# %cd trustfed-iot

# --- OR upload manually ---
# from google.colab import drive
# drive.mount('/content/drive')
# !cp -r /content/drive/MyDrive/trustfed-iot /content/
# %cd /content/trustfed-iot

# ============================================================
# CELL 3: Rebuild partition cache (for validation split)
# ============================================================
# !python -m data.build_partition_cache

# ============================================================
# CELL 4: Run full benchmark — 100 rounds, 5 seeds, all attacks
#          This takes ~6 hours on T4 GPU.
# ============================================================
# !python -m experiments.run_all_experiments \
#   --rounds 100 \
#   --seeds 1 2 3 4 5 \
#   --attacks clean gaussian sign_flip scaling label_flip \
#   --methods proposed fedavg multikrum \
#   --export-zip

# ============================================================
# CELL 5: Generate plots and summary tables
# ============================================================
# !python -m experiments.plot_experiments --root results/benchmark/

# ============================================================
# CELL 6: Download results
# ============================================================
# !zip -r benchmark_results.zip results/benchmark/ results/exports/
# from google.colab.files import download
# download('benchmark_results.zip')
