import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# 1. 文件路径
# =========================
csv_path = "imu_processed.csv"
out_dir = "imu_plots"

os.makedirs(out_dir, exist_ok=True)

# =========================
# 2. 读取 CSV
# =========================
df = pd.read_csv(csv_path)

print("CSV columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# =========================
# 3. 检查必要列
# =========================
required_cols = ["t", "gx", "gy", "gz", "ax", "ay", "az"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"CSV 中缺少列: {col}")

# =========================
# 4. 时间归零
# =========================
df["t_rel"] = df["t"] - df["t"].iloc[0]

# =========================
# 5. 画角速度曲线
# =========================
plt.figure(figsize=(12, 6))

plt.plot(df["t_rel"], df["gx"], label="gyro_x")
plt.plot(df["t_rel"], df["gy"], label="gyro_y")
plt.plot(df["t_rel"], df["gz"], label="gyro_z")

plt.xlabel("Time / s")
plt.ylabel("Angular Velocity / rad/s")
plt.title("IMU Angular Velocity")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "gyro_xyz.png"), dpi=300)
plt.show()

# =========================
# 6. 画线加速度曲线
# =========================
plt.figure(figsize=(12, 6))

plt.plot(df["t_rel"], df["ax"], label="acc_x")
plt.plot(df["t_rel"], df["ay"], label="acc_y")
plt.plot(df["t_rel"], df["az"], label="acc_z")

plt.xlabel("Time / s")
plt.ylabel("Linear Acceleration / m/s^2")
plt.title("IMU Linear Acceleration")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "acc_xyz.png"), dpi=300)
plt.show()

# =========================
# 7. 画模长曲线
# =========================
plt.figure(figsize=(12, 6))

if "gyro_norm" in df.columns:
    plt.plot(df["t_rel"], df["gyro_norm"], label="gyro_norm")

if "acc_norm" in df.columns:
    plt.plot(df["t_rel"], df["acc_norm"], label="acc_norm")

plt.xlabel("Time / s")
plt.ylabel("Norm")
plt.title("IMU Norm")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "imu_norm.png"), dpi=300)
plt.show()

print("\nSaved plots to:", out_dir)