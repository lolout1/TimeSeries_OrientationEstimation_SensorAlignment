#!/usr/bin/env python3
import os, sys, argparse, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from fastdtw import fastdtw
import scipy.spatial.distance as dist
from sklearn.metrics import mean_squared_error

from utils.imu_fusion import MadgwickFilter, KalmanFilter, ExtendedKalmanFilter
from utils.processor.base import csvloader

def lowpass_filter(x, cutoff=5.0, fs=30.0, order=2):
    if x is None or len(x)<3: return x
    b, a = butter(order, cutoff/(0.5*fs), btype='low')
    y = np.zeros_like(x)
    for i in range(x.shape[1]):
        try: y[:, i] = filtfilt(b,a,x[:, i])
        except: y[:, i] = x[:, i]
    return y

def compute_3d_alignment_metrics(signal1, signal2):
    if signal1 is None or signal2 is None or len(signal1)<3 or len(signal2)<3: return None, None, None, None, None
    s1_mean, s2_mean = np.mean(signal1, axis=0), np.mean(signal2, axis=0)
    s1_std, s2_std = np.std(signal1, axis=0), np.std(signal2, axis=0)
    s1_std[s1_std < 1e-8], s2_std[s2_std < 1e-8] = 1.0, 1.0
    s1_norm, s2_norm = (signal1 - s1_mean) / s1_std, (signal2 - s2_mean) / s2_std
    distance, path = fastdtw(s1_norm, s2_norm, dist=dist.euclidean)
    path = np.array(path)
    idx1, idx2 = path[:, 0], path[:, 1]
    s1_aligned, s2_aligned = signal1[idx1], signal2[idx2]
    min_len = min(len(signal1), len(signal2))
    rmse_per_axis = np.array([np.sqrt(mean_squared_error(signal1[:min_len, i], signal2[:min_len, i])) for i in range(3)])
    rmse = np.sqrt(mean_squared_error(signal1[:min_len].flatten(), signal2[:min_len].flatten()))
    corr_per_axis = np.array([np.corrcoef(signal1[:min_len, i], signal2[:min_len, i])[0, 1] for i in range(3)])
    corr = np.mean(corr_per_axis)
    aligned_rmse_per_axis = np.array([np.sqrt(mean_squared_error(s1_aligned[:, i], s2_aligned[:, i])) for i in range(3)])
    aligned_rmse = np.sqrt(mean_squared_error(s1_aligned.flatten(), s2_aligned.flatten()))
    metrics = {
        'rmse': rmse, 'rmse_per_axis': rmse_per_axis, 'correlation': corr,
        'correlation_per_axis': corr_per_axis, 'dtw_distance': distance,
        'aligned_rmse': aligned_rmse, 'aligned_rmse_per_axis': aligned_rmse_per_axis
    }
    return path, metrics, s1_aligned, s2_aligned, distance

def load_skel_data(skel_file, fs=30.0, assume_positions=True, wrist_joint_idx=9):
    if not os.path.exists(skel_file): return None, None, None
    df = csvloader(skel_file)
    if df is None or df.shape[0]<3: return None, None, None
    has_time = False
    try: 
        float(df[0, 0])
        has_time = True
    except (ValueError, TypeError): 
        has_time = False
    
    if has_time:
        t = df[:, 0]
        joint_start_idx = 1 + wrist_joint_idx * 3
        if df.shape[1] >= joint_start_idx + 3:
            raw = df[:, joint_start_idx:joint_start_idx+3]
        else:
            print(f"Skeleton data insufficient columns for joint {wrist_joint_idx}: need {joint_start_idx+3}, got {df.shape[1]}")
            return None, None, None
    else:
        t = np.arange(len(df), dtype=float)
        joint_start_idx = wrist_joint_idx * 3
        if df.shape[1] >= joint_start_idx + 3:
            raw = df[:, joint_start_idx:joint_start_idx+3]
        else:
            print(f"Skeleton data insufficient columns: need {joint_start_idx+3}, got {df.shape[1]}")
            return None, None, None

    t = t - t[0]
    if t[-1]>1e9: t = t/1e9
    
    start, end = t[0], t[-1]
    if end<=start: return None, None, None
    newt = np.arange(start, end, 1.0/fs)
    if len(newt)<3: return None, None, None

    arr = np.zeros((len(newt), 3))
    for i in range(3):
        try:
            f = interp1d(t, raw[:, i], bounds_error=False, fill_value=(raw[0, i], raw[-1, i]))
            arr[:, i] = f(newt)
        except Exception as e:
            print(f"Interpolation error on axis {i}: {e}")
            idx = np.argmin(np.abs(t[:,None]-newt), axis=0)
            arr[:, i] = raw[idx, i]

    arr = lowpass_filter(arr, cutoff=10.0, fs=fs)
    skl_positions = arr.copy()
    
    if assume_positions:
        vel = np.gradient(arr, axis=0)*fs
        acc = np.gradient(vel, axis=0)*fs
        acc = acc - np.mean(acc, axis=0)
        return acc, newt, skl_positions, len(t)
    else:
        return arr, newt, None, len(t)

def load_watch_data(acc_file, gyro_file, fs=30.0, filter_type='none', is_linear_acc=True):
    if not os.path.exists(acc_file) or not os.path.exists(gyro_file): return None, None, None, None, None, 0, 0
    df_a = csvloader(acc_file)
    df_g = csvloader(gyro_file)
    if df_a is None or df_g is None or df_a.shape[0]<2 or df_g.shape[0]<2 or df_a.shape[1]<3 or df_g.shape[1]<3: 
        return None, None, None, None, None, 0, 0

    acc_samples = df_a.shape[0]
    gyro_samples = df_g.shape[0]

    if df_a.shape[1]>=4:
        at, a = df_a[:, 0], df_a[:, 1:4]
    else:
        at, a = np.arange(len(df_a), dtype=float), df_a[:, 0:3]
    
    if df_g.shape[1]>=4:
        gt, g = df_g[:, 0], df_g[:, 1:4]
    else:
        gt, g = np.arange(len(df_g), dtype=float), df_g[:, 0:3]
    
    at = at - at[0]
    gt = gt - gt[0]
    if at[-1]>1e9: at = at/1e9
    if gt[-1]>1e9: gt = gt/1e9

    start, end = max(at[0], gt[0]), min(at[-1], gt[-1])
    if end<=start: return None, None, None, None, None, acc_samples, gyro_samples
    
    newt = np.arange(start, end, 1.0/fs)
    if len(newt)<3: return None, None, None, None, None, acc_samples, gyro_samples

    A = np.zeros((len(newt), 3))
    G = np.zeros((len(newt), 3))
    for i in range(3):
        try:
            f = interp1d(at, a[:, i], bounds_error=False, fill_value=(a[0, i], a[-1, i]))
            A[:, i] = f(newt)
        except:
            idxA = np.argmin(np.abs(at[:,None]-newt), axis=0)
            A[:, i] = a[idxA, i]

        try:
            f = interp1d(gt, g[:, i], bounds_error=False, fill_value=(g[0, i], g[-1, i]))
            G[:, i] = f(newt)
        except:
            idxG = np.argmin(np.abs(gt[:,None]-newt), axis=0)
            G[:, i] = g[idxG, i]

    A_raw, G_raw = A.copy(), G.copy()
    A = lowpass_filter(A.copy(), 10.0, fs)
    G = lowpass_filter(G.copy(), 10.0, fs)

    acc_before = A.copy()
    acc_after, quats = None, None
    
    if filter_type.lower() in ['madgwick', 'kalman', 'ekf']:
        if filter_type.lower() == 'madgwick':
            f = MadgwickFilter()
        elif filter_type.lower() == 'kalman':
            f = KalmanFilter()
        else:
            f = ExtendedKalmanFilter()
        
        quats = np.zeros((len(A), 4))
        
        if is_linear_acc:
            A_with_gravity = A.copy()
            A_with_gravity[:, 2] += 9.81 
            for i in range(len(A)):
                quats[i] = f.update(A_with_gravity[i], G[i])
        else:
            for i in range(len(A)):
                quats[i] = f.update(A[i], G[i])
        
        acc_after = np.zeros_like(A)
        for i in range(len(A)):
            q = quats[i]
            r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            acc_after[i] = r.apply(A[i])
        
        if not is_linear_acc:
            gravity_vector = np.array([0, 0, 9.81])
            for i in range(len(acc_after)):
                acc_after[i] = acc_after[i] - gravity_vector
    
    return acc_before, acc_after, G, quats, newt, acc_samples, gyro_samples, A_raw, G_raw

def plot_raw_vs_processed(raw_signal, processed_signal, time, out_dir, title_prefix, axes_names=['X', 'Y', 'Z']):
    if raw_signal is None or processed_signal is None: return
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i in range(3):
        axes[i].plot(time, raw_signal[:, i], label=f'Raw {axes_names[i]}', color='blue', alpha=0.7)
        axes[i].plot(time, processed_signal[:, i], label=f'Processed {axes_names[i]}', color='red')
        axes[i].set_ylabel(f'{axes_names[i]}-Axis')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    axes[2].set_xlabel('Time (s)')
    plt.suptitle(f'{title_prefix} - Raw vs Processed')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{title_prefix.lower().replace(" ", "_")}_raw_vs_processed.png'), dpi=300)
    plt.close()

def plot_orientation_3d(quats, timestamps, out_dir):
    if quats is None or len(quats) < 3: return
    from mpl_toolkits.mplot3d import Axes3D
    n_samples = min(20, len(quats))
    sample_indices = np.linspace(0, len(quats)-1, n_samples, dtype=int)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, idx in enumerate(sample_indices):
        q = quats[idx]
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        axes = np.eye(3) * 0.5
        rotated_axes = r.apply(axes)
        origin = np.array([timestamps[idx], 0, 0])
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z'] if i == 0 else [None, None, None]
        for j in range(3):
            ax.quiver(origin[0], origin[1], origin[2],
                     rotated_axes[j, 0], rotated_axes[j, 1], rotated_axes[j, 2],
                     color=colors[j], label=labels[j])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Device Orientation Over Time')
    ax.legend()
    plt.savefig(os.path.join(out_dir, 'orientation_3d.png'), dpi=300)
    plt.close()
    
    euler_angles = np.zeros((len(quats), 3))
    for i in range(len(quats)):
        q = quats[i]
        r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        euler_angles[i] = r.as_euler('zyx', degrees=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timestamps, euler_angles[:, 0], 'r-', label='Yaw (Z)')
    ax.plot(timestamps, euler_angles[:, 1], 'g-', label='Pitch (Y)')
    ax.plot(timestamps, euler_angles[:, 2], 'b-', label='Roll (X)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Orientation as Euler Angles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'euler_angles.png'), dpi=300)
    plt.close()

def plot_3d_dtw_alignment(aligned_signal1, aligned_signal2, out_dir, title="3D DTW Alignment"):
    if aligned_signal1 is None or aligned_signal2 is None: return
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    n_samples = min(20, len(aligned_signal1))
    sample_indices = np.linspace(0, len(aligned_signal1)-1, n_samples, dtype=int)
    ax.plot(aligned_signal1[:, 0], aligned_signal1[:, 1], aligned_signal1[:, 2], 'b-', linewidth=1, label='Signal 1', alpha=0.7)
    ax.plot(aligned_signal2[:, 0], aligned_signal2[:, 1], aligned_signal2[:, 2], 'r-', linewidth=1, label='Signal 2', alpha=0.7)
    for idx in sample_indices:
        ax.plot([aligned_signal1[idx, 0], aligned_signal2[idx, 0]],
               [aligned_signal1[idx, 1], aligned_signal2[idx, 1]],
               [aligned_signal1[idx, 2], aligned_signal2[idx, 2]],
               'k--', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('X Acceleration')
    ax.set_ylabel('Y Acceleration')
    ax.set_zlabel('Z Acceleration')
    ax.set_title(title)
    ax.legend()
    plt.savefig(os.path.join(out_dir, title.lower().replace(' ', '_') + '.png'), dpi=300)
    plt.close()

def plot_signals_before_after_alignment(signal1, signal2, aligned_signal1, aligned_signal2, out_dir, label1="Signal 1", label2="Signal 2"):
    if signal1 is None or signal2 is None or aligned_signal1 is None or aligned_signal2 is None: return
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes_names = ['X', 'Y', 'Z']
    for i in range(3):
        axes[i].plot(np.arange(len(signal1)), signal1[:, i], label=f'{label1} {axes_names[i]}', color='blue')
        axes[i].plot(np.arange(len(signal2)), signal2[:, i], label=f'{label2} {axes_names[i]}', color='red')
        axes[i].set_ylabel(f'{axes_names[i]}-Axis Acc. (m/s²)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    axes[2].set_xlabel('Sample Index')
    plt.suptitle(f'Before DTW Alignment: {label1} vs {label2}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'before_alignment_{label1.lower()}_{label2.lower()}.png'), dpi=300)
    plt.close()
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i in range(3):
        axes[i].plot(np.arange(len(aligned_signal1)), aligned_signal1[:, i], label=f'{label1} {axes_names[i]} (aligned)', color='blue')
        axes[i].plot(np.arange(len(aligned_signal2)), aligned_signal2[:, i], label=f'{label2} {axes_names[i]} (aligned)', color='red')
        axes[i].set_ylabel(f'{axes_names[i]}-Axis Acc. (m/s²)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    axes[2].set_xlabel('DTW Index')
    plt.suptitle(f'After DTW Alignment: {label1} vs {label2}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'after_alignment_{label1.lower()}_{label2.lower()}.png'), dpi=300)
    plt.close()
def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive plots comparing watch and skeleton data")
    parser.add_argument('--acc-file', required=True, help='Watch accelerometer CSV')
    parser.add_argument('--gyro-file', required=True, help='Watch gyroscope CSV')
    parser.add_argument('--skl-file', required=True, help='Skeleton CSV (positions or accel) for same trial')
    parser.add_argument('--fs', type=float, default=30.0, help='Resample frequency e.g. 30 Hz')
    parser.add_argument('--filter-type', default='none', help='madgwick|kalman|ekf|none')
    parser.add_argument('--out-dir', default='viz_output', help='Output directory for final plot')
    parser.add_argument('--assume-skl-positions', action='store_true', help='If set, skeleton data is positions => compute 2nd derivative as acceleration.')
    parser.add_argument('--wrist-joint-idx', type=int, default=9, help='Index of wrist joint in skeleton data (default: 9 for left wrist)')
    parser.add_argument('--is-linear-acc', action='store_true', help='If set, accelerometer data is assumed to be linear (gravity already removed)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # Check if ALL required files exist
    if not (os.path.exists(args.acc_file) and os.path.exists(args.gyro_file) and os.path.exists(args.skl_file)):
        print(f"Skipping - one or more required files not found.")
        return 1  # Return error code to indicate failure
    watch_before, watch_after, watch_gyro, watch_ori, watch_t, acc_samples, gyro_samples, acc_raw, gyro_raw = load_watch_data(
        args.acc_file, args.gyro_file, fs=args.fs, filter_type=args.filter_type, is_linear_acc=args.is_linear_acc
    )
    if watch_before is None or watch_t is None or watch_gyro is None:
        print(f"Skipping - failed to load watch data.")
        return 0

    skl_acc, skl_t, skl_pos, skl_samples = load_skel_data(
        args.skl_file, fs=args.fs, assume_positions=args.assume_skl_positions, wrist_joint_idx=args.wrist_joint_idx
    )
    
    if skl_acc is None or skl_t is None:
        print(f"Skipping - failed to load skeleton data.")
        return 0
    
    stats_file = os.path.join(args.out_dir, 'data_stats.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Data Statistics\n")
        f.write(f"==============\n\n")
        f.write(f"Accelerometer: {args.acc_file}\n")
        f.write(f"  - Original samples: {acc_samples}\n")
        f.write(f"Gyroscope: {args.gyro_file}\n")
        f.write(f"  - Original samples: {gyro_samples}\n")
        f.write(f"Skeleton: {args.skl_file}\n")
        f.write(f"  - Original samples: {skl_samples}\n")
        f.write(f"  - Wrist joint idx: {args.wrist_joint_idx}\n\n")
        f.write(f"Sampling frequency: {args.fs} Hz\n")
        f.write(f"Filter type: {args.filter_type}\n")
        f.write(f"Linear acceleration mode: {args.is_linear_acc}\n\n")
    
    # Plot raw vs processed data
    plot_raw_vs_processed(acc_raw, watch_before, watch_t, args.out_dir, "Accelerometer", axes_names=['X', 'Y', 'Z'])
    plot_raw_vs_processed(gyro_raw, watch_gyro, watch_t, args.out_dir, "Gyroscope", axes_names=['X', 'Y', 'Z'])
    
    # Find common time range
    minT, maxT = max(watch_t[0], skl_t[0]), min(watch_t[-1], skl_t[-1])
    
    # Check if there's enough overlap
    if maxT <= minT or maxT - minT < 1.0:
        print(f"Skipping - insufficient time overlap between watch and skeleton data.")
        return 0

    # Subset data to common time range
    valid_idx = np.where((watch_t >= minT) & (watch_t <= maxT))[0]
    watch_t_sub = watch_t[valid_idx]
    wb_sub = watch_before[valid_idx]
    wg_sub = watch_gyro[valid_idx]
    wa_sub = None if watch_after is None else watch_after[valid_idx]
    wo_sub = None if watch_ori is None else watch_ori[valid_idx]

    with open(stats_file, 'a') as f:
        f.write(f"Time Range Statistics\n")
        f.write(f"====================\n\n")
        f.write(f"Watch time range: {watch_t[0]:.3f} - {watch_t[-1]:.3f} seconds\n")
        f.write(f"Skeleton time range: {skl_t[0]:.3f} - {skl_t[-1]:.3f} seconds\n")
        f.write(f"Common time range: {minT:.3f} - {maxT:.3f} seconds\n")
        f.write(f"Watch samples after alignment: {len(watch_t_sub)}\n\n")

    # Process skeleton data
    skl_acc_sub = np.zeros((len(watch_t_sub), 3))
    for i in range(3):
        try:
            f = interp1d(skl_t, skl_acc[:, i], bounds_error=False, fill_value=(skl_acc[0, i], skl_acc[-1, i]))
            skl_acc_sub[:, i] = f(watch_t_sub)
        except:
            idxS = np.argmin(np.abs(skl_t[:,None]-watch_t_sub), axis=0)
            skl_acc_sub[:, i] = skl_acc[idxS, i]
    
    # Get skeleton positions if available
    skl_pos_sub = None
    if skl_pos is not None:
        skl_pos_sub = np.zeros((len(watch_t_sub), 3))
        for i in range(3):
            try:
                f = interp1d(skl_t, skl_pos[:, i], bounds_error=False, fill_value=(skl_pos[0, i], skl_pos[-1, i]))
                skl_pos_sub[:, i] = f(watch_t_sub)
            except:
                idxS = np.argmin(np.abs(skl_t[:,None]-watch_t_sub), axis=0)
                skl_pos_sub[:, i] = skl_pos[idxS, i]

    # Calculate magnitudes
    mag_wb = np.linalg.norm(wb_sub, axis=1)
    mag_wa = None if wa_sub is None else np.linalg.norm(wa_sub, axis=1)
    mag_skl = np.linalg.norm(skl_acc_sub, axis=1)

    # Compute alignment metrics
    alignment_stats = {}
    if wa_sub is not None and wb_sub is not None:
        path_wb_wa, metrics_wb_wa, aligned_wb, aligned_wa, dist_wb_wa = compute_3d_alignment_metrics(wb_sub, wa_sub)
        alignment_stats['watch_before_after'] = metrics_wb_wa
        plot_signals_before_after_alignment(wb_sub, wa_sub, aligned_wb, aligned_wa, args.out_dir, 
                                          "Watch Before", "Watch After")
        plot_3d_dtw_alignment(aligned_wb, aligned_wa, args.out_dir, 
                            f"3D DTW Alignment: Watch Before/After (dist={dist_wb_wa:.2f})")
    
    if wb_sub is not None and skl_acc_sub is not None:
        path_wb_skl, metrics_wb_skl, aligned_wb_skl, aligned_skl_wb, dist_wb_skl = compute_3d_alignment_metrics(wb_sub, skl_acc_sub)
        alignment_stats['watch_before_skeleton'] = metrics_wb_skl
        plot_signals_before_after_alignment(wb_sub, skl_acc_sub, aligned_wb_skl, aligned_skl_wb, args.out_dir, 
                                          "Watch Before", "Skeleton")
        plot_3d_dtw_alignment(aligned_wb_skl, aligned_skl_wb, args.out_dir, 
                            f"3D DTW Alignment: Watch Before/Skeleton (dist={dist_wb_skl:.2f})")
    
    if wa_sub is not None and skl_acc_sub is not None:
        path_wa_skl, metrics_wa_skl, aligned_wa_skl, aligned_skl_wa, dist_wa_skl = compute_3d_alignment_metrics(wa_sub, skl_acc_sub)
        alignment_stats['watch_after_skeleton'] = metrics_wa_skl
        plot_signals_before_after_alignment(wa_sub, skl_acc_sub, aligned_wa_skl, aligned_skl_wa, args.out_dir, 
                                          "Watch After", "Skeleton")
        plot_3d_dtw_alignment(aligned_wa_skl, aligned_skl_wa, args.out_dir, 
                            f"3D DTW Alignment: Watch After/Skeleton (dist={dist_wa_skl:.2f})")

    # Plot orientation data if available
    if wo_sub is not None:
        plot_orientation_3d(wo_sub, watch_t_sub, args.out_dir)

    # Save alignment statistics
    with open(os.path.join(args.out_dir, 'alignment_stats.txt'), 'w') as f:
        f.write(f"Alignment Statistics (Filter: {args.filter_type}, Wrist Joint: {args.wrist_joint_idx}, Linear Acc: {args.is_linear_acc})\n")
        f.write(f"=========================================\n\n")
        
        for comp_name, metrics in alignment_stats.items():
            f.write(f"{comp_name.replace('_', ' ').title()}:\n")
            f.write(f"  RMSE (3D): {metrics['rmse']:.4f}\n")
            f.write(f"  RMSE per axis (X,Y,Z): {metrics['rmse_per_axis'][0]:.4f}, {metrics['rmse_per_axis'][1]:.4f}, {metrics['rmse_per_axis'][2]:.4f}\n")
            f.write(f"  Correlation (3D avg): {metrics['correlation']:.4f}\n")
            f.write(f"  Correlation per axis (X,Y,Z): {metrics['correlation_per_axis'][0]:.4f}, {metrics['correlation_per_axis'][1]:.4f}, {metrics['correlation_per_axis'][2]:.4f}\n")
            f.write(f"  DTW Distance: {metrics['dtw_distance']:.4f}\n")
            f.write(f"  RMSE after DTW alignment (3D): {metrics['aligned_rmse']:.4f}\n")
            f.write(f"  RMSE after alignment per axis (X,Y,Z): {metrics['aligned_rmse_per_axis'][0]:.4f}, {metrics['aligned_rmse_per_axis'][1]:.4f}, {metrics['aligned_rmse_per_axis'][2]:.4f}\n\n")

    # Generate plots for magnitudes
    plt.figure(figsize=(10, 6))
    plt.plot(watch_t_sub, mag_wb, label='Watch Before (mag)', color='blue')

    if mag_wa is not None:
        plt.plot(watch_t_sub, mag_wa, label='Watch After Filter', color='orange')

    plt.plot(watch_t_sub, mag_skl, label='Skeleton Acc', color='green')

    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration Magnitude (m/s²)')
    plt.title(f'Watch vs Skeleton Acceleration (Filter={args.filter_type}, Wrist={args.wrist_joint_idx}, Linear={args.is_linear_acc})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, 'acceleration_magnitude.png'), dpi=300)
    plt.close()

    # Plot per-axis acceleration
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes_names = ['X', 'Y', 'Z']

    for i in range(3):
        axes[i].plot(watch_t_sub, wb_sub[:, i], label=f'Watch Before {axes_names[i]}', color='blue')

        if wa_sub is not None:
            axes[i].plot(watch_t_sub, wa_sub[:, i], label=f'Watch After {axes_names[i]}', color='orange')

        axes[i].plot(watch_t_sub, skl_acc_sub[:, i], label=f'Skeleton {axes_names[i]}', color='green')

        axes[i].set_ylabel(f'{axes_names[i]}-Axis Acc. (m/s²)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[2].set_xlabel('Time (s)')
    plt.suptitle(f'Per-Axis Acceleration Comparison (Filter={args.filter_type}, Linear={args.is_linear_acc})')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'per_axis_acceleration.png'), dpi=300)
    plt.close()

    # Plot gyroscope data
    mag_wg = np.linalg.norm(wg_sub, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(watch_t_sub, mag_wg, label='Gyroscope Magnitude', color='purple')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity Magnitude (rad/s)')
    plt.title('Gyroscope Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, 'gyroscope_magnitude.png'), dpi=300)
    plt.close()

    # Plot per-axis gyroscope data
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i in range(3):
        axes[i].plot(watch_t_sub, wg_sub[:, i], label=f'Gyro {axes_names[i]}', color='purple')
        axes[i].set_ylabel(f'{axes_names[i]}-Axis Gyro (rad/s)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[2].set_xlabel('Time (s)')
    plt.suptitle('Per-Axis Gyroscope Data')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'per_axis_gyroscope.png'), dpi=300)
    plt.close()

    # Plot 3D skeleton trajectory if available
    if skl_pos_sub is not None:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(skl_pos_sub[:, 0], skl_pos_sub[:, 1], skl_pos_sub[:, 2], 'g-', linewidth=1, label='Skeleton Trajectory')
        n_markers = 10
        indices = np.linspace(0, len(skl_pos_sub)-1, n_markers, dtype=int)
        ax.scatter(skl_pos_sub[indices, 0], skl_pos_sub[indices, 1], skl_pos_sub[indices, 2], c='red', s=50, label='Time Markers')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'3D Skeleton Trajectory (Wrist={args.wrist_joint_idx})')
        ax.legend()
        plt.savefig(os.path.join(args.out_dir, 'skeleton_3d_trajectory.png'), dpi=300)
        plt.close()

    with open(stats_file, 'a') as f:
        f.write(f"Alignment Results\n")
        f.write(f"===============\n\n")
        f.write(f"Generated plots in {args.out_dir}:\n")
        f.write(f"- Raw vs processed accelerometer and gyroscope data\n")
        if wa_sub is not None:
            f.write(f"- Before and after filter comparison\n")
        f.write(f"- 3D DTW alignment visualizations\n")
        f.write(f"- Per-axis signal comparisons\n")
        if skl_pos_sub is not None:
            f.write(f"- 3D skeleton trajectory\n")
        f.write(f"- Orientation visualizations\n\n")
        f.write(f"DTW alignment metrics saved in alignment_stats.txt\n")

    print(f"Success. Generated visualizations in {args.out_dir}. Linear Acc Mode: {args.is_linear_acc}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
