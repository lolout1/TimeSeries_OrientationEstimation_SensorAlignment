#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from fastdtw import fastdtw
import scipy.spatial.distance as dist
from sklearn.metrics import mean_squared_error
import pandas as pd
from utils.imu_fusion import MadgwickFilter, KalmanFilter, ExtendedKalmanFilter, align_sensor_data, apply_lowpass_filter
from utils.processor.base import csvloader

def load_csv_and_split(file_path):
    """
    Helper that wraps csvloader(file_path) and splits the returned
    data array into (sensor_data, timestamps).
    If file has >=4 columns, assume col 0 is timestamps, cols 1..3 are x,y,z.
    If file has 3 columns, assume no timestamps and just x,y,z.
    """
    loaded = csvloader(file_path)
    if loaded is None or loaded.shape[0] < 2:
        return None, None

    # If there are >=4 columns, we take first as time, next 3 as sensor
    if loaded.shape[1] >= 4:
        timestamps = loaded[:, 0]
        sensor_data = loaded[:, 1:4]
    else:
        # Only 3 columns => treat them as x,y,z, and build a synthetic time array
        timestamps = np.arange(len(loaded))
        sensor_data = loaded[:, :3]

    # Basic check that we indeed have 3 columns
    if sensor_data.shape[1] < 3:
        return None, None
    return sensor_data, timestamps

def evaluate_filter_parameters(acc_file, gyro_file, filter_type='madgwick', param_values=None, is_linear_acc=True, fs=30.0):
    if not os.path.exists(acc_file) or not os.path.exists(gyro_file):
        return None

    if param_values is None:
        if filter_type == 'madgwick':
            param_values = [0.05, 0.1, 0.15, 0.2, 0.3]
        elif filter_type == 'kalman':
            param_values = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
        elif filter_type == 'ekf':
            param_values = [0.01, 0.05, 0.1, 0.2, 0.3]
        else:
            return None

    # Split CSVs properly
    acc_data, acc_timestamps = load_csv_and_split(acc_file)
    gyro_data, gyro_timestamps = load_csv_and_split(gyro_file)

    if acc_data is None or gyro_data is None:
        return None

    aligned_acc, aligned_gyro, aligned_times = align_sensor_data(
        acc_data, gyro_data, acc_timestamps, gyro_timestamps, fs
    )
    if aligned_acc is None or aligned_gyro is None:
        return None

    results = {}
    for param_value in param_values:
        acc_before = aligned_acc.copy()

        if filter_type == 'madgwick':
            filter_obj = MadgwickFilter(beta=param_value)
        elif filter_type == 'kalman':
            filter_obj = KalmanFilter(process_noise=param_value)
        elif filter_type == 'ekf':
            filter_obj = ExtendedKalmanFilter(measurement_noise=param_value)

        quats = np.zeros((len(aligned_acc), 4))
        for i in range(len(aligned_acc)):
            acc = aligned_acc[i]
            gyro = aligned_gyro[i]

            if is_linear_acc:
                acc_with_gravity = acc.copy()
                acc_with_gravity[2] += 9.81
                quats[i] = filter_obj.update(acc_with_gravity, gyro)
            else:
                quats[i] = filter_obj.update(acc, gyro)

        acc_after = np.zeros_like(aligned_acc)
        for i in range(len(aligned_acc)):
            q = quats[i]
            r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            acc_after[i] = r.apply(aligned_acc[i])

        if not is_linear_acc:
            gravity_vector = np.array([0, 0, 9.81])
            for i in range(len(acc_after)):
                acc_after[i] = acc_after[i] - gravity_vector

        dtw_path, dtw_metrics, aligned_before, aligned_after = compute_3d_alignment_metrics(acc_before, acc_after)

        results[param_value] = {
            'acc_before': acc_before,
            'acc_after': acc_after,
            'quaternions': quats,
            'dtw_metrics': dtw_metrics,
            'aligned_before': aligned_before,
            'aligned_after': aligned_after,
            'dtw_path': dtw_path
        }

    return {'results': results, 'aligned_times': aligned_times, 'aligned_gyro': aligned_gyro}

def compute_3d_alignment_metrics(signal1, signal2):
    if signal1 is None or signal2 is None or len(signal1) < 3 or len(signal2) < 3:
        return None, None, None, None

    from fastdtw import fastdtw
    import scipy.spatial.distance as dist
    from sklearn.metrics import mean_squared_error

    s1_mean, s2_mean = np.mean(signal1, axis=0), np.mean(signal2, axis=0)
    s1_std, s2_std = np.std(signal1, axis=0), np.std(signal2, axis=0)
    s1_std[s1_std < 1e-8], s2_std[s2_std < 1e-8] = 1.0, 1.0

    s1_norm = (signal1 - s1_mean) / s1_std
    s2_norm = (signal2 - s2_mean) / s2_std

    distance, path = fastdtw(s1_norm, s2_norm, dist=dist.euclidean)
    path = np.array(path)
    idx1, idx2 = path[:, 0], path[:, 1]
    s1_aligned = signal1[idx1]
    s2_aligned = signal2[idx2]

    min_len = min(len(signal1), len(signal2))
    rmse_per_axis = np.array([
        np.sqrt(mean_squared_error(signal1[:min_len, i], signal2[:min_len, i]))
        for i in range(3)
    ])
    rmse = np.sqrt(mean_squared_error(signal1[:min_len].flatten(), signal2[:min_len].flatten()))
    corr_per_axis = np.array([
        np.corrcoef(signal1[:min_len, i], signal2[:min_len, i])[0, 1]
        for i in range(3)
    ])
    corr = np.mean(corr_per_axis)

    aligned_rmse_per_axis = np.array([
        np.sqrt(mean_squared_error(s1_aligned[:, i], s2_aligned[:, i]))
        for i in range(3)
    ])
    aligned_rmse = np.sqrt(mean_squared_error(s1_aligned.flatten(), s2_aligned.flatten()))

    metrics = {
        'rmse': rmse,
        'rmse_per_axis': rmse_per_axis,
        'correlation': corr,
        'correlation_per_axis': corr_per_axis,
        'dtw_distance': distance,
        'aligned_rmse': aligned_rmse,
        'aligned_rmse_per_axis': aligned_rmse_per_axis
    }

    return path, metrics, s1_aligned, s2_aligned

def plot_filter_comparison(eval_results, out_dir, filter_type, is_linear_acc):
    param_values = list(eval_results['results'].keys())
    dtw_distances = [eval_results['results'][p]['dtw_metrics']['dtw_distance'] for p in param_values]
    aligned_rmses = [eval_results['results'][p]['dtw_metrics']['aligned_rmse'] for p in param_values]
    correlations = [eval_results['results'][p]['dtw_metrics']['correlation'] for p in param_values]

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(param_values, dtw_distances, 'o-')
    plt.xlabel(f"{filter_type.capitalize()} Parameter Value")
    plt.ylabel('DTW Distance')
    plt.title(f"{filter_type.capitalize()} Parameter Optimization")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(param_values, aligned_rmses, 'o-')
    plt.xlabel(f"{filter_type.capitalize()} Parameter Value")
    plt.ylabel('RMSE After Alignment')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(param_values, correlations, 'o-')
    plt.xlabel(f"{filter_type.capitalize()} Parameter Value")
    plt.ylabel('Average Correlation')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{filter_type}_parameter_optimization.png"), dpi=300)
    plt.close()

    best_param_idx = np.argmin(aligned_rmses)
    best_param = param_values[best_param_idx]
    best_results = eval_results['results'][best_param]

    times = eval_results['aligned_times']
    axes_names = ['X', 'Y', 'Z']

    plt.figure(figsize=(12, 9))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(times, best_results['acc_before'][:, i], label=f'Before {axes_names[i]}')
        plt.plot(times, best_results['acc_after'][:, i], label=f'After {axes_names[i]}')
        plt.ylabel(f'{axes_names[i]}-Axis Acc. (m/s²)')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.suptitle(f'Best {filter_type.capitalize()} Param: {best_param} (Linear Acc: {is_linear_acc})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{filter_type}_best_parameter_comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 9))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(range(len(best_results['aligned_before'])), best_results['aligned_before'][:, i],
                 label=f'Before {axes_names[i]}')
        plt.plot(range(len(best_results['aligned_after'])), best_results['aligned_after'][:, i],
                 label=f'After {axes_names[i]}')
        plt.ylabel(f'{axes_names[i]}-Axis Acc. (aligned)')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.suptitle(f'DTW Aligned Signals ({filter_type}, {best_param})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{filter_type}_dtw_aligned_signals.png"), dpi=300)
    plt.close()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(best_results['aligned_before'][:, 0],
            best_results['aligned_before'][:, 1],
            best_results['aligned_before'][:, 2],
            '-', linewidth=1, label='Before', alpha=0.7)
    ax.plot(best_results['aligned_after'][:, 0],
            best_results['aligned_after'][:, 1],
            best_results['aligned_after'][:, 2],
            '-', linewidth=1, label='After', alpha=0.7)

    ax.set_xlabel('X Acceleration')
    ax.set_ylabel('Y Acceleration')
    ax.set_zlabel('Z Acceleration')
    ax.set_title(f'3D DTW Alignment ({filter_type}, {best_param})')
    ax.legend()

    plt.savefig(os.path.join(out_dir, f"{filter_type}_3d_dtw_alignment.png"), dpi=300)
    plt.close()

    return best_param

def compare_filter_types(acc_file, gyro_file, out_dir, is_linear_acc=True, fs=30.0):
    filter_params = {
        'madgwick': 0.15,
        'kalman': 1e-5,
        'ekf': 0.05
    }

    acc_data, acc_timestamps = load_csv_and_split(acc_file)
    gyro_data, gyro_timestamps = load_csv_and_split(gyro_file)
    if acc_data is None or gyro_data is None:
        return None, None

    aligned_acc, aligned_gyro, aligned_times = align_sensor_data(
        acc_data, gyro_data, acc_timestamps, gyro_timestamps, fs
    )
    if aligned_acc is None or aligned_gyro is None:
        return None, None

    results = {}
    for filter_type, param in filter_params.items():
        acc_before = aligned_acc.copy()

        if filter_type == 'madgwick':
            filter_obj = MadgwickFilter(beta=param)
        elif filter_type == 'kalman':
            filter_obj = KalmanFilter(process_noise=param)
        elif filter_type == 'ekf':
            filter_obj = ExtendedKalmanFilter(measurement_noise=param)

        quats = np.zeros((len(aligned_acc), 4))
        for i in range(len(aligned_acc)):
            acc = aligned_acc[i]
            gyro = aligned_gyro[i]

            if is_linear_acc:
                acc_with_gravity = acc.copy()
                acc_with_gravity[2] += 9.81
                quats[i] = filter_obj.update(acc_with_gravity, gyro)
            else:
                quats[i] = filter_obj.update(acc, gyro)

        acc_after = np.zeros_like(aligned_acc)
        for i in range(len(aligned_acc)):
            q = quats[i]
            r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            acc_after[i] = r.apply(aligned_acc[i])

        if not is_linear_acc:
            gravity_vector = np.array([0, 0, 9.81])
            for i in range(len(acc_after)):
                acc_after[i] = acc_after[i] - gravity_vector

        dtw_path, dtw_metrics, aligned_before, aligned_after = compute_3d_alignment_metrics(acc_before, acc_after)
        results[filter_type] = {
            'acc_before': acc_before,
            'acc_after': acc_after,
            'quaternions': quats,
            'dtw_metrics': dtw_metrics,
            'aligned_before': aligned_before,
            'aligned_after': aligned_after,
            'dtw_path': dtw_path,
            'param': param
        }

    plt.figure(figsize=(12, 8))

    metrics = ['dtw_distance', 'aligned_rmse', 'correlation']
    metric_titles = [
        'DTW Distance (lower is better)',
        'RMSE After Alignment (lower is better)',
        'Average Correlation (higher is better)'
    ]

    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i+1)
        values = [results[ft]['dtw_metrics'][metric] for ft in results]
        filter_types = list(results.keys())
        plt.bar(filter_types, values)
        plt.xlabel('Filter Type')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(metric_titles[i])
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "filter_types_comparison.png"), dpi=300)
    plt.close()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(15, 10))
    for i, filter_type in enumerate(['madgwick', 'kalman', 'ekf']):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.plot(results[filter_type]['aligned_before'][:, 0],
                results[filter_type]['aligned_before'][:, 1],
                results[filter_type]['aligned_before'][:, 2],
                '-', linewidth=1, label='Before', alpha=0.7)
        ax.plot(results[filter_type]['aligned_after'][:, 0],
                results[filter_type]['aligned_after'][:, 1],
                results[filter_type]['aligned_after'][:, 2],
                '-', linewidth=1, label='After', alpha=0.7)
        ax.set_title(f'{filter_type.capitalize()} (param={results[filter_type]["param"]})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    plt.suptitle('3D DTW Alignment Comparison Between Filter Types', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "filter_types_3d_comparison.png"), dpi=300)
    plt.close()

    # Find best filter by aligned RMSE
    best_filter = min(results.keys(), key=lambda x: results[x]['dtw_metrics']['aligned_rmse'])
    metrics_table = pd.DataFrame({
        'Filter Type': list(results.keys()),
        'Parameter': [results[ft]['param'] for ft in results],
        'DTW Distance': [results[ft]['dtw_metrics']['dtw_distance'] for ft in results],
        'RMSE After DTW': [results[ft]['dtw_metrics']['aligned_rmse'] for ft in results],
        'Correlation': [results[ft]['dtw_metrics']['correlation'] for ft in results]
    })
    metrics_table.to_csv(os.path.join(out_dir, "filter_comparison_metrics.csv"), index=False)

    print(f"Best filter type: {best_filter} with parameter {results[best_filter]['param']}")
    print(f"DTW Distance: {results[best_filter]['dtw_metrics']['dtw_distance']:.4f}")
    print(f"RMSE After DTW: {results[best_filter]['dtw_metrics']['aligned_rmse']:.4f}")
    print(f"Correlation: {results[best_filter]['dtw_metrics']['correlation']:.4f}")

    with open(os.path.join(out_dir, "best_filter.txt"), 'w') as f:
        f.write(f"Best filter type: {best_filter} with parameter {results[best_filter]['param']}\n")
        f.write(f"DTW Distance: {results[best_filter]['dtw_metrics']['dtw_distance']:.4f}\n")
        f.write(f"RMSE After DTW: {results[best_filter]['dtw_metrics']['aligned_rmse']:.4f}\n")
        f.write(f"Correlation: {results[best_filter]['dtw_metrics']['correlation']:.4f}\n")

    return best_filter, results[best_filter]['param']

def main():
    parser = argparse.ArgumentParser(description="Evaluate IMU filter parameters and generate visualizations")
    parser.add_argument('--acc-file', required=True, help='Path to accelerometer CSV file')
    parser.add_argument('--gyro-file', required=True, help='Path to gyroscope CSV file')
    parser.add_argument('--out-dir', default='filter_evaluation', help='Output directory for visualizations')
    parser.add_argument('--filter-type', default='all', help='Filter type to evaluate (madgwick, kalman, ekf, all)')
    parser.add_argument('--is-linear-acc', action='store_true', help='Whether input acceleration data already has gravity removed')
    parser.add_argument('--optimize-params', action='store_true', help='Whether to optimize filter parameters')
    parser.add_argument('--fs', type=float, default=30.0, help='Sampling frequency in Hz')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.acc_file) or not os.path.exists(args.gyro_file):
        print(f"Error: Input files not found")
        return 1

    print(f"Evaluating filters for accelerometer data: {args.acc_file}")
    print(f"Using gyroscope data: {args.gyro_file}")
    print(f"Output directory: {args.out_dir}")
    print(f"Linear acceleration mode: {args.is_linear_acc}")

    if args.optimize_params:
        best_params = {}
        if args.filter_type == 'all' or args.filter_type == 'madgwick':
            print("Optimizing Madgwick filter parameters...")
            madgwick_results = evaluate_filter_parameters(
                args.acc_file, args.gyro_file, 'madgwick',
                [0.05, 0.1, 0.15, 0.2, 0.3],
                args.is_linear_acc, args.fs
            )
            if madgwick_results is not None:
                best_params['madgwick'] = plot_filter_comparison(madgwick_results, args.out_dir, 'madgwick', args.is_linear_acc)

        if args.filter_type == 'all' or args.filter_type == 'kalman':
            print("Optimizing Kalman filter parameters...")
            kalman_results = evaluate_filter_parameters(
                args.acc_file, args.gyro_file, 'kalman',
                [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
                args.is_linear_acc, args.fs
            )
            if kalman_results is not None:
                best_params['kalman'] = plot_filter_comparison(kalman_results, args.out_dir, 'kalman', args.is_linear_acc)

        if args.filter_type == 'all' or args.filter_type == 'ekf':
            print("Optimizing EKF parameters...")
            ekf_results = evaluate_filter_parameters(
                args.acc_file, args.gyro_file, 'ekf',
                [0.01, 0.05, 0.1, 0.2, 0.3],
                args.is_linear_acc, args.fs
            )
            if ekf_results is not None:
                best_params['ekf'] = plot_filter_comparison(ekf_results, args.out_dir, 'ekf', args.is_linear_acc)

        with open(os.path.join(args.out_dir, "best_parameters.txt"), 'w') as f:
            f.write(f"Best Parameters (Linear Acc: {args.is_linear_acc})\n")
            f.write("===============================\n\n")
            for filter_type, param in best_params.items():
                f.write(f"{filter_type.capitalize()}: {param}\n")

    print("Comparing filter types...")
    best_filter, best_param = compare_filter_types(
        args.acc_file, args.gyro_file, args.out_dir, args.is_linear_acc, args.fs
    )
    if best_filter is not None:
        print(f"Analysis complete. Best filter: {best_filter} with parameter {best_param}")
    else:
        print("Analysis complete. No valid data to compare.")
    print(f"All results and visualizations saved to {args.out_dir}")
    return 0

if __name__ == '__main__':
    sys.exit(main())

