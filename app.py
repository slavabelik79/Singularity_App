#!/usr/bin/env python3
"""
Singularity App v2 — 4-chart dashboard
"""

from flask import Flask, render_template, jsonify, request
import numpy as np
import datetime
import sys, os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import singularity_v2_1 as sv2
SingularityModelV2 = sv2.SingularityModelV2


app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


def copy_cfg():
    import json
    return json.loads(json.dumps(sv2.CONFIG))


def pct(arr, p):
    fin = arr[np.isfinite(arr)]
    return float(np.percentile(fin, p)) if len(fin) > 0 else float('inf')


def cdf(arr, x):
    fin = arr[np.isfinite(arr)]
    if len(fin) == 0:
        return 0.0
    return 100.0 * np.sum(fin <= x) / len(arr)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/run', methods=['POST'])
def run():
    try:
        cfg = copy_cfg()
        if request.is_json:
            body = request.get_json()
            cfg['SIMULATION']['n_monte_carlo'] = int(body.get('n', cfg['SIMULATION']['n_monte_carlo']))
            cfg['HARDWARE']['doubling_time_months'] = float(body.get('hw', cfg['HARDWARE']['doubling_time_months']))
            cfg['ALGORITHMS']['doubling_time_months'] = float(body.get('algo', cfg['ALGORITHMS']['doubling_time_months']))
            cfg['SCALING_LAW']['paradigm_ceiling'] = float(body.get('ceil', cfg['SCALING_LAW']['paradigm_ceiling']))
            if 'dw' in body:
                cfg['DATA_WALL']['start_year'] = float(body['dw'])

        model = SingularityModelV2(cfg)
        n = cfg['SIMULATION']['n_monte_carlo']
        now = datetime.datetime.now()

        agi_list, asi_list = [], []
        all_trajectories = []

        for i in range(n):
            d_agi, d_asi, traj = model.run_trajectory(return_all=True)
            agi_list.append((d_agi - now).days / 365.25 if d_agi else float('inf'))
            asi_list.append((d_asi - now).days / 365.25 if d_asi else float('inf'))
            all_trajectories.append(traj)

        agi_a = np.array(agi_list)
        asi_a = np.array(asi_list)

        # ====== 1. MC HISTOGRAM ======
        bins = np.arange(0.5, 16.5, 0.5)
        h_agi, _ = np.histogram(agi_a[np.isfinite(agi_a)], bins=bins)
        h_asi, _ = np.histogram(asi_a[np.isfinite(asi_a)], bins=bins)
        histogram = {
            'labels': [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)],
            'agi': h_agi.tolist(),
            'asi': h_asi.tolist(),
        }

        # ====== 2. CAPABILITY TRAJECTORY ======
        dt = cfg['SIMULATION']['dt_months']
        step = max(1, int(3.0 / dt))
        min_len = min(len(t) for t in all_trajectories if t)
        n_pts = min(min_len // step, 200)

        years_from_base = [(i * step * dt) / 12.0 for i in range(1, n_pts + 1)]
        med_cap, p10_cap, p25_cap, p50_cap, p75_cap, p90_cap = [], [], [], [], [], []

        for i in range(n_pts):
            idx = i * step
            vals = []
            for traj in all_trajectories:
                if idx < len(traj):
                    vals.append(traj[idx][1])
            if vals:
                arr = np.array(vals)
                med_cap.append(pct(arr, 50))
                p10_cap.append(pct(arr, 10))
                p25_cap.append(pct(arr, 25))
                p50_cap.append(pct(arr, 50))
                p75_cap.append(pct(arr, 75))
                p90_cap.append(pct(arr, 90))
            else:
                med_cap.append(0); p10_cap.append(0); p25_cap.append(0)
                p50_cap.append(0); p75_cap.append(0); p90_cap.append(0)

        trajectory = {
            'years': years_from_base,
            'median': med_cap,
            'p10': p10_cap,
            'p25': p25_cap,
            'p50': p50_cap,
            'p75': p75_cap,
            'p90': p90_cap,
            'agi_threshold': cfg['THRESHOLDS']['agi'],
            'asi_threshold': cfg['THRESHOLDS']['asi'],
        }

        # ====== 3. CUMULATIVE PROBABILITY ======
        yq = []
        y = 0.25
        while y <= 10.0:
            yq.append(round(y, 4))
            y += 0.25
        y = 11.0
        while y <= 50.0:
            yq.append(round(y, 4))
            y += 1.0
        cumulative = {
            'x': yq,
            'agi': [cdf(agi_a, y) for y in yq],
            'asi': [cdf(asi_a, y) for y in yq],
        }

        # ====== 4. SENSITIVITY ======
        base_med = pct(agi_a, 50)
        variations = {}
        mini_n = min(600, max(300, n))

        simple_tests = [
            ('hw_fast',   'HW faster', 4.0, 'HARDWARE', 'doubling_time_months'),
            ('hw_slow',   'HW slower', 12.0, 'HARDWARE', 'doubling_time_months'),
            ('algo_fast', 'Algo faster', 3.0, 'ALGORITHMS', 'doubling_time_months'),
            ('algo_slow', 'Algo slower', 9.0, 'ALGORITHMS', 'doubling_time_months'),
            ('ceil_up',   'Ceiling up', 12.0, 'SCALING_LAW', 'paradigm_ceiling'),
            ('ceil_down', 'Ceiling down', 6.0, 'SCALING_LAW', 'paradigm_ceiling'),
        ]

        for key, label, val, section, param in simple_tests:
            c2 = copy_cfg()
            c2['SIMULATION']['n_monte_carlo'] = mini_n
            c2[section][param] = val
            m = SingularityModelV2(c2)
            mini_agi = []
            for _ in range(mini_n):
                da, ds = m.run_trajectory(return_all=False)
                mini_agi.append((da - now).days / 365.25 if da else float('inf'))
            variations[key] = {'label': label, 'agi_median': pct(np.array(mini_agi), 50)}

        dw_tests = [
            ('dw_early', 'DW earlier', 2025.0),
            ('dw_late',  'DW later', 2027.0),
        ]

        for key, label, val in dw_tests:
            c2 = copy_cfg()
            c2['SIMULATION']['n_monte_carlo'] = mini_n
            c2['DATA_WALL']['start_year'] = val
            m = SingularityModelV2(c2)
            mini_agi = []
            for _ in range(mini_n):
                da, ds = m.run_trajectory(return_all=False)
                mini_agi.append((da - now).days / 365.25 if da else float('inf'))
            variations[key] = {'label': label, 'agi_median': pct(np.array(mini_agi), 50)}

        sensitivity = {'base': base_med, 'variations': variations}

        summary = {
            'agi_median': pct(agi_a, 50),
            'asi_median': pct(asi_a, 50),
            'p_asi_by_2028': cdf(asi_a, 2.0),
            'p_asi_by_2030': cdf(asi_a, 4.0),
            'p_asi_by_2035': cdf(asi_a, 9.0),
            'p_asi_by_2040': cdf(asi_a, 14.0),
        }

        return jsonify({
            'status': 'ok',
            'n': n,
            'histogram': histogram,
            'trajectory': trajectory,
            'cumulative': cumulative,
            'sensitivity': sensitivity,
            'summary': summary,
        })

    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("  Singularity App v2  —  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
