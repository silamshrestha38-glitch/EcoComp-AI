"""
EcoComp AI - Part 2 simulation  (fixed for Mac/Windows)
Saves output CSV files in the SAME FOLDER as this script.
"""
import numpy as np
import pandas as pd
import os

# ── saves next to this file, works on any computer ──
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

RNG_SEED = 42
N_RUNS_PER_CONTROLLER = 18
MAX_HOURS = 45 * 24
DT = 1.0

def temp_factor(temp):
    return np.exp(-((temp - 60.0) ** 2) / (2 * 18.0 ** 2))

def moisture_factor(moisture):
    return np.exp(-((moisture - 55.0) ** 2) / (2 * 14.0 ** 2))

def o2_factor(o2):
    return 1.0 / (1.0 + np.exp(-(o2 - 6.0)))

def baseline_controller(state, hours_since_turn):
    temp = state["temp"]
    moisture = state["moisture"]
    fan = 10.0
    water = 0.0
    if temp > 70:
        fan = 100.0
    elif temp < 45:
        fan = 0.0
    if moisture < 40:
        water = 30.0
    elif moisture > 70:
        fan = max(fan, 50.0)
    turn = hours_since_turn >= 4
    return fan, water, turn

def proposed_controller(state, hours_since_turn, target_temp=60.0, target_moisture=55.0):
    temp = state["temp"]
    moisture = state["moisture"]
    o2 = state["o2"]
    ch4 = state["ch4"]
    if o2 < 5 or ch4 > 500 or temp > 75:
        return 100.0, 0.0, True
    fan = 35.0 + 3.5 * (temp - target_temp)
    fan = float(np.clip(fan, 5.0, 100.0))
    water = 0.0
    moisture_error = target_moisture - moisture
    if moisture_error > 5:
        water = float(np.clip(moisture_error * 2.0, 0.0, 50.0))
    elif moisture > target_moisture + 12:
        fan = max(fan, 55.0)
    turn = (hours_since_turn >= 6) or (o2 < 8)
    return fan, water, turn

def run_batch(controller_name, base_rate, ambient_base, init_moisture, rng):
    state = {"temp": ambient_base, "moisture": init_moisture,
             "o2": 18.0, "ch4": 0.0, "progress": 0.0}
    hours_since_turn = 0
    rows = []
    anaerobic_events = 0
    fan_on_hours = 0.0
    water_total = 0.0
    controller = baseline_controller if controller_name == "baseline" else proposed_controller
    min_o2 = 21.0
    hours_o2_below_8 = 0

    for h in range(MAX_HOURS):
        ambient = ambient_base + 3.0 * np.sin(2 * np.pi * h / 24.0) + rng.normal(0, 0.4)
        fan, water, turn = controller(state, hours_since_turn)
        rate = (base_rate * temp_factor(state["temp"])
                * moisture_factor(state["moisture"]) * o2_factor(state["o2"]))
        state["progress"] = float(np.clip(state["progress"] + rate * DT, 0, 100))
        heat_gain = rate * 22.0
        passive_loss = 0.06 * (state["temp"] - ambient)
        fan_cooling = (fan / 100.0) * 0.10 * (state["temp"] - ambient)
        state["temp"] = state["temp"] + heat_gain - passive_loss - fan_cooling + rng.normal(0, 0.3)
        state["temp"] = float(np.clip(state["temp"], ambient - 2, 90))
        evap = 0.004 * (fan / 100.0) * (1 + 0.01 * max(state["temp"] - 40, 0))
        state["moisture"] = state["moisture"] - evap * DT + (water / 25.0)
        state["moisture"] = float(np.clip(state["moisture"], 15, 85))
        water_total += water
        passive_diffusion = 0.010 * (21 - state["o2"])
        fan_diffusion = 0.07 * (fan / 100.0) * (21 - state["o2"])
        consumption = 5.6 * rate
        state["o2"] = state["o2"] + passive_diffusion + fan_diffusion - consumption
        if turn:
            state["o2"] += 2.5
        state["o2"] = float(np.clip(state["o2"], 0, 21))
        min_o2 = min(min_o2, state["o2"])
        if state["o2"] < 8:
            hours_o2_below_8 += 1
        if state["o2"] < 5:
            state["ch4"] = state["ch4"] + (5 - state["o2"]) * 60.0
            anaerobic_events += 1
        else:
            state["ch4"] = max(0.0, state["ch4"] - 80.0)
        if fan >= 50:
            fan_on_hours += 1
        if turn:
            hours_since_turn = 0
        else:
            hours_since_turn += 1
        rows.append({"controller": controller_name, "hour": h, "day": h / 24.0,
                     "temp": state["temp"], "moisture": state["moisture"],
                     "o2": state["o2"], "ch4": state["ch4"], "fan": fan,
                     "water": water, "turned": int(turn), "progress": state["progress"]})
        if state["progress"] >= 90:
            break

    df = pd.DataFrame(rows)
    days_to_maturity = df["day"].iloc[-1] if df["progress"].iloc[-1] >= 85 else np.nan
    in_temp_band = ((df["temp"] >= 50) & (df["temp"] <= 65)).mean()
    in_moist_band = ((df["moisture"] >= 45) & (df["moisture"] <= 65)).mean()
    quality_score = 100 * (0.6 * in_temp_band + 0.4 * in_moist_band)
    summary = {"controller": controller_name, "base_rate": base_rate,
               "ambient_base": ambient_base, "init_moisture": init_moisture,
               "days_to_maturity": days_to_maturity,
               "final_progress": df["progress"].iloc[-1],
               "quality_score": quality_score,
               "anaerobic_event_hours": anaerobic_events,
               "min_o2_pct": min_o2, "hours_o2_below_8pct": hours_o2_below_8,
               "frac_hours_o2_below_8pct": hours_o2_below_8 / len(df),
               "fan_on_hours": fan_on_hours, "water_total_ml": water_total,
               "n_hours_simulated": len(df)}
    return df, summary

def main():
    rng = np.random.default_rng(RNG_SEED)
    all_runs = []
    all_summaries = []
    run_id = 0
    for controller_name in ["baseline", "proposed"]:
        for i in range(N_RUNS_PER_CONTROLLER):
            base_rate = rng.uniform(0.16, 0.26)
            ambient_base = rng.uniform(16, 27)
            init_moisture = rng.uniform(42, 62)
            print(f"  Batch {run_id+1:02d}/36 | Controller: {controller_name:8s} | Rate: {base_rate:.3f}")
            df, summary = run_batch(controller_name, base_rate, ambient_base, init_moisture, rng)
            df["run_id"] = run_id
            summary["run_id"] = run_id
            all_runs.append(df)
            all_summaries.append(summary)
            run_id += 1

    full_df = pd.concat(all_runs, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    ts_path  = os.path.join(OUT_DIR, "all_runs_timeseries.csv")
    sum_path = os.path.join(OUT_DIR, "run_summaries.csv")
    full_df.to_csv(ts_path,  index=False)
    summary_df.to_csv(sum_path, index=False)

    print("\n=== RESULTS SUMMARY ===")
    result = summary_df.groupby("controller")[
        ["days_to_maturity","quality_score","frac_hours_o2_below_8pct",
         "fan_on_hours","water_total_ml"]].mean()
    result.columns = ["Days to Maturity","Quality Score","Anaerobic Risk","Fan Hours","Water mL"]
    print(result.round(2).to_string())
    print(f"\nSaved: {ts_path}")
    print(f"Saved: {sum_path}")

if __name__ == "__main__":
    main()
