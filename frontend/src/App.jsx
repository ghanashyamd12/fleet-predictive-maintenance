import { useState } from "react";
import { motion } from "framer-motion";
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import "./App.css";

export default function App() {
  const [sensors, setSensors] = useState(Array(21).fill(""));
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (index, value) => {
    const updated = [...sensors];
    updated[index] = value;
    setSensors(updated);
  };

  const handlePredict = async () => {
    const numericSensors = sensors.map(Number);

    if (numericSensors.some(isNaN)) {
      alert("Please enter all 21 sensor values.");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch("https://fleet-predictive-maintenance.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sensors: numericSensors }),
      });

      const data = await res.json();
      setPrediction(data.predicted_RUL);
    } catch {
      alert("Error connecting to backend.");
    }

    setLoading(false);
  };

  const getHealth = () => {
    if (prediction === null) return { label: "N/A", color: "#aaa" };
    if (prediction > 80) return { label: "Healthy", color: "#22c55e" };
    if (prediction > 40) return { label: "Warning", color: "#eab308" };
    return { label: "Critical", color: "#ef4444" };
  };

  const health = getHealth();

  const pieData = [
    { name: "Remaining Life", value: prediction ?? 0 },
    { name: "Used Life", value: prediction ? 130 - prediction : 0 },
  ];

  const PIE_COLORS = ["#22c55e", "#ef4444"];

  const barData = sensors
    .slice(0, 8)
    .map((val, i) => ({ name: `S${i + 1}`, value: Number(val) || 0 }));

  const BAR_COLORS = [
    "#6366f1",
    "#8b5cf6",
    "#a855f7",
    "#c084fc",
    "#22c55e",
    "#eab308",
    "#f97316",
    "#ef4444",
  ];

  return (
    <div className="page engine-bg">
      <motion.h1 initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="title">
        Fleet Predictive Maintenance Dashboard
      </motion.h1>

      <div className={prediction === null ? "center-before" : "layout"}>
        {/* Sensor Panel */}
        <div className="card sensor-card">
          <h2>Sensor Inputs</h2>
          <div className="sensor-grid">
            {sensors.map((value, i) => (
              <input
                key={i}
                type="number"
                placeholder={`S${i + 1}`}
                value={value}
                onChange={(e) => handleChange(i, e.target.value)}
              />
            ))}
          </div>

          <button onClick={handlePredict} className="predict-btn">
            {loading ? "Predicting..." : "Predict RUL"}
          </button>
        </div>

        {/* Result Panel */}
        {prediction !== null && (
          <div className="result-area">
            {/* RUL Card */}
            <div className="card center glow">
              <h2>Predicted RUL</h2>
              <div className="rul-number">{prediction.toFixed(1)}</div>
              <p style={{ color: health.color }}>{health.label}</p>
            </div>

            {/* Charts */}
            <div className="charts">
              {/* Pie */}
              <div className="card">
                <h3>Life Usage</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" outerRadius={95} label>
                      {pieData.map((_, i) => (
                        <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Legend />
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Bar */}
              <div className="card">
                <h3>Sensor Snapshot</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={barData}>
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value">
                      {barData.map((_, i) => (
                        <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}