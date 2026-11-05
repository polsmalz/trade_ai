"use client";

import { useEffect, useRef, useState } from "react";
import axios from "axios";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from "lightweight-charts";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100";

const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "EURUSD", "GBPUSD", "XAUUSD"];
const INTERVALS = ["5m", "15m", "1h", "4h", "1d"];
const HORIZONS = [1, 2, 4, 12, 24];

type Signal = {
  symbol: string;
  price: number | null;
  rsi: number | null;
  signal: string;
  error?: string;
};
type LegacyPrediction = { symbol: string; prediction: string; confidence: number };
type HorizonPrediction = {
  symbol: string;
  prediction: string;
  confidence: number;
  horizon: string;
  model: string;
  disclaimer: string;
};
type Candle = { time: number; open: number; high: number; low: number; close: number };

function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    return err.response?.data?.detail ?? err.message;
  }
  return String(err);
}

export default function Home() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [interval, setIntervalValue] = useState("1h");
  const [horizon, setHorizon] = useState(1);

  const [signals, setSignals] = useState<Signal[]>([]);
  const [legacy, setLegacy] = useState<LegacyPrediction | null>(null);
  const [v2, setV2] = useState<HorizonPrediction | null>(null);
  const [v2Error, setV2Error] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  // ---------------- CHART (created once, cleaned up, resizes with its container)
  useEffect(() => {
    const el = chartContainerRef.current;
    if (!el) return;

    const chart = createChart(el, {
      width: el.clientWidth,
      height: 400,
      layout: { background: { color: "#0f0f0f" }, textColor: "#ffffff" },
      grid: { vertLines: { color: "#1f1f1f" }, horzLines: { color: "#1f1f1f" } },
      timeScale: { timeVisible: true },
    });
    chartRef.current = chart;
    seriesRef.current = chart.addCandlestickSeries();

    const observer = new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth }));
    observer.observe(el);

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // ---------------- v1 data: signals, legacy prediction, chart (every 5s; the API caches)
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      const [sig, pred, chart] = await Promise.allSettled([
        axios.get<Signal[]>(`${API}/signals`),
        axios.get<LegacyPrediction>(`${API}/predict/${symbol}`),
        axios.get<Candle[]>(`${API}/chart/${symbol}`),
      ]);
      if (cancelled) return;

      if (sig.status === "fulfilled") setSignals(sig.value.data);
      setLegacy(pred.status === "fulfilled" ? pred.value.data : null);
      if (chart.status === "fulfilled" && seriesRef.current) {
        seriesRef.current.setData(
          chart.value.data.map((c) => ({
            time: (c.time / 1000) as UTCTimestamp,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
          })),
        );
      }
      const failed = [sig, pred, chart].find((r) => r.status === "rejected");
      setApiError(failed && failed.status === "rejected" ? errorMessage(failed.reason) : null);
    };

    load();
    const timer = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [symbol]);

  // ---------------- v2 data: prediction over a chosen time horizon (every 30s)
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const res = await axios.get<HorizonPrediction>(`${API}/v2/predict/${symbol}`, {
          params: { interval, horizon },
        });
        if (cancelled) return;
        setV2(res.data);
        setV2Error(null);
      } catch (err) {
        if (cancelled) return;
        setV2(null);
        setV2Error(errorMessage(err));
      }
    };

    load();
    const timer = setInterval(load, 30000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [symbol, interval, horizon]);

  const select = "bg-gray-800 border border-gray-700 rounded px-2 py-1";

  return (
    <main className="min-h-screen bg-black text-white p-6 md:p-10">
      <h1 className="text-3xl font-bold text-center mb-6">Trade AI v2 Dashboard</h1>

      <div className="flex gap-4 justify-center flex-wrap mb-8">
        <label className="flex items-center gap-2">
          Asset
          <select className={select} value={symbol} onChange={(e) => setSymbol(e.target.value)}>
            {SYMBOLS.map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          Candle
          <select className={select} value={interval} onChange={(e) => setIntervalValue(e.target.value)}>
            {INTERVALS.map((i) => (
              <option key={i}>{i}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          Look ahead (candles)
          <select className={select} value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
            {HORIZONS.map((h) => (
              <option key={h} value={h}>
                {h}
              </option>
            ))}
          </select>
        </label>
      </div>

      {apiError && (
        <p className="text-center text-red-400 mb-6">API problem: {apiError}</p>
      )}

      <div className="flex gap-10 justify-center flex-wrap">
        {/* ---------------- CHART (1-minute candles) ---------------- */}
        <div className="w-full max-w-[800px]">
          <div ref={chartContainerRef} />
        </div>

        {/* ---------------- SIDE PANEL ---------------- */}
        <div className="bg-gray-900 p-6 rounded-xl w-[350px] space-y-6">
          <div>
            <h2 className="text-xl font-bold mb-3">Prediction (next {v2?.horizon ?? "…"})</h2>
            {v2 ? (
              <div className="border-b border-gray-700 pb-3">
                <p><strong>Symbol:</strong> {v2.symbol}</p>
                <p>
                  <strong>Direction:</strong>{" "}
                  <span className={v2.prediction === "UP" ? "text-green-400" : "text-red-400"}>
                    {v2.prediction}
                  </span>
                </p>
                <p><strong>Confidence:</strong> {v2.confidence}%</p>
                <p className="text-xs text-gray-400 mt-2">{v2.disclaimer}</p>
              </div>
            ) : v2Error ? (
              <p className="text-yellow-400 text-sm">{v2Error}</p>
            ) : (
              <p>Loading…</p>
            )}
          </div>

          <div>
            <h2 className="text-xl font-bold mb-3">Legacy 1m model</h2>
            {legacy ? (
              <p>
                {legacy.prediction} ({legacy.confidence}%){" "}
                <span className="text-xs text-gray-400">raw-price model, low value</span>
              </p>
            ) : (
              <p className="text-sm text-gray-400">Unavailable</p>
            )}
          </div>

          <div>
            <h2 className="text-xl font-bold mb-3">Signals (RSI)</h2>
            {signals.map((coin) => (
              <div key={coin.symbol} className="mb-3 border-b border-gray-700 pb-2">
                <p><strong>{coin.symbol}</strong></p>
                {coin.error ? (
                  <p className="text-sm text-yellow-400">Data unavailable</p>
                ) : (
                  <>
                    <p>Price: {coin.price !== null ? `$${coin.price}` : "n/a"}</p>
                    <p>RSI: {coin.rsi ?? "n/a"}</p>
                    <p>Signal: {coin.signal}</p>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
