// Streaming quality profiles. Each one trades latency for conversion quality:
// a larger chunk gives the server more time per render, and more past/future
// context gives the model steadier timbre and intonation at window seams.
// Ordered from highest quality to lowest latency.
export const PERFORMANCE_PROFILES = [
  { id: 'quality', label: 'Quality', chunkSize: 8192, lookbackSeconds: 1.0, lookaheadSeconds: 0.15 },
  { id: 'balanced', label: 'Balanced', chunkSize: 4096, lookbackSeconds: 1.0, lookaheadSeconds: 0.1 },
  { id: 'responsive', label: 'Responsive', chunkSize: 2048, lookbackSeconds: 0.5, lookaheadSeconds: 0.05 },
];

function finitePositive(values) {
  return (values || []).map(Number).filter((value) => Number.isFinite(value) && value > 0);
}

export function percentile(values, ratio) {
  const sorted = finitePositive(values).sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1);
  return sorted[Math.max(0, index)];
}

export function bufferDurationMs(chunkSize, sampleRate) {
  const size = Number(chunkSize);
  const rate = Number(sampleRate);
  if (!Number.isFinite(size) || !Number.isFinite(rate) || size <= 0 || rate <= 0) return 0;
  return (size * 1000) / rate;
}

export function profileForChunkSize(chunkSize) {
  const size = Number(chunkSize);
  return PERFORMANCE_PROFILES.find((profile) => profile.chunkSize === size) || {
    id: 'custom',
    label: 'Custom',
    chunkSize: size,
  };
}

/** Context timing the server should use for a chunk size (custom sizes keep server defaults). */
export function streamTimingForChunkSize(chunkSize) {
  const profile = profileForChunkSize(chunkSize);
  if (profile.id === 'custom') return {};
  return { lookback_seconds: profile.lookbackSeconds, lookahead_seconds: profile.lookaheadSeconds };
}

/**
 * Highest-quality profile whose chunk budget covers the measured processing
 * time with headroom (70%) and whose network jitter (p95 round trip) fits.
 */
export function recommendPerformanceProfile({
  latencyHistory,
  sampleRate,
  serverMs,
  serverStats,
  isRunning,
}) {
  const samples = finitePositive(latencyHistory).slice(-30);
  const processingMs = Math.max(
    Number(serverMs) || 0,
    (Number(serverStats?.modelMs) || 0) + (Number(serverStats?.dspMs) || 0)
  );

  if (!isRunning || samples.length < 8 || processingMs <= 0 || serverStats?.inferenceSleeping) {
    return {
      ready: false,
      sampleCount: samples.length,
      p95Ms: percentile(samples, 0.95),
      processingMs,
    };
  }

  const p95Ms = percentile(samples, 0.95);
  const fits = (candidate) => {
    const budgetMs = bufferDurationMs(candidate.chunkSize, sampleRate);
    return processingMs <= budgetMs * 0.7 && p95Ms <= budgetMs * 1.65;
  };
  const profile = PERFORMANCE_PROFILES.find(fits) || PERFORMANCE_PROFILES[0];

  return {
    ready: true,
    profile,
    p95Ms,
    processingMs,
    constrained: !fits(profile),
    sampleCount: samples.length,
  };
}
