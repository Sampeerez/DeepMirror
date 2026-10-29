import React, { useEffect, useRef, useState } from 'react';

const DENOISER_LABELS = { neural: 'DeepFilterNet', spectral: 'Spectral', off: 'Off' };

export default function StreamDiagnostics({ streamInfo, playbackStats, transport = {}, serverMs = 0, serverStats = {} }) {
  const previous = useRef(0);
  const [recentIssue, setRecentIssue] = useState(false);
  const running = Boolean(streamInfo);
  const loss = (transport.dropped || 0) + (transport.timedOut || 0)
    + (playbackStats?.underruns || 0) + (playbackStats?.droppedSamples || 0);
  useEffect(() => {
    if (!running || loss < previous.current) { previous.current = loss; setRecentIssue(false); return undefined; }
    if (loss === previous.current) return undefined;
    previous.current = loss;
    setRecentIssue(true);
    const timer = setTimeout(() => setRecentIssue(false), 5000);
    return () => clearTimeout(timer);
  }, [loss, running]);

  const rate = streamInfo?.sampleRate || 0;
  const chunkMs = rate ? streamInfo.chunkSize / rate * 1000 : 0;
  const ms = (value) => rate && value != null ? `${Math.round(value)} ms` : '--';
  const rows = [
    ['Capture chunk', ms(chunkMs)],
    ['Queued capture', ms((transport.pending || 0) * chunkMs)],
    ['Playback buffered', ms(playbackStats == null ? null : playbackStats.bufferedSamples / rate * 1000)],
    ['Browser output latency', ms(streamInfo?.outputLatency == null ? null : streamInfo.outputLatency * 1000)],
    ['In flight / queued', rate ? `${transport.inFlight || 0} / ${transport.pending || 0}` : '--'],
    ['Dropped / timed out frames', rate ? `${transport.dropped || 0} / ${transport.timedOut || 0}` : '--'],
    ['Playback underruns', rate && playbackStats ? playbackStats.underruns : '--'],
    ['Playback trimmed', ms(playbackStats == null ? null : playbackStats.droppedSamples / rate * 1000)],
    ['Mic level / sleep threshold', serverStats.inputLevelDb == null ? '--'
      : `${Math.round(serverStats.inputLevelDb)} / ${Math.round(serverStats.silenceThresholdDb ?? -60)} dB`],
    ['Voice detector / speech now', serverStats.voiceDetector == null ? '--'
      : `${serverStats.voiceDetector === 'neural' ? 'Silero VAD' : 'Level'} / ${Math.round((serverStats.speechProbability || 0) * 100)}%`],
    ['Noise filter / removed', serverStats.voiceDetector == null ? '--'
      : `${DENOISER_LABELS[serverStats.denoiser] || '--'} / ${serverStats.noiseReducedDb ?? 0} dB`],
    ['Volume match', serverStats.voiceDetector == null ? '--'
      : `${(serverStats.loudnessGainDb ?? 0) > 0 ? '+' : ''}${serverStats.loudnessGainDb ?? 0} dB`],
    ['Model / effects time', rate ? `${serverStats.modelMs > 0 ? Math.round(serverStats.modelMs) : '--'} / ${
      serverStats.dspMs > 0 ? Math.max(1, Math.round(serverStats.dspMs)) : '--'} ms` : '--'],
  ];
  return (
    <section className="border-t border-line pt-4" aria-label="Stream diagnostics">
      <h3 className="text-sm font-semibold text-fg-secondary">Stream diagnostics</h3>
      <dl className="mt-3 space-y-2 text-xs">
        {rows.map(([label, value]) => <div key={label} className="flex justify-between gap-3">
          <dt className="min-w-0 truncate text-fg-subtle">{label}</dt><dd className="shrink-0 font-mono tabular-nums text-fg-secondary">{value}</dd>
        </div>)}
      </dl>
      {streamInfo && (recentIssue || serverMs > chunkMs) && <p role="status" className="mt-3 text-xs text-warn-fg">
        {serverMs > chunkMs
          ? 'Processing exceeds the capture interval. Select Stable, use a lighter F0 method, or a larger buffer.'
          : transport.dropped || transport.timedOut
            ? 'Frames were lost before playback. Keep the selected model loaded and reduce background CPU load.'
            : playbackStats?.underruns
              ? 'Playback ran out of converted audio. Try the Stable profile; check the output device if it persists.'
              : 'Recent audio discontinuities detected. Check the output device and browser audio load.'}
      </p>}
    </section>
  );
}
