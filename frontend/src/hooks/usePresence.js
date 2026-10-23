import { useEffect } from 'react';

const RETRY_MS = 1500;

/**
 * Holds a ``/ws/presence`` socket for as long as the page is open. The studio
 * has no console window: when the last page closes, the server shuts down a
 * few seconds later (a reload reconnects in time, so it survives).
 */
export default function usePresence() {
  useEffect(() => {
    if (typeof WebSocket === 'undefined' || typeof location === 'undefined') return undefined;
    let socket = null;
    let closed = false;
    let timer = null;
    const url = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/presence`;
    const open = () => {
      socket = new WebSocket(url);
      socket.onclose = () => {
        if (!closed) timer = setTimeout(open, RETRY_MS);
      };
    };
    open();
    return () => {
      closed = true;
      clearTimeout(timer);
      socket?.close();
    };
  }, []);
}
