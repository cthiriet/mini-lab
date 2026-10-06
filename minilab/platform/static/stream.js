// Streaming chat for the playground and the chat app.
//
// The server answers with Server-Sent Events: one JSON object per "data:" line,
// separated by blank lines. EventSource only does GET, so we POST with fetch()
// and parse the stream ourselves.
//
// Events: {type: "delta", content | reasoning}, {type: "tool", name, input, output, ok},
//         {type: "error", message, code}, {type: "done", usage, cost, latency_ms, ...}

window.streamChat = async function streamChat(url, body, onEvent, signal) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    let message = `The server answered ${response.status}.`;
    try {
      const data = await response.json();
      message = data.error?.message || data.detail?.[0]?.msg || message;
    } catch {}
    if (response.status === 401) message = "Your session has expired. Reload the page and log in again.";
    onEvent({ type: "error", message, status: response.status });
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      for (const line of block.split("\n")) {
        if (line.startsWith("data:")) onEvent(JSON.parse(line.slice(5)));
      }
    }
  }
};

// Tokens leave the server every ~10 ms, but on the way (Wi-Fi above all) they often wait 50-400 ms
// and land in a bunch. Like a video player, smoothText keeps a small buffer: the text starts BUFFER
// ms after the first characters, then plays at the average rate they arrive, a little faster or
// slower as the buffer fills or drains, so a stall slows the text down instead of stopping it.
// text(): everything received so far; streaming(): whether more may come; show(n): put the first n
// characters on screen. Resolves once the stream has ended and everything is on screen.
// setTimeout, not requestAnimationFrame: in a background tab it still runs (once a second), and
// then shows everything at once.
window.smoothText = function smoothText(text, streaming, show) {
  const BUFFER = 250;
  return new Promise((resolve) => {
    let shown = 0; // characters on screen, fractional
    let first = null;
    let last = performance.now();
    const tick = () => {
      const now = performance.now();
      const dt = now - last;
      last = now;
      const received = text().length;
      if (first === null && received) first = now;
      const hidden = received - shown;
      if (hidden > 0 && (now - first >= BUFFER || !streaming())) {
        const rate = received / Math.max(1, now - first); // characters per ms
        const pace = Math.min(2, Math.max(0.5, hidden / Math.max(1, rate * BUFFER)));
        shown = Math.min(received, shown + (streaming() ? rate * dt * pace : Math.max(rate * dt, (hidden * dt) / 150)));
        show(Math.floor(shown));
      }
      if (streaming() || shown < text().length) setTimeout(tick, 16);
      else resolve();
    };
    tick();
  });
};
