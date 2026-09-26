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
