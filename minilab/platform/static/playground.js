// Playground: edit a conversation, run it against a model, watch the answer stream in.

(() => {
  const form = document.getElementById("pg-form");
  if (!form) return;
  const $ = (id) => document.getElementById(id);
  const list = $("pg-messages");
  const template = $("pg-message-template");
  const runButton = $("pg-run");
  const stopButton = $("pg-stop");
  const usage = $("pg-usage");
  const temperature = $("pg-temperature");
  let controller = null;

  // ---- messages ---------------------------------------------------------------

  function addMessage(role = "user", content = "") {
    const row = template.content.firstElementChild.cloneNode(true);
    setRole(row, role);
    const text = row.querySelector("textarea");
    text.value = content;
    list.appendChild(row);
    autosize(text);
    return row;
  }

  function setRole(row, role) {
    row.dataset.role = role;
    const label = role === "user" ? "User" : "Assistant";
    row.querySelector("[data-role-toggle]").textContent = label;
    row.querySelector("textarea").setAttribute("aria-label", `${label} message`);
  }

  list.addEventListener("click", (event) => {
    const row = event.target.closest("[data-message]");
    if (!row) return;
    if (event.target.closest("[data-role-toggle]")) setRole(row, row.dataset.role === "user" ? "assistant" : "user");
    if (event.target.closest("[data-remove]")) row.remove();
  });

  $("pg-add").addEventListener("click", () => {
    const last = list.querySelector("[data-message]:last-child");
    addMessage(last?.dataset.role === "user" ? "assistant" : "user").querySelector("textarea").focus();
  });

  $("pg-clear").addEventListener("click", () => {
    list.replaceChildren();
    usage.hidden = true;
    addMessage("user").querySelector("textarea").focus();
  });

  temperature.addEventListener("input", () => {
    $("pg-temperature-value").textContent = Number(temperature.value).toFixed(1);
  });

  // ---- request ------------------------------------------------------------------

  function buildRequest() {
    const messages = [];
    const system = $("pg-system").value.trim();
    if (system) messages.push({ role: "system", content: system });
    for (const row of list.querySelectorAll("[data-message]")) {
      const content = row.querySelector("textarea").value;
      if (content.trim()) messages.push({ role: row.dataset.role, content });
    }
    const maxTokens = parseInt($("pg-max-tokens").value, 10);
    return {
      model: $("pg-model").value,
      messages,
      temperature: Number(temperature.value),
      max_tokens: Number.isFinite(maxTokens) && maxTokens > 0 ? maxTokens : null,
      calculator: $("pg-calculator").checked,
    };
  }

  function setRunning(running) {
    runButton.disabled = running;
    stopButton.hidden = !running;
  }

  async function run() {
    const request = buildRequest();
    if (!request.messages.some((m) => m.role === "user")) {
      const row = list.querySelector("[data-message]") || addMessage("user");
      row.querySelector("textarea").focus();
      return;
    }
    const row = addMessage("assistant");
    const text = row.querySelector("textarea");
    text.readOnly = true;
    text.placeholder = "Waiting for the model…";
    usage.hidden = true;
    setRunning(true);
    controller = new AbortController();
    try {
      await streamChat("/playground/api/chat", request, (event) => onEvent(event, row), controller.signal);
    } catch (error) {
      if (error.name !== "AbortError") showError(row, `The stream was interrupted (${error.message}).`);
    } finally {
      setRunning(false);
      text.readOnly = false;
      text.placeholder = "Enter a message";
      controller = null;
      addMessage("user").querySelector("textarea").focus();
    }
  }

  function onEvent(event, row) {
    const text = row.querySelector("textarea");
    if (event.type === "delta" && event.content) {
      text.value += event.content;
      autosize(text);
    } else if (event.type === "delta" && event.reasoning) {
      const details = row.querySelector("[data-reasoning]");
      details.hidden = false;
      row.querySelector("[data-reasoning-text]").textContent += event.reasoning;
    } else if (event.type === "tool") {
      const tools = row.querySelector("[data-tools]");
      tools.hidden = false;
      const item = document.createElement("li");
      item.className = "inline-flex items-center gap-2 rounded-full border border-rule px-3 py-1 text-xs whitespace-nowrap text-muted";
      item.innerHTML = '<span class="font-semibold"></span><span class="mono text-ink"></span>';
      item.children[0].textContent = `used ${event.name}:`;
      item.children[1].textContent = event.ok ? `${event.input} = ${event.output}` : `${event.input} (${event.output})`;
      tools.appendChild(item);
    } else if (event.type === "error") {
      showError(row, event.message, event.code);
    } else if (event.type === "done") {
      showUsage(event);
      if (!text.value) text.placeholder = "(empty answer)";
    }
  }

  function showError(row, message, code) {
    const box = row.querySelector("[data-error]");
    box.hidden = false;
    box.textContent = message;
    if (code === "insufficient_quota") {
      box.append(" ");
      const link = Object.assign(document.createElement("a"), { href: "/billing", className: "link", textContent: "Add credits" });
      box.append(link);
    }
  }

  function showUsage(done) {
    const parts = [
      ["Tokens in", done.usage.prompt_tokens],
      ["Tokens out", done.usage.completion_tokens],
      ["Cost", done.cost ?? "unknown"],
      ["Time", `${done.latency_ms} ms`],
    ];
    if (done.ttft_ms != null) parts.push(["First token", `${done.ttft_ms} ms`]);
    usage.replaceChildren(...parts.map(([label, value]) => {
      const span = document.createElement("span");
      span.innerHTML = '<span></span> <span class="mono font-semibold text-ink"></span>';
      span.children[0].textContent = label;
      span.children[1].textContent = value;
      return span;
    }));
    const lastId = done.request_ids?.at(-1);
    if (lastId) {
      const link = Object.assign(document.createElement("a"), {
        href: `/logs/${encodeURIComponent(lastId)}`, className: "link ml-auto", textContent: "Open in logs",
      });
      usage.append(link);
    }
    usage.hidden = false;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!controller) run();
  });
  form.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      if (!controller) run();
    }
  });
  stopButton.addEventListener("click", () => controller?.abort());

  // ---- "View code" ----------------------------------------------------------------

  $("pg-code").addEventListener("click", () => {
    const req = buildRequest();
    // Always the temperature: without one, the API uses the model's default, not the playground's.
    const body = { model: req.model, messages: req.messages.length ? req.messages : [{ role: "user", content: "Hello!" }],
                   temperature: req.temperature };
    if (req.max_tokens) body.max_tokens = req.max_tokens;
    if (req.calculator) body.tools = [JSON.parse(document.getElementById("calculator-tool").textContent)];
    const apiUrl = form.dataset.apiUrl;
    const json = JSON.stringify(body, null, 2);
    $("pg-code-curl").textContent =
      `curl ${apiUrl}/v1/chat/completions \\\n  -H "Authorization: Bearer $MINILAB_API_KEY" \\\n` +
      `  -H "Content-Type: application/json" \\\n  -d '${json.replaceAll("'", "'\\''")}'`;
    const args = Object.entries(body).map(([k, v]) => `    ${k}=${JSON.stringify(v, null, 4).replaceAll("\n", "\n    ")},`);
    $("pg-code-python").textContent =
      `import os\nfrom openai import OpenAI\n\nclient = OpenAI(base_url="${apiUrl}/v1", api_key=os.environ["MINILAB_API_KEY"])\n\n` +
      `response = client.chat.completions.create(\n${args.join("\n")}\n)\nprint(response.choices[0].message)`;
    window.highlight($("pg-code-curl"));
    window.highlight($("pg-code-python"));
    $("pg-code-dialog").showModal();
  });

  addMessage("user", "What is 347 + 58?");
})();
