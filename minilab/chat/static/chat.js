// The chat app. Conversations live in localStorage (this browser only); each turn
// sends the recent messages to /chat/api/chat and streams the answer back.

(() => {
  const app = document.getElementById("chat-app");
  if (!app) return;
  const $ = (id) => document.getElementById(id);
  const STORE_KEY = `minilab.chat.v1.${app.dataset.user}`; // per user: browsers can be shared
  const PREFS_KEY = `minilab.chat.prefs.${app.dataset.user}`;
  const HISTORY_SENT = 10; // the model's context is tiny; the server trims further if needed

  // ---- storage (wrapped: localStorage can be full, disabled or blocked) --------------

  function load(key, fallback) {
    try {
      return JSON.parse(localStorage.getItem(key)) ?? fallback;
    } catch {
      return fallback;
    }
  }
  function save(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* private mode or quota exceeded: keep working in memory */
    }
  }

  let conversations = load(STORE_KEY, []); // [{id, title, updatedAt, messages: [...]}], newest first
  if (!Array.isArray(conversations)) conversations = [];
  // A tab closed mid-answer leaves a message marked as streaming: it isn't anymore.
  conversations.forEach((c) => c.messages?.forEach((m) => (m.streaming = false)));
  let activeId = null;
  let controller = null;
  const prefs = load(PREFS_KEY, {});

  const modelSelect = $("model");
  const calculator = $("calculator");
  if (prefs.model && [...modelSelect.options].some((o) => o.value === prefs.model)) modelSelect.value = prefs.model;
  if (typeof prefs.calculator === "boolean") calculator.checked = prefs.calculator;
  const savePrefs = () => save(PREFS_KEY, { model: modelSelect.value, calculator: calculator.checked });
  modelSelect.addEventListener("change", savePrefs);
  calculator.addEventListener("change", savePrefs);

  const persist = () => save(STORE_KEY, conversations.slice(0, 100));
  const active = () => conversations.find((c) => c.id === activeId) || null;

  // ---- sidebar -----------------------------------------------------------------

  function renderList() {
    const list = $("conversation-list");
    list.replaceChildren(...conversations.map((c) => {
      const item = document.createElement("li");
      item.className = "group flex items-center rounded-lg " + (c.id === activeId ? "bg-reagent-soft" : "hover:bg-sunken");
      const open = document.createElement("button");
      open.type = "button";
      open.className = "min-w-0 flex-1 truncate px-3 py-1.5 text-left text-sm " + (c.id === activeId ? "font-semibold" : "text-muted group-hover:text-ink");
      open.textContent = c.title;
      if (c.id === activeId) open.setAttribute("aria-current", "true");
      open.addEventListener("click", () => select(c.id));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "mr-1 rounded p-1 text-muted opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-bad";
      remove.setAttribute("aria-label", `Delete “${c.title}”`);
      remove.innerHTML = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
      remove.addEventListener("click", () => removeConversation(c.id));
      item.append(open, remove);
      return item;
    }));
    $("conversation-empty").hidden = conversations.length > 0;
  }

  function select(id) {
    if (controller) controller.abort();
    activeId = id;
    renderList();
    renderMessages();
    closeSidebar();
  }

  function removeConversation(id) {
    conversations = conversations.filter((c) => c.id !== id);
    persist();
    if (activeId === id) activeId = null;
    renderList();
    renderMessages();
  }

  function closeSidebar() {
    $("sidebar").dataset.open = "false";
    document.querySelector("[data-nav-backdrop]")?.classList.add("hidden");
  }

  $("new-chat").addEventListener("click", () => {
    select(null);
    $("prompt").focus();
  });

  // ---- messages ------------------------------------------------------------------

  function renderMessages() {
    const list = $("messages");
    const conversation = active();
    const messages = conversation?.messages || [];
    list.replaceChildren(...messages.map(renderMessage));
    list.classList.toggle("hidden", messages.length === 0);
    $("empty-state").classList.toggle("hidden", messages.length > 0);
    scrollToEnd();
  }

  function renderMessage(message) {
    if (message.role === "user") {
      const node = $("tpl-user").content.firstElementChild.cloneNode(true);
      node.querySelector("[data-text]").textContent = message.content;
      return node;
    }
    const node = $("tpl-assistant").content.firstElementChild.cloneNode(true);
    updateAssistant(node, message);
    return node;
  }

  // (Re)draw an assistant message from its data: called on every streamed event.
  function updateAssistant(node, message) {
    node.querySelector("[data-text]").textContent = message.content || (message.streaming && !message.tools?.length ? "…" : "");
    const reasoning = node.querySelector("[data-reasoning]");
    reasoning.hidden = !message.reasoning;
    node.querySelector("[data-reasoning-text]").textContent = message.reasoning || "";
    const tools = node.querySelector("[data-tools]");
    tools.replaceChildren(...(message.tools || []).map((t) => {
      const chip = $("tpl-tool").content.firstElementChild.cloneNode(true);
      chip.querySelector("[data-label]").textContent = `used ${t.name}:`;
      chip.querySelector("[data-value]").textContent = t.ok ? `${t.input} = ${t.output}` : `${t.input} (${t.output})`;
      return chip;
    }));
    const error = node.querySelector("[data-error]");
    error.hidden = !message.error;
    error.textContent = message.error || "";
    if (message.errorCode === "insufficient_quota") {
      error.append(" ", Object.assign(document.createElement("a"), { href: "/billing", className: "link", textContent: "Add credits" }));
    }
    node.querySelector("[data-actions]").hidden = message.streaming || !message.content;
    node.querySelector("[data-meta]").textContent = message.usage
      ? `${message.usage.prompt_tokens} tokens in, ${message.usage.completion_tokens} out${message.cost ? `, ${message.cost}` : ""}`
      : "";
    node.querySelector("[data-copy-message]").onclick = async (event) => {
      try {
        await navigator.clipboard.writeText(message.content);
        event.target.textContent = "Copied";
        setTimeout(() => (event.target.textContent = "Copy"), 1500);
      } catch {}
    };
  }

  function scrollToEnd() {
    const scroller = $("scroller");
    scroller.scrollTop = scroller.scrollHeight;
  }

  // ---- sending ---------------------------------------------------------------------

  function setStreaming(on) {
    const send = $("send");
    send.querySelector('[data-icon="send"]').classList.toggle("hidden", on);
    send.querySelector('[data-icon="stop"]').classList.toggle("hidden", !on);
    send.setAttribute("aria-label", on ? "Stop generating" : "Send message");
  }

  async function send(text) {
    text = text.trim();
    if (!text || controller || modelSelect.disabled) return;
    let conversation = active();
    if (!conversation) {
      conversation = { id: crypto.randomUUID?.() || String(Date.now()), title: text.slice(0, 48), messages: [] };
      conversations.unshift(conversation);
      activeId = conversation.id;
    }
    conversation.messages.push({ role: "user", content: text });
    // Only finished, successful turns go back to the model.
    const history = conversation.messages
      .filter((m) => m.role === "user" || (m.content && !m.error))
      .slice(-HISTORY_SENT)
      .map((m) => ({ role: m.role, content: m.content }));
    const answer = { role: "assistant", content: "", reasoning: "", tools: [], streaming: true };
    conversation.messages.push(answer);
    conversation.updatedAt = Date.now();
    // Most recent conversation first.
    conversations = [conversation, ...conversations.filter((c) => c !== conversation)];
    persist();
    renderList();
    renderMessages();
    const node = $("messages").lastElementChild;

    controller = new AbortController();
    setStreaming(true);
    try {
      await streamChat("/chat/api/chat", {
        model: modelSelect.value,
        messages: history,
        calculator: calculator.checked,
      }, (event) => {
        if (event.type === "delta") {
          if (event.content) answer.content += event.content;
          if (event.reasoning) answer.reasoning += event.reasoning;
        } else if (event.type === "tool") {
          answer.tools.push({ name: event.name, input: event.input, output: event.output, ok: event.ok });
        } else if (event.type === "error") {
          answer.error = event.message;
          answer.errorCode = event.code;
        } else if (event.type === "done") {
          answer.usage = event.usage;
          answer.cost = event.cost;
        }
        updateAssistant(node, answer);
        scrollToEnd();
      }, controller.signal);
    } catch (error) {
      if (error.name !== "AbortError") answer.error = `The connection was interrupted (${error.message}).`;
    } finally {
      answer.streaming = false;
      if (!answer.content && !answer.error && !answer.tools.length) answer.error = "The model returned an empty answer. Try again.";
      updateAssistant(node, answer);
      controller = null;
      setStreaming(false);
      persist();
    }
  }

  const prompt = $("prompt");
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    if (controller) return controller.abort(); // the send button doubles as "stop"
    const text = prompt.value;
    prompt.value = "";
    autosize(prompt);
    send(text);
  });
  prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  prompt.addEventListener("input", () => autosize(prompt));
  document.querySelectorAll("[data-suggestion]").forEach((button) => {
    button.addEventListener("click", () => send(button.textContent));
  });

  // Start on a new chat (with the suggestions); past chats are in the sidebar.
  renderList();
  renderMessages();
  prompt.focus();
})();
