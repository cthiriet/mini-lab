// Small progressive enhancements shared by every page. Everything works without JS,
// this only adds copy buttons, the mobile menu, tooltips, tabs and local times.

(() => {
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  // Copy buttons: data-copy="#element" copies that element's value or text.
  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-copy]");
    if (!button) return;
    const target = document.querySelector(button.dataset.copy);
    const text = target ? (target.value ?? target.textContent) : "";
    try {
      await navigator.clipboard.writeText(text);
      flashLabel(button, "Copied");
    } catch {
      if (target && target.select) target.select(); // fallback: select it so ⌘C works
      flashLabel(button, "Press ⌘C");
    }
  });

  function flashLabel(button, label) {
    const original = button.dataset.label || button.textContent;
    button.dataset.label = original;
    button.textContent = label;
    setTimeout(() => (button.textContent = original), 1500);
  }

  // Mobile navigation drawer.
  const sidebar = document.getElementById("sidebar");
  const backdrop = document.querySelector("[data-nav-backdrop]");
  function setNav(open) {
    if (!sidebar) return;
    sidebar.dataset.open = String(open);
    backdrop?.classList.toggle("hidden", !open);
    if (open) sidebar.querySelector("a")?.focus();
  }
  $$("[data-nav-open]").forEach((b) => b.addEventListener("click", () => setNav(true)));
  $$("[data-nav-close]").forEach((b) => b.addEventListener("click", () => setNav(false)));
  backdrop?.addEventListener("click", () => setNav(false));
  document.addEventListener("keydown", (e) => e.key === "Escape" && setNav(false));

  // Org switcher: switch as soon as an org is picked; the last entry creates a new org.
  $$("select[data-org-switch]").forEach((select) => {
    const initial = select.value;
    select.addEventListener("change", () => {
      if (select.value === "__new") {
        select.value = initial;
        window.location.href = "/orgs/new";
      } else {
        select.form.requestSubmit();
      }
    });
  });

  // Ask before destructive actions: <form data-confirm="Are you sure?">.
  document.addEventListener("submit", (event) => {
    const message = event.target.dataset?.confirm;
    if (message && !window.confirm(message)) event.preventDefault();
  });

  // Tooltips for chart bars and badges: data-tip="text" (newlines allowed).
  const tip = document.getElementById("tooltip");
  function showTip(el) {
    if (!tip || !el.dataset.tip) return;
    tip.textContent = el.dataset.tip;
    tip.classList.remove("hidden");
    const r = el.getBoundingClientRect();
    const t = tip.getBoundingClientRect();
    const left = Math.min(Math.max(8, r.left + r.width / 2 - t.width / 2), window.innerWidth - t.width - 8);
    const top = r.top - t.height - 8 < 8 ? r.bottom + 8 : r.top - t.height - 8;
    tip.style.left = `${left}px`;
    tip.style.top = `${top}px`;
  }
  const hideTip = () => tip?.classList.add("hidden");
  $$("[data-tip]").forEach((el) => {
    el.addEventListener("mouseenter", () => showTip(el));
    el.addEventListener("focus", () => showTip(el));
    el.addEventListener("mouseleave", hideTip);
    el.addEventListener("blur", hideTip);
  });
  window.addEventListener("scroll", hideTip, { passive: true });

  // Tabs: [data-tabs] containing role=tab buttons with aria-controls.
  $$("[data-tabs]").forEach((root) => {
    const tabs = $$('[role="tab"]', root);
    function select(tab) {
      tabs.forEach((t) => {
        const on = t === tab;
        t.setAttribute("aria-selected", String(on));
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      });
    }
    tabs.forEach((tab, i) => {
      tab.addEventListener("click", () => select(tab));
      tab.addEventListener("keydown", (e) => {
        const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
        if (!step) return;
        const next = tabs[(i + step + tabs.length) % tabs.length];
        select(next);
        next.focus();
      });
    });
  });

  // Times are rendered in UTC by the server; show them in the viewer's time zone.
  $$("time[data-local]").forEach((el) => {
    const date = new Date(el.getAttribute("datetime"));
    if (isNaN(date)) return;
    el.title = el.textContent;
    el.textContent = el.dataset.local === "date"
      ? date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" })
      : date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  });

  // Textareas that grow with their content.
  window.autosize = (el) => {
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  };
  document.addEventListener("input", (e) => e.target.matches?.("textarea[data-autosize]") && window.autosize(e.target));

  // Syntax colors for code samples: <pre data-lang="python|bash|json|headers">. A few regexes
  // per language cover our snippets; whatever no rule matches stays plain text.
  const STRING = /"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'/;
  const SYNTAX = Object.fromEntries(Object.entries({
    python: {
      comment: /#.*/, string: STRING, number: /\b\d+(?:\.\d+)?\b/,
      keyword: /\b(?:import|from|as|def|return|for|in|if|elif|else|and|or|not|with|True|False|None)\b/,
      function: /\b[A-Za-z_]\w*(?=\()/,
    },
    bash: { string: /"(?:[^"\\]|\\.)*"|'[^']*'/, variable: /\$\w+/, function: /^(?:curl|export)\b/ },
    json: { key: /"(?:[^"\\\n]|\\.)*"(?=\s*:)/, string: STRING, number: /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/, keyword: /\b(?:true|false|null)\b/ },
    headers: { key: /^[\w-]+(?=:)/, number: /\d+/ },
  }).map(([lang, rules]) => [lang, new RegExp(Object.entries(rules).map(([kind, re]) => `(?<${kind}>${re.source})`).join("|"), "gm")]));

  window.highlight = (pre) => {
    const pattern = SYNTAX[pre.dataset.lang];
    if (!pattern) return;
    const el = pre.querySelector("code") || pre;
    const text = el.textContent;
    const out = document.createDocumentFragment();
    let last = 0;
    for (const match of text.matchAll(pattern)) {
      const kind = Object.keys(match.groups).find((k) => match.groups[k] !== undefined);
      out.append(text.slice(last, match.index), Object.assign(document.createElement("span"), { className: `tok-${kind}`, textContent: match[0] }));
      last = match.index + match[0].length;
    }
    out.append(text.slice(last));
    el.replaceChildren(out);
  };
  $$("pre[data-lang]").forEach(window.highlight);
})();
