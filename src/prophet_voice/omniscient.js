// The voice's part of Omniscient, through the plugin API only (Omniscient's own page is
// never changed): a page of the dialog and a page of the settings, under Plugins in the
// menu, and a Private button by the message field.
//
// The dialog's page is a cloud of points: it breathes while waiting, turns faster while
// Prophet thinks, and widens with the owner's voice and with Prophet's; beside it, what
// is said in the chat meanwhile. The microphone and the speakers are the voice's own
// (`prophet-voice run` on the Mac): its microphone is open only while the dialog is. This
// script tells it when the dialog begins and ends and follows how it is, at the plugin's
// address (dialog.py). A dialog lives while the browser page that began it is open; the
// button again or Esc on the dialog's page ends it at once.
const STYLE = `
.pv-view{display:flex;flex-direction:column;gap:1rem}
.pv-stage{position:relative;height:55vh;min-height:18rem;overflow:hidden;border-radius:1rem}
.pv-stage canvas{position:absolute;inset:0;width:100%;height:100%}
.pv-caption{position:absolute;left:0;right:0;bottom:12%;padding:0 1rem;display:flex;align-items:center;justify-content:center;gap:.5rem;font-size:.875rem;line-height:1.25rem;font-weight:500;letter-spacing:.01em;color:rgb(63 63 70/.8)}
.dark .pv-caption{color:rgb(228 228 231/.8)}
.pv-spinner{width:.9rem;height:.9rem;flex-shrink:0;border-radius:9999px;border:2px solid currentColor;border-right-color:transparent;animation:pv-turn .8s linear infinite}
@keyframes pv-turn{to{transform:rotate(360deg)}}
.pv-said{display:flex;flex-direction:column;gap:.75rem;max-height:55vh;overflow-y:auto}
.pv-said p{white-space:pre-wrap;font-size:.875rem;line-height:1.4rem}
.pv-pressed[aria-pressed="true"]{color:#059669}
.dark .pv-pressed[aria-pressed="true"]{color:#34d399}
@media (min-width:1024px){.pv-view{flex-direction:row}.pv-stage{flex:1 1 0%;height:70vh}.pv-side{flex:none;width:22rem}.pv-said{max-height:62vh}}`;
const quoted = (list) => list?.length ? `«${list[0][0].toUpperCase()}${list[0].slice(1)}»` : "«Пророк»";
const CAPTIONS = { idle: "Press Private to talk", away: "The voice is not running on this Mac",
  starting: "Opening the microphone…", waiting: (w) => `Say ${quoted(w.wake)} and ask`,
  alone: "I'm listening, just speak", listening: "Listening…", hearing: "Processing voice…", thinking: "Thinking…",
  speaking: "Speaking" };
const SPINNING = new Set(["hearing"]);  // captions with a turning circle beside them
const IMPULSES = { idle: 0.4, away: 0.2, starting: 0.3, waiting: 0.8, listening: 2, hearing: 4, thinking: 7, speaking: 2 };
const SETTLE = 1.5;  // seconds the voice has to take a change before this page follows the voice
const SAID_EVERY = 2000;  // ms between looks at the chat while the dialog's page is open
const SAID_MAX = 30;  // messages shown beside the cloud

// ---- the voice's settings: a page of its own in the menu, under Plugins ---------------
// All of them, in one form. The approved manifest's (the chat, the threshold, speech to
// text) are the core's: filled from prophet.settings() and saved with the owner's API, as
// the modal in Settings → Plugins does; prophet.settings() is the page load's until a
// reload, so the values saved here are kept here. This Mac's own (the microphone, the
// Groq key, the words) the voice keeps, at the plugin's address: it never sends a key
// back, only whether one is set. The voice takes them all as the next dialog begins.
function settingsPage(prophet) {
  let now = { ...prophet.settings() }, form = null;
  prophet.on("settings", (values) => { now = { ...values }; if (form) form.fill(); });
  prophet.ui.page({ id: "settings", title: "Voice settings", icon: "puzzle", render(body) { form = drawSettings(prophet, body, () => now, (v) => { now = v; }); } });
}

const LABELS = { stt: { local: "On this Mac", groq: "Groq first, then this Mac" } };

function drawSettings(prophet, body, values, saved) {
  const voiceAt = prophet.plugin.address;
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    n.append(...kids.filter((k) => k != null)); return n;
  };
  const voice = async (path, body) => {
    const r = await fetch(`${voiceAt}${path}`, body === undefined ? {}
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  };
  const hint = (text) => el("p", { class: "hint mt-1" }, text);
  const status = () => el("p", { class: "hint", "aria-live": "polite" });
  const line = "border-t border-zinc-200 dark:border-zinc-800";
  const section = (title, about, ...kids) => el("div", { class: `space-y-4 px-5 py-4 ${line}` },
    el("div", {}, el("h3", { class: "section-title" }, title), about ? hint(about) : null), ...kids);

  // The core's settings go into these places; this Mac's are drawn once the voice answers.
  const slots = { project: el("div"), threshold: el("div"), stt: el("div"), whisper_model: el("div") };
  const known = el("p", { class: "text-sm", "aria-live": "polite" }, "Asking the voice…");
  const check = el("button", { type: "button", class: "btn btn-sm" }, "Check");
  const listen = el("button", { type: "button", class: "btn btn-sm" }, "Listen");
  const listened = status();
  const microphone = el("div"), groq = el("div"), words = el("div", { class: "space-y-4" });
  const error = el("span", { class: "mr-auto text-sm text-red-600 dark:text-red-400", "aria-live": "polite" });
  const state = el("span", { class: "hint", "aria-live": "polite" });
  const save = el("button", { type: "submit", class: "btn btn-primary" }, "Save");
  const form = el("form", {},
    el("div", { class: "px-5 py-4" }, el("h3", { class: "section-title" }, "Settings"),
      hint("The voice takes them as the next dialog begins. After a change of the local Whisper model, run prophet-voice install on the Mac.")),
    section("Prophet", null, el("div", { class: "flex items-center gap-3" }, el("div", { class: "mr-auto" }, known), check)),
    section("Chat", null, slots.project),
    section("Voice", null,
      el("div", { class: "flex items-center gap-3" }, el("span", { class: "mr-auto text-sm" }, "Prophet's voice: Silero, baya"), listen),
      listened, microphone, slots.threshold),
    section("Speech to text", "Only the phrase you say to Prophet goes to Whisper, and to Groq only when you choose it.",
      slots.stt, slots.whisper_model, groq),
    section("Words", "One variant per line. The wake word is heard by Vosk before anything leaves the Mac, so each is one Russian word that Vosk's small model knows.", words),
    el("div", { class: `flex items-center gap-3 px-5 py-3 ${line}` }, error, state, save));
  body.append(el("section", { class: "panel" }, form));

  let me = null, specs = {}, mac = null, options = null;
  const id = (key) => `voice-setting-${key}`;
  const input = (key) => document.getElementById(id(key));
  function labelled(key, title, about, field) {
    return el("div", {}, el("label", { class: "label", for: id(key) }, title), about ? hint(about) : null, field);
  }
  function field(key, spec) {
    const value = values()[key] ?? spec.default, title = spec.title || key;
    if (spec.type === "boolean") {
      const box = el("input", { id: id(key), type: "checkbox", class: "size-4 accent-blue-600" }); box.checked = !!value;
      return el("label", { class: "flex items-start gap-3 text-sm" }, box,
        el("span", {}, el("span", { class: "font-medium" }, title), spec.description ? hint(spec.description) : null));
    }
    let field;
    if (spec.enum) {
      field = el("select", { id: id(key), class: "input mt-1" },
        ...spec.enum.map((v) => el("option", { value: String(v) }, LABELS[key]?.[v] ?? String(v))));
      field.value = String(value);
    } else {
      const number = spec.type !== "string";
      field = el("input", { id: id(key), class: "input mt-1", type: number ? "number" : "text", autocomplete: "off",
        ...(number ? { step: spec.type === "integer" ? "1" : "any" } : {}),
        ...(spec.minimum !== undefined ? { min: String(spec.minimum) } : {}), ...(spec.maximum !== undefined ? { max: String(spec.maximum) } : {}),
        ...(spec.maxLength ? { maxlength: String(spec.maxLength) } : {}) });
      field.value = value === undefined ? "" : String(value);
    }
    return labelled(key, title, spec.description, field);
  }
  function fill() {
    for (const [key, slot] of Object.entries(slots)) slot.replaceChildren(...(specs[key] ? [field(key, specs[key])] : []));
  }
  // What the form says of the core's settings, as the modal reads it: an empty field is the default.
  function read() {
    const out = {};
    for (const [key, spec] of Object.entries(specs)) {
      const field = input(key);
      if (!field) continue;
      if (spec.type === "boolean") { out[key] = field.checked; continue; }
      const raw = field.value;
      if (raw === "") continue;
      if (spec.type === "string") out[key] = raw;
      else if (spec.enum) out[key] = spec.enum.find((v) => String(v) === raw);
      else out[key] = Number(raw);
    }
    return out;
  }

  // ---- this Mac's settings, from the voice ----
  function fillMac() {
    const mics = el("select", { id: id("microphone"), class: "input mt-1" },
      el("option", { value: "" }, "The system's microphone"), ...options.microphones.map((m) => el("option", { value: m }, m)));
    if (mac.microphone && !options.microphones.includes(mac.microphone))
      mics.append(el("option", { value: mac.microphone }, `${mac.microphone} (not found now)`));
    mics.value = mac.microphone;
    microphone.replaceChildren(labelled("microphone", "Microphone", null, mics));
    const key = el("input", { id: id("groq_key"), class: "input mt-1", type: "password", autocomplete: "off",
      placeholder: mac.groq_key.set ? `saved, ends in …${mac.groq_key.hint}` : "gsk_…" });
    groq.replaceChildren(labelled("groq_key", "Groq key", "Kept on the Mac; an empty field keeps the saved one.", key));
    words.replaceChildren(...options.words.map((kind) => {
      const area = el("textarea", { id: id(`w-${kind.name}`), class: "input mt-1", rows: "2" });
      area.value = mac.words[kind.name].join("\n");
      return labelled(`w-${kind.name}`, kind.label, kind.hint, area);
    }));
  }
  function readMac() {
    return { microphone: input("microphone").value, groq_key: input("groq_key").value,
      words: Object.fromEntries(options.words.map((k) => [k.name, input(`w-${k.name}`).value.split("\n")])) };
  }
  function away(err) {
    const why = `The voice does not answer on this Mac (${err.message}): the microphone, the Groq key and the words are set while it runs.`;
    for (const slot of [microphone, groq, words]) slot.replaceChildren();
    words.append(el("p", { class: "text-sm text-red-600 dark:text-red-400" }, why));
    known.textContent = "The voice is not running on this Mac.";
  }
  async function checkKey() {
    known.textContent = "Asking the core…";
    try {
      const r = await voice("/api/check", {});
      known.textContent = r.error ? (r.name ? `client:${r.name}: ${r.error}` : r.error)
        : `Prophet at ${mac?.core ?? "its address"} knows this voice as client:${r.name} (${r.scopes.join(", ")}).`;
    } catch (err) { known.textContent = `The voice does not answer: ${err.message}`; }
  }
  check.onclick = checkKey;
  listen.onclick = async () => {
    listened.textContent = "Loading the voice…";
    try { await voice("/api/say", {}); listened.textContent = ""; }
    catch (err) { listened.textContent = err.message; }
  };

  async function load() {
    try {
      const { plugins } = await prophet.api("/api/plugins");
      me = plugins.find((p) => p.name === prophet.plugin.name);
      if (!me) throw new Error("the voice is not among the plugins");
      specs = me.manifest?.settings?.properties || {};
      fill();
    } catch (err) { error.textContent = `The settings are not read: ${err.message}`; }
    try {
      ({ values: mac, options } = await voice("/api/settings"));
      fillMac(); checkKey();
    } catch (err) { mac = null; away(err); }
    save.disabled = !me && !mac;
  }
  form.onsubmit = async (e) => {
    e.preventDefault(); error.textContent = ""; state.textContent = "";
    save.disabled = true;
    try {
      if (me) {
        const r = await prophet.api(`/api/plugins/${me.id}/settings`, { method: "PUT", body: JSON.stringify({ values: read() }) });
        saved({ ...r.settings }); fill();
      }
      if (mac) { mac = (await voice("/api/settings", readMac())).values; fillMac(); }
      state.textContent = "Saved. The voice takes them as the next dialog begins.";
    } catch (err) { error.textContent = err.message; }
    finally { save.disabled = false; }
  };
  load();
  return { fill: () => { if (me) fill(); } };
}

export function activate(prophet) {
  try { settingsPage(prophet); } catch (err) { console.error("voice: the settings page", err); }
  try { dialogPage(prophet); } catch (err) { console.error("voice: the dialog", err); }
}

function dialogPage(prophet) {
  const voiceAt = prophet.plugin.address;
  const page = crypto.randomUUID();  // this browser page, to the voice: its dialog ends when it closes
  const make = (tag, cls, attrs = {}, ...kids) => { const n = document.createElement(tag); if (cls) n.className = cls; for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); n.append(...kids); return n; };
  const dark = () => prophet.context().dark;
  const TAB = `plugin:${prophet.plugin.name}:dialog`;

  // ---- the dialog's page and the button by the message field --------------------------
  const style = make("style"); style.textContent = STYLE;
  const stage = make("div", "pv-stage"), canvas = make("canvas"), caption = make("div", "pv-caption", { "aria-live": "polite" });
  stage.append(canvas, caption);
  const pageButton = make("button", "btn btn-primary", { type: "button" }, "Private");
  const problem = make("p", "text-sm text-red-600 dark:text-red-400", { "aria-live": "polite" });
  const said = make("div", "pv-said");
  const side = make("section", "panel pv-side",
    {}, make("div", "flex items-center gap-3 px-5 py-4", {},
      make("div", "mr-auto", {}, make("h3", "section-title", {}, "The dialog"), make("p", "hint mt-1", {}, "What is said in the chat meanwhile."))),
    make("div", "border-t border-zinc-200 px-5 py-3 dark:border-zinc-800", {}, problem, said));
  side.firstChild.append(pageButton);
  const view = make("div", "pv-view", {}, stage, side);
  const shown = prophet.ui.page({ id: "dialog", title: "Voice", icon: "wave", render(body) { body.append(view); startOrb(); } });
  shown.body.append(style);  // the page's own styles, kept with it
  const composer = prophet.ui.composerButton({ icon: "mic", title: "Private", label: "Private", onClick: () => toggle() });
  composer.button.classList.add("pv-pressed");

  // ---- the voice: what it is now, and what this page asked of it ----------------------
  let live = null;     // the voice's latest snapshot; null while it does not answer
  let dialog = null;   // { private } while this page shows a dialog
  let asked = 0;       // when this page last changed the dialog (performance.now)
  let sentProject;     // the chat the voice was told of
  const words = () => ({ wake: live?.wake });

  // Only what changed is set: this runs with every snapshot, twenty times a second.
  function setButtons() {
    const can = !!live, on = !!dialog, w = words();
    const title = !can ? "A dialog needs the voice: run prophet-voice run on the Mac" : on ? "End the dialog (Esc on its page)"
      : `Just the two of you: the microphone opens, speak and Prophet listens`;
    for (const b of [composer.button, pageButton]) {
      if (b.disabled !== !can) b.disabled = !can;
      if (b.title !== title) b.title = title;
    }
    if (composer.button.getAttribute("aria-pressed") !== String(on)) composer.button.setAttribute("aria-pressed", String(on));
    const label = on ? "End the dialog" : "Private";
    if (pageButton.textContent !== label) pageButton.textContent = label;
  }
  async function tell(kind) {
    asked = performance.now(); sentProject = prophet.context().project;
    try {
      const r = await fetch(`${voiceAt}/dialog`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dialog: kind, project: sentProject ?? null, page }) });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `HTTP ${r.status}`);
      problem.textContent = "";
    } catch (err) {
      problem.textContent = `Voice: ${err.message}`;
      if (kind) end(false);
    }
  }
  // The buttons begin the dialog, or end one: a Casual one begun elsewhere too.
  function toggle() {
    if (dialog) end(true); else begin(true, true);
  }
  function begin(alone, ask) {
    dialog = { private: alone }; setButtons();
    if (ask) { tell(alone ? "private" : "casual"); shown.open(); } else sentProject = prophet.context().project;
    startSaid();
  }
  function end(ask) {
    if (!dialog) return;
    dialog = null; setButtons();
    if (ask) tell(null);
  }
  pageButton.onclick = toggle;
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && dialog && prophet.context().tab === TAB) end(true); });
  // Leaving or reloading the page ends its dialog at once: the voice would see the stream
  // close only a moment later, after the new page had come. A beacon is what a leaving
  // page can still send; it can only end a dialog.
  addEventListener("pagehide", () => { if (dialog) navigator.sendBeacon(`${voiceAt}/dialog/end`); });

  // The voice's stream: the picture follows it, and so do the buttons when the dialog was
  // changed elsewhere (another tab) or the voice went away.
  // A refusal (the plugin's address answers, with no voice ready yet) closes an
  // EventSource for good, so it is opened again a little later.
  function follow() {
    const source = new EventSource(`${voiceAt}/dialog/events?page=${page}`);
    source.onmessage = (e) => heard(JSON.parse(e.data));
    source.onerror = () => {
      live = null; end(false); setButtons();
      if (source.readyState === EventSource.CLOSED) setTimeout(follow, 5000);  // else it tries again by itself
    };
  }
  function heard(snapshot) {
    live = snapshot;
    orb.mic = Math.max(orb.mic, live.mic || 0);
    if (performance.now() - asked > SETTLE * 1000) {
      if (live.dialog && (!dialog || dialog.private !== (live.dialog === "private"))) {
        if (dialog) dialog.private = live.dialog === "private"; else begin(live.dialog === "private", false);
      } else if (!live.dialog && dialog) end(false);
    }
    setButtons();
  }
  follow();
  setButtons();

  // ---- what is said in the chat meanwhile, beside the cloud ---------------------------
  // The messages of the chat the dialog asks in, after the dialog began, read with the
  // owner's API while the dialog's page is open (Omniscient reads the chat only on its own
  // page). Shown as text: an answer's Markdown stays as it is.
  let saidAfter = null, saidTimer = 0;
  const chatPath = () => { const p = sentProject ?? prophet.context().project; return "/api/messages" + (p ? `?project_id=${encodeURIComponent(p)}` : ""); };
  async function startSaid() {
    clearTimeout(saidTimer); said.replaceChildren(make("p", "hint", {}, "Nothing yet."));
    try { const msgs = await prophet.api(chatPath()); saidAfter = msgs.length ? msgs[msgs.length - 1].id : 0; } catch { saidAfter = 0; }
    saidTimer = setTimeout(readSaid, SAID_EVERY);
  }
  async function readSaid() {
    if (prophet.context().tab === TAB && saidAfter !== null) {
      try {
        const msgs = (await prophet.api(chatPath())).filter((m) => m.id > saidAfter).slice(-SAID_MAX);
        if (msgs.length) said.replaceChildren(...msgs.map((m) => make("div", "", {},
          make("div", "hint", {}, m.role === "user" ? "You" : "Prophet"), make("p", "", {}, String(m.content ?? "")))));
        said.scrollTop = said.scrollHeight;
      } catch { /* the next look */ }
    }
    if (dialog || prophet.context().tab === TAB) saidTimer = setTimeout(readSaid, SAID_EVERY);
  }

  // ---- the dialog's picture: a cloud of points that breathes, thinks and speaks ----------
  // Points drift inside a sphere that turns slowly; near ones are joined now and then,
  // and impulses run along the joins, more of them while Prophet thinks. Speech widens and
  // narrows the cloud softly: Prophet's voice by the sound the voice plays, and the
  // owner's by the microphone's level.
  const orb = { raf: 0, last: 0, turn: 0, energy: 0, mic: 0, pts: [], links: [], near: [], sparks: [], caption: "" };
  // 300 points in the cloud; only every second one is joined, so the joins are as many as
  // 150 points would have. Around it, a halo of faint points that are never joined.
  function orbCloud() {
    const n = 300, pts = [];
    for (let i = 0; i < n; i++) {
      const y = 1 - 2 * (i + 0.5) / n, r = Math.sqrt(1 - y * y), a = i * 2.39996, k = 0.62 + Math.random() * 0.42;
      pts.push({ x: Math.cos(a) * r * k, y: y * k, z: Math.sin(a) * r * k, ph: Math.random() * 6.283, sp: 0.35 + Math.random() * 0.6 });
    }
    for (let i = 0; i < 56; i++) {
      const y = 2 * Math.random() - 1, r = Math.sqrt(1 - y * y), a = Math.random() * 6.283, k = 1.2 + Math.random() * 0.3;
      pts.push({ x: Math.cos(a) * r * k, y: y * k, z: Math.sin(a) * r * k, ph: Math.random() * 6.283, sp: 0.15 + Math.random() * 0.3, halo: true });
    }
    const links = [], near = pts.map(() => []);
    pts.forEach((p, i) => {
      if (p.halo || i % 2) return;
      pts.map((q, j) => ({ j, d: Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z) }))
        .filter(o => o.j > i && o.j % 2 === 0 && !pts[o.j].halo && o.d < 0.4).sort((a, b) => a.d - b.d).slice(0, 3)
        .forEach(o => { near[i].push(links.length); near[o.j].push(links.length);
                        links.push({ a: i, b: o.j, ph: Math.random() * 6.283, sp: 0.12 + Math.random() * 0.3 }); });
    });
    Object.assign(orb, { pts, links, near, sparks: [] });
  }
  function startOrb() { if (!orb.pts.length) orbCloud(); if (!orb.raf) { orb.last = 0; orb.raf = requestAnimationFrame(orbFrame); } }
  function stopOrb() { cancelAnimationFrame(orb.raf); orb.raf = 0; }
  function dialogMode() {
    if (!live) return "away";
    if (!dialog) return "idle";
    if (!live.dialog) return "starting";
    return IMPULSES[live.state] != null ? live.state : "waiting";
  }
  function orbFrame(now) {
    orb.raf = requestAnimationFrame(orbFrame);
    if (!canvas.offsetParent) return;
    // In a dialog the question goes to the chat open here: tell the voice when it changes.
    if (dialog && performance.now() - asked > 300 && prophet.context().project !== sentProject) tell(dialog.private ? "private" : "casual");
    const t = now / 1000, dt = orb.last ? Math.max(0, Math.min(0.05, t - orb.last)) : 0.016; orb.last = t;
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const mode = dialogMode(), isDark = dark();
    let text = CAPTIONS[mode === "waiting" && dialog?.private ? "alone" : mode];
    if (typeof text === "function") text = text(words());
    if (text !== orb.caption) {
      orb.caption = text; caption.textContent = "";
      if (SPINNING.has(mode)) caption.append(make("span", "pv-spinner", { "aria-hidden": "true" }));
      caption.append(text);
    }

    // How loud now: Prophet's voice, else the owner.
    let level = 0;
    if (mode === "speaking") level = live.voice > 0 ? Math.min(1, live.voice) : 0.3 + 0.1 * Math.sin(t * 2.4);
    else if (mode === "listening") level = orb.mic * 0.6;
    orb.mic *= Math.exp(-dt * 4);
    // Followed slowly both ways: the cloud widens and narrows with the speech, it does not throb.
    orb.energy += (level - orb.energy) * Math.min(1, dt * (level > orb.energy ? 2.6 : 1.6));
    const en = orb.energy;

    // A soft sky-to-rose ground with two slow glows.
    const ground = g.createLinearGradient(0, 0, w, h);
    ground.addColorStop(0, isDark ? "#0d1426" : "#dfeafd"); ground.addColorStop(1, isDark ? "#231329" : "#fbe3ef");
    g.fillStyle = ground; g.fillRect(0, 0, w, h);
    // A glow fades into its own colour: into transparent black, its middle would grey.
    const blob = (x, y, r, rgb, a) => {
      const b = g.createRadialGradient(x, y, 0, x, y, r); b.addColorStop(0, `rgba(${rgb},${a})`); b.addColorStop(1, `rgba(${rgb},0)`);
      g.fillStyle = b; g.fillRect(0, 0, w, h);
    };
    blob(w * (0.3 + 0.08 * Math.sin(t * 0.11)), h * (0.35 + 0.06 * Math.cos(t * 0.13)), Math.max(w, h) * 0.55, isDark ? "59,130,246" : "147,197,253", isDark ? 0.18 : 0.45);
    blob(w * (0.72 + 0.07 * Math.cos(t * 0.09)), h * (0.68 + 0.05 * Math.sin(t * 0.12)), Math.max(w, h) * 0.5, isDark ? "236,72,153" : "249,168,212", isDark ? 0.15 : 0.45);

    // The cloud: it turns faster while thinking and widens with the voice. A slow wave runs
    // down it and sways the points left and right.
    orb.turn += dt * (mode === "thinking" || mode === "hearing" ? 0.45 : 0.14);
    const cx = w / 2, cy = h * 0.46, R = Math.min(w, h) * 0.24 * (1 + 0.035 * Math.sin(t * 1.2) + 0.16 * en);
    const ca = Math.cos(orb.turn), sa = Math.sin(orb.turn), tilt = 0.35 + 0.1 * Math.sin(t * 0.07), ct = Math.cos(tilt), st = Math.sin(tilt);
    const P = orb.pts.map(p => {
      const swell = 1 + en * 0.06 * Math.sin(p.ph + t * 1.5), drift = p.halo ? 0.1 : 0.06;
      let x = (p.x + drift * Math.sin(t * p.sp + p.ph)) * swell, y = (p.y + drift * Math.cos(t * p.sp * 1.3 + p.ph)) * swell,
          z = (p.z + drift * Math.sin(t * p.sp * 0.7 + 2 * p.ph)) * swell;
      [x, z] = [x * ca + z * sa, z * ca - x * sa];
      [y, z] = [y * ct - z * st, y * st + z * ct];
      x += (p.halo ? 0.03 : 0.055) * Math.sin(y * 2.6 - t * 0.8);
      const f = 2.8 / (2.8 + z), depth = (1 - z) / 2;
      return { x: cx + x * R * f, y: cy + y * R * f, f, depth: Math.max(0, Math.min(1, depth)), mix: (x + 1) / 2, halo: p.halo };
    });
    const color = (mix, a) => {
      const [b, p] = isDark ? [[147, 197, 253], [249, 168, 212]] : [[59, 130, 246], [219, 39, 119]];
      const m = Math.max(0, Math.min(1, mix));
      return `rgba(${b.map((v, i) => Math.round(v + (p[i] - v) * m)).join(",")},${a})`;
    };
    const glow = g.createRadialGradient(cx, cy, 0, cx, cy, R * 1.5);
    glow.addColorStop(0, isDark ? `rgba(196,181,253,${0.16 + 0.25 * en})` : `rgba(255,255,255,${0.45 + 0.3 * en})`);
    glow.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = glow; g.fillRect(0, 0, w, h);

    // Joins come and go; speech lights more of them.
    g.lineWidth = 1;
    for (const l of orb.links) {
      const on = Math.max(0, Math.sin(t * l.sp + l.ph)) ** 4 * 0.55 + en * 0.25; if (on < 0.03) continue;
      const a = P[l.a], b = P[l.b];
      g.strokeStyle = color((a.mix + b.mix) / 2, (on * (0.25 + 0.75 * Math.min(a.depth, b.depth))).toFixed(3));
      g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
    }
    // Impulses: a bright point runs along a join and on to the next, a few hops.
    if (Math.random() < dt * (IMPULSES[mode] + en * 8)) {
      const l = Math.floor(Math.random() * orb.links.length);
      orb.sparks.push({ l, from: orb.links[l].a, t: 0, hops: 2 + Math.floor(Math.random() * 4) });
    }
    for (const s of orb.sparks) {
      s.t += dt / 0.55;
      if (s.t >= 1) {
        const l = orb.links[s.l], at = l.a === s.from ? l.b : l.a, next = orb.near[at].filter(i => i !== s.l);
        if (--s.hops <= 0 || !next.length) { s.done = true; continue; }
        Object.assign(s, { l: next[Math.floor(Math.random() * next.length)], from: at, t: 0 });
      }
      const l = orb.links[s.l], a = P[s.from], b = P[l.a === s.from ? l.b : l.a];
      const x = a.x + (b.x - a.x) * s.t, y = a.y + (b.y - a.y) * s.t;
      g.strokeStyle = color(a.mix, 0.55); g.lineWidth = 1.4;
      g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(x, y); g.stroke();
      const spark = g.createRadialGradient(x, y, 0, x, y, 7);
      spark.addColorStop(0, isDark ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,1)"); spark.addColorStop(1, color(a.mix, 0));
      g.fillStyle = spark; g.beginPath(); g.arc(x, y, 7, 0, 6.283); g.fill();
    }
    orb.sparks = orb.sparks.filter(s => !s.done);
    // The points, far ones small and faint; the halo's fainter still.
    for (const p of P) {
      g.fillStyle = color(p.mix, ((p.halo ? 0.15 + 0.3 * p.depth : 0.3 + 0.65 * p.depth) * (p.halo ? 0.6 + 0.4 * Math.sin(t * 0.7 + p.mix * 9) ** 2 : 1)).toFixed(3));
      g.beginPath(); g.arc(p.x, p.y, (p.halo ? 0.7 + 1 * p.depth : 0.8 + 1.3 * p.depth) * p.f * (1 + 0.2 * en), 0, 6.283); g.fill();
    }
  }
}
