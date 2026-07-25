"""
Star Tracker — Tkinter GUI
Run: python gui.py
"""

import os, sys, threading, queue, time, io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── colour palette ────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"
ACCENT2  = "#3fb950"
WARN     = "#f85149"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
CARD     = "#1c2128"

# ── tiny helpers ──────────────────────────────────────────────────────────────

def _btn(parent, text, cmd, color=ACCENT, width=18):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=color, fg=BG, activebackground=color,
                  activeforeground=BG, relief="flat", bd=0,
                  font=("Segoe UI", 9, "bold"), width=width,
                  cursor="hand2", pady=6)
    b.bind("<Enter>", lambda e: b.config(bg=_lighten(color)))
    b.bind("<Leave>", lambda e: b.config(bg=color))
    return b

def _lighten(hex_col):
    r,g,b = int(hex_col[1:3],16), int(hex_col[3:5],16), int(hex_col[5:7],16)
    r,g,b = min(255,r+30), min(255,g+30), min(255,b+30)
    return f"#{r:02x}{g:02x}{b:02x}"

def _label(parent, text, size=9, bold=False, color=TEXT, bg=None, **kw):
    bg = bg if bg is not None else parent["bg"]
    w = tk.Label(parent, text=text, bg=bg, fg=color,
                 font=("Segoe UI", size, "bold" if bold else "normal"), **kw)
    return w

def _sep(parent):
    return tk.Frame(parent, bg=BORDER, height=1)

# ── main application ───────────────────────────────────────────────────────────

class StarTrackerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("⭐ Star Tracker Attitude Determination")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(900, 640)

        # state
        self._q        = queue.Queue()
        self._running  = False
        self._fits_path = tk.StringVar(value="")
        self._ra_var    = tk.StringVar(value="—")
        self._dec_var   = tk.StringVar(value="—")
        self._roll_var  = tk.StringVar(value="—")
        self._pitch_var = tk.StringVar(value="—")
        self._yaw_var   = tk.StringVar(value="—")
        self._res_var   = tk.StringVar(value="—")
        self._stars_var = tk.StringVar(value="—")
        self._err_var   = tk.StringVar(value="—")
        self._status    = tk.StringVar(value="Ready")
        self._gen_n     = tk.IntVar(value=10)
        self._occ_frac  = tk.DoubleVar(value=0.0)

        self._build_ui()
        self._poll()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── header ──
        hdr = tk.Frame(self, bg=PANEL, pady=10)
        hdr.pack(fill="x")
        _label(hdr, "⭐  STAR TRACKER", size=14, bold=True, color=ACCENT,
               bg=PANEL).pack(side="left", padx=18)
        _label(hdr, "Attitude Determination Pipeline", size=9, color=MUTED,
               bg=PANEL).pack(side="left")
        self._status_lbl = _label(hdr, "● Ready", size=9, color=ACCENT2, bg=PANEL)
        self._status_lbl.pack(side="right", padx=18)

        _sep(self).pack(fill="x")

        # ── main body ──
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=10)

        # left column
        left = tk.Frame(body, bg=BG, width=270)
        left.pack(side="left", fill="y", padx=(0,8))
        left.pack_propagate(False)
        self._build_controls(left)

        # right column
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self._build_results(right)
        self._build_log(right)

    def _card(self, parent, title):
        f = tk.Frame(parent, bg=CARD, bd=0, highlightbackground=BORDER,
                     highlightthickness=1)
        f.pack(fill="x", pady=(0,8))
        _label(f, title, size=8, bold=True, color=MUTED, bg=CARD).pack(
            anchor="w", padx=10, pady=(8,4))
        _sep(f).pack(fill="x", padx=10)
        inner = tk.Frame(f, bg=CARD)
        inner.pack(fill="x", padx=10, pady=8)
        return inner

    def _build_controls(self, parent):
        # ── Section: Load Image ──
        c = self._card(parent, "LOAD IMAGE")
        entry = tk.Entry(c, textvariable=self._fits_path,
                         bg=PANEL, fg=TEXT, insertbackground=TEXT,
                         relief="flat", font=("Consolas",8), bd=4)
        entry.pack(fill="x", pady=(0,6))
        _btn(c, "Browse FITS…", self._browse, width=24).pack(fill="x", pady=2)
        _btn(c, "▶  Process Image", self._run_process, color=ACCENT2, width=24).pack(fill="x", pady=2)

        # ── Section: Quick Actions ──
        c2 = self._card(parent, "QUICK ACTIONS")
        _btn(c2, "🚀  Run Demo", self._run_demo, width=24).pack(fill="x", pady=2)
        _btn(c2, "✔  Validate", self._run_validate, width=24).pack(fill="x", pady=2)
        _btn(c2, "🔬  Occlusion Exp.", self._run_occlusion, width=24).pack(fill="x", pady=2)

        # ── Section: Generate Images ──
        c3 = self._card(parent, "GENERATE SYNTHETIC")
        tk.Frame(c3, bg=CARD).pack(fill="x")
        row = tk.Frame(c3, bg=CARD)
        row.pack(fill="x", pady=(0,4))
        _label(row, "Count:", size=8, color=MUTED, bg=CARD).pack(side="left")
        tk.Spinbox(row, from_=1, to=500, textvariable=self._gen_n,
                   width=6, bg=PANEL, fg=TEXT, buttonbackground=PANEL,
                   relief="flat", font=("Segoe UI",8)).pack(side="right")
        row2 = tk.Frame(c3, bg=CARD)
        row2.pack(fill="x", pady=(0,4))
        _label(row2, "Occlusion:", size=8, color=MUTED, bg=CARD).pack(side="left")
        tk.Spinbox(row2, from_=0.0, to=0.9, increment=0.1,
                   textvariable=self._occ_frac, width=6,
                   bg=PANEL, fg=TEXT, buttonbackground=PANEL,
                   relief="flat", font=("Segoe UI",8)).pack(side="right")
        _btn(c3, "⚙  Generate", self._run_generate, color="#8b5cf6", width=24).pack(fill="x", pady=2)

        # ── Section: Misc ──
        c4 = self._card(parent, "TOOLS")
        _btn(c4, "🗑  Clear Log", self._clear_log, color="#374151", width=24).pack(fill="x", pady=2)

    def _metric(self, parent, label, var, unit="", color=TEXT):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=2)
        _label(row, label, size=8, color=MUTED, bg=CARD).pack(side="left")
        val_lbl = tk.Label(row, textvariable=var, bg=CARD, fg=color,
                           font=("Consolas",9,"bold"), anchor="e")
        val_lbl.pack(side="right")
        if unit:
            _label(row, unit, size=8, color=MUTED, bg=CARD).pack(side="right", padx=(0,2))
        return val_lbl

    def _build_results(self, parent):
        top = tk.Frame(parent, bg=BG)
        top.pack(fill="x", pady=(0,8))

        # Boresight card
        bc = tk.Frame(top, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        bc.pack(side="left", fill="both", expand=True, padx=(0,4))
        _label(bc, "BORESIGHT POINTING", size=8, bold=True, color=MUTED, bg=CARD).pack(
            anchor="w", padx=10, pady=(8,4))
        _sep(bc).pack(fill="x", padx=10)
        inn = tk.Frame(bc, bg=CARD)
        inn.pack(fill="x", padx=10, pady=8)
        self._metric(inn, "Right Ascension", self._ra_var,  "°", ACCENT)
        self._metric(inn, "Declination",      self._dec_var, "°", ACCENT)

        # Euler card
        ec = tk.Frame(top, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        ec.pack(side="left", fill="both", expand=True, padx=(4,4))
        _label(ec, "EULER ANGLES", size=8, bold=True, color=MUTED, bg=CARD).pack(
            anchor="w", padx=10, pady=(8,4))
        _sep(ec).pack(fill="x", padx=10)
        inn2 = tk.Frame(ec, bg=CARD)
        inn2.pack(fill="x", padx=10, pady=8)
        self._metric(inn2, "Roll",  self._roll_var,  "°", "#f0a500")
        self._metric(inn2, "Pitch", self._pitch_var, "°", "#f0a500")
        self._metric(inn2, "Yaw",   self._yaw_var,   "°", "#f0a500")

        # Quality card
        qc = tk.Frame(top, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        qc.pack(side="left", fill="both", expand=True, padx=(4,0))
        _label(qc, "QUALITY METRICS", size=8, bold=True, color=MUTED, bg=CARD).pack(
            anchor="w", padx=10, pady=(8,4))
        _sep(qc).pack(fill="x", padx=10)
        inn3 = tk.Frame(qc, bg=CARD)
        inn3.pack(fill="x", padx=10, pady=8)
        self._metric(inn3, "QUEST Residual", self._res_var,   "arcsec", ACCENT2)
        self._metric(inn3, "Stars Used",      self._stars_var, "",       ACCENT2)
        self._metric(inn3, "GT Error",        self._err_var,   "arcsec", WARN)

    def _build_log(self, parent):
        lf = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        lf.pack(fill="both", expand=True)
        hdr = tk.Frame(lf, bg=CARD)
        hdr.pack(fill="x", padx=10, pady=(8,4))
        _label(hdr, "PIPELINE LOG", size=8, bold=True, color=MUTED, bg=CARD).pack(side="left")
        self._prog = ttk.Progressbar(hdr, mode="indeterminate", length=120)
        self._prog.pack(side="right")
        _sep(lf).pack(fill="x", padx=10)

        self._log = tk.Text(lf, bg=BG, fg=TEXT, font=("Consolas",8),
                            relief="flat", bd=0, state="disabled",
                            wrap="word", insertbackground=TEXT,
                            selectbackground=ACCENT, selectforeground=BG)
        scroll = ttk.Scrollbar(lf, command=self._log.yview)
        self._log.config(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y", padx=(0,4), pady=4)
        self._log.pack(fill="both", expand=True, padx=10, pady=(4,8))

        # colour tags
        self._log.tag_config("ok",    foreground=ACCENT2)
        self._log.tag_config("err",   foreground=WARN)
        self._log.tag_config("hdr",   foreground=ACCENT)
        self._log.tag_config("muted", foreground=MUTED)

    # ── logging helpers ───────────────────────────────────────────────────────

    def _log_write(self, text, tag=""):
        self._log.config(state="normal")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0","end")
        self._log.config(state="disabled")

    def _set_status(self, text, color=ACCENT2):
        self._status_lbl.config(text=f"● {text}", fg=color)

    # ── pipeline runner ───────────────────────────────────────────────────────

    def _lock(self):
        self._running = True
        self._prog.start(12)
        self._set_status("Running…", ACCENT)

    def _unlock(self):
        self._running = False
        self._prog.stop()

    def _reset_results(self):
        for v in (self._ra_var, self._dec_var, self._roll_var,
                  self._pitch_var, self._yaw_var, self._res_var,
                  self._stars_var, self._err_var):
            v.set("—")

    def _poll(self):
        try:
            while True:
                item = self._q.get_nowait()
                if item[0] == "log":
                    self._log_write(item[1], item[2])
                elif item[0] == "result":
                    self._populate_result(item[1])
                elif item[0] == "done":
                    self._unlock()
                    self._set_status(item[1], ACCENT2 if "OK" in item[1] else WARN)
                elif item[0] == "err":
                    self._unlock()
                    self._set_status("Error", WARN)
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _populate_result(self, attitude):
        self._ra_var.set(f"{attitude.ra_boresight:.4f}")
        self._dec_var.set(f"{attitude.dec_boresight:+.4f}")
        self._roll_var.set(f"{attitude.roll:+.4f}")
        self._pitch_var.set(f"{attitude.pitch:+.4f}")
        self._yaw_var.set(f"{attitude.yaw:+.4f}")
        self._res_var.set(f"{attitude.residual_arcsec:.2f}")
        self._stars_var.set(str(attitude.n_stars_used))
        if attitude.angular_error_arcsec is not None:
            self._err_var.set(f"{attitude.angular_error_arcsec:.2f}")
        else:
            self._err_var.set("N/A")

    # ── capturing stdout ──────────────────────────────────────────────────────

    class _StdoutCapture(io.StringIO):
        def __init__(self, q):
            super().__init__()
            self._q = q
        def write(self, s):
            if s.strip():
                tag = "ok" if "✓" in s or "SUCCESS" in s else \
                      "err" if "✗" in s or "Error" in s or "FAILED" in s else \
                      "hdr" if s.startswith("─") or s.startswith("=") else ""
                self._q.put(("log", s.rstrip(), tag))
        def flush(self): pass
        def reconfigure(self, **kwargs): pass   # stub — real stdout has this

    # ── actions ───────────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select FITS image",
            filetypes=[("FITS files","*.fits *.fit *.fts"), ("All files","*.*")])
        if path:
            self._fits_path.set(path)

    def _guard(self):
        if self._running:
            messagebox.showwarning("Busy", "A pipeline task is already running.")
            return True
        return False

    def _spawn(self, fn):
        self._reset_results()
        self._lock()
        cap = self._StdoutCapture(self._q)
        old_stdout = sys.stdout
        def run():
            sys.stdout = cap
            try:
                fn()
                self._q.put(("done","OK — done"))
            except Exception as ex:
                self._q.put(("log", f"ERROR: {ex}", "err"))
                self._q.put(("done","Failed"))
            finally:
                sys.stdout = old_stdout
        threading.Thread(target=run, daemon=True).start()

    # ─ process single image ─
    def _run_process(self):
        if self._guard(): return
        path = self._fits_path.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("No file", "Please select a valid FITS file first.")
            return
        def task():
            import config
            from main import load_resources, process_single_image
            self._q.put(("log", "Loading resources…", "muted"))
            catalogue, tri_db = load_resources()
            attitude = process_single_image(path, catalogue, tri_db, verbose=True)
            if attitude:
                self._q.put(("result", attitude))
        self._spawn(task)

    # ─ demo ─
    def _run_demo(self):
        if self._guard(): return
        def task():
            from main import cmd_demo
            import argparse
            cmd_demo(argparse.Namespace())
        self._spawn(task)

    # ─ validate ─
    def _run_validate(self):
        if self._guard(): return
        def task():
            from main import load_resources
            from validation.run_validation import run_validation
            import config
            self._q.put(("log", "Loading resources…", "muted"))
            catalogue, tri_db = load_resources()
            run_validation(catalogue, tri_db, image_dir=config.SYNTHETIC_TEST_DIR)
        self._spawn(task)

    # ─ occlusion ─
    def _run_occlusion(self):
        if self._guard(): return
        def task():
            from main import load_resources
            from validation.occlusion_experiment import run_occlusion_experiment
            import config
            self._q.put(("log", "Loading resources…", "muted"))
            catalogue, tri_db = load_resources()
            run_occlusion_experiment(catalogue, tri_db,
                                     image_dir=config.SYNTHETIC_TEST_DIR)
        self._spawn(task)

    # ─ generate ─
    def _run_generate(self):
        if self._guard(): return
        n   = self._gen_n.get()
        occ = self._occ_frac.get()
        def task():
            from catalogue.hipparcos import load_catalogue
            from synthetic.image_generator import generate_dataset
            import config
            self._q.put(("log", f"Generating {n} images (occlusion={occ:.0%})…", "muted"))
            catalogue = load_catalogue(config.CATALOGUE_FILE)
            generate_dataset(catalogue, config.SYNTHETIC_TEST_DIR, n,
                             occlusion_fraction=occ)
            self._q.put(("log", f"✓ {n} images saved to {config.SYNTHETIC_TEST_DIR}", "ok"))
        self._spawn(task)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = StarTrackerApp()
    # Centre on screen
    app.update_idletasks()
    w, h = 1020, 700
    x = (app.winfo_screenwidth()  - w) // 2
    y = (app.winfo_screenheight() - h) // 2
    app.geometry(f"{w}x{h}+{x}+{y}")
    app.mainloop()
