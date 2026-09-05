"""
Desktop GUI for the news-headline classifier.

Wraps the existing pipeline (data_processor / train_models / visualizer) in a
Tkinter window: train the models, compare their scores, read the classification
reports, view the charts, and classify your own headlines live.
"""

import contextlib
import io
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.tree import plot_tree

from data_processor import load_and_preprocess_data
from train_models import train_and_evaluate_models
from visualizer import generate_performance_visuals

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE_PATH = os.path.join(PROJECT_DIR, 'train.parquet.parquet')

# --- Colour palette (coolors.co/d9ed92-...-184e77) ---
CREAM = "#D9ED92"
LIME  = "#B5E48C"
GREEN = "#99D98C"
MINT  = "#76C893"
TEAL  = "#52B69A"
AQUA  = "#34A0A4"
CYAN  = "#168AAD"
BLUE  = "#1A759F"
DEEP  = "#1E6091"
NAVY  = "#184E77"

WHITE = "#FFFFFF"

# AG News category ids, in the order the dataset stores them
CATEGORY_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
CATEGORY_COLORS = [TEAL, MINT, CYAN, BLUE]

SAMPLE_HEADLINES = [
    "Peace talks resume after weeks of tension along the border",
    "Real Madrid beat Barcelona in extra time to lift the cup",
    "Oil prices climb as OPEC signals a cut to production quotas",
    "NASA telescope spots water vapour on a distant exoplanet",
]

FONT = "Segoe UI"


class NewsClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("News Headline Classifier  |  Logistic Regression vs Decision Tree")
        self.root.geometry("1120x780")
        self.root.minsize(980, 680)
        self.root.configure(bg=NAVY)

        # Filled in once training finishes
        self.vectorizer = None
        self.lr_model = None
        self.dt_model = None
        self.metrics = None
        self.reports_text = ""

        self.message_queue = queue.Queue()

        self._configure_styles()
        self._build_header()
        self._build_control_bar()
        self._build_tabs()
        self._build_footer()

        self.root.after(100, self._drain_queue)

    # ------------------------------------------------------------------ setup

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=NAVY, borderwidth=0)
        style.configure('TNotebook.Tab',
                        background=DEEP, foreground=CREAM,
                        padding=(20, 11), borderwidth=0,
                        font=(FONT, 10, 'bold'))
        style.map('TNotebook.Tab',
                  background=[('selected', AQUA)],
                  foreground=[('selected', WHITE)])
        style.configure('Bar.Horizontal.TProgressbar',
                        background=CREAM, troughcolor=DEEP,
                        borderwidth=0, thickness=8)

    def _accent_button(self, parent, text, command, bg=TEAL, fg=NAVY, state=tk.NORMAL):
        return tk.Button(parent, text=text, command=command, state=state,
                         bg=bg, fg=fg, activebackground=GREEN, activeforeground=NAVY,
                         font=(FONT, 10, 'bold'), relief=tk.FLAT, cursor="hand2",
                         padx=18, pady=9, borderwidth=0,
                         disabledforeground="#7FA6BC")

    def _build_header(self):
        header = tk.Frame(self.root, bg=NAVY)
        header.pack(fill=tk.X, padx=26, pady=(20, 6))

        tk.Label(header, text="News Headline Classifier", bg=NAVY, fg=CREAM,
                 font=(FONT, 22, 'bold')).pack(anchor='w')
        tk.Label(header,
                 text="TF-IDF features  -  Logistic Regression vs Decision Tree  -  AG News (120,000 headlines)",
                 bg=NAVY, fg=GREEN, font=(FONT, 10)).pack(anchor='w', pady=(3, 0))

        tk.Frame(self.root, bg=AQUA, height=3).pack(fill=tk.X, padx=26, pady=(12, 0))

    def _build_control_bar(self):
        bar = tk.Frame(self.root, bg=DEEP)
        bar.pack(fill=tk.X, padx=26, pady=(0, 14))

        inner = tk.Frame(bar, bg=DEEP)
        inner.pack(fill=tk.X, padx=16, pady=14)

        self.train_button = self._accent_button(inner, "Train models", self.start_training, bg=CREAM)
        self.train_button.pack(side=tk.LEFT)

        self.save_button = self._accent_button(inner, "Save charts as PNG", self.save_charts,
                                               bg=AQUA, fg=WHITE, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(10, 0))

        self.status_label = tk.Label(inner, text="Ready - click Train models to begin.",
                                     bg=DEEP, fg=LIME, font=(FONT, 10))
        self.status_label.pack(side=tk.LEFT, padx=(20, 0))

        self.progress = ttk.Progressbar(bar, style='Bar.Horizontal.TProgressbar', mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=16, pady=(0, 14))

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=26, pady=(0, 10))

        self.results_tab = tk.Frame(self.notebook, bg=NAVY)
        self.classify_tab = tk.Frame(self.notebook, bg=NAVY)
        self.reports_tab = tk.Frame(self.notebook, bg=NAVY)
        self.comparison_tab = tk.Frame(self.notebook, bg=NAVY)
        self.tree_tab = tk.Frame(self.notebook, bg=NAVY)

        self.notebook.add(self.results_tab, text="Results")
        self.notebook.add(self.classify_tab, text="Classify a headline")
        self.notebook.add(self.reports_tab, text="Full reports")
        self.notebook.add(self.comparison_tab, text="Comparison chart")
        self.notebook.add(self.tree_tab, text="Decision tree")

        self._build_results_tab()
        self._build_classify_tab()
        self._build_reports_tab()

        for tab, note in ((self.comparison_tab, "The performance chart appears here after training."),
                          (self.tree_tab, "The decision tree diagram appears here after training.")):
            tk.Label(tab, text=note, bg=NAVY, fg=GREEN,
                     font=(FONT, 12)).place(relx=0.5, rely=0.5, anchor='center')

    def _build_footer(self):
        tk.Label(self.root, text="Dataset: " + os.path.basename(DATASET_FILE_PATH) + "   -   " + PROJECT_DIR,
                 bg=NAVY, fg=BLUE, font=(FONT, 8)).pack(anchor='w', padx=26, pady=(0, 10))

    # ------------------------------------------------------------ results tab

    def _build_results_tab(self):
        wrap = tk.Frame(self.results_tab, bg=NAVY)
        wrap.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        wrap.columnconfigure(0, weight=1, uniform='card')
        wrap.columnconfigure(1, weight=1, uniform='card')

        self.score_cards = {
            'lr': self._build_score_card(wrap, 0, "Logistic Regression", CYAN),
            'dt': self._build_score_card(wrap, 1, "Decision Tree (max_depth=20)", TEAL),
        }

        self.verdict_label = tk.Label(wrap, text="No models trained yet.",
                                      bg=NAVY, fg=GREEN, font=(FONT, 11), justify=tk.LEFT,
                                      wraplength=980)
        self.verdict_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=(24, 0))

    def _build_score_card(self, parent, column, title, accent):
        card = tk.Frame(parent, bg=DEEP, highlightbackground=accent, highlightthickness=2)
        card.grid(row=0, column=column, sticky='nsew',
                  padx=(0, 10) if column == 0 else (10, 0))

        tk.Frame(card, bg=accent, height=6).pack(fill=tk.X)
        tk.Label(card, text=title, bg=DEEP, fg=WHITE,
                 font=(FONT, 13, 'bold')).pack(anchor='w', padx=22, pady=(18, 14))

        widgets = {}
        for key, label in (('acc', 'Accuracy'), ('f1', 'Weighted F1-score')):
            block = tk.Frame(card, bg=DEEP)
            block.pack(fill=tk.X, padx=22, pady=(0, 16))

            tk.Label(block, text=label, bg=DEEP, fg=GREEN,
                     font=(FONT, 9, 'bold')).pack(anchor='w')

            value = tk.Label(block, text="-", bg=DEEP, fg=CREAM, font=(FONT, 30, 'bold'))
            value.pack(anchor='w')

            meter = tk.Canvas(block, height=12, bg=NAVY, highlightthickness=0)
            meter.pack(fill=tk.X, pady=(6, 0))

            widgets[key] = value
            widgets[key + '_meter'] = meter

        widgets['accent'] = accent
        return widgets

    def _paint_meter(self, canvas, fraction, colour):
        """Draw a proportional bar that redraws itself when the window resizes."""
        def redraw(_event=None):
            canvas.delete('all')
            width = max(canvas.winfo_width(), 1)
            height = max(canvas.winfo_height(), 1)
            canvas.create_rectangle(0, 0, width, height, fill=NAVY, outline='')
            canvas.create_rectangle(0, 0, width * fraction, height, fill=colour, outline='')

        canvas.bind('<Configure>', redraw)
        redraw()

    # ----------------------------------------------------------- classify tab

    def _build_classify_tab(self):
        wrap = tk.Frame(self.classify_tab, bg=NAVY)
        wrap.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(wrap, text="Type or paste a news headline", bg=NAVY, fg=CREAM,
                 font=(FONT, 12, 'bold')).pack(anchor='w')

        self.headline_input = tk.Text(wrap, height=3, wrap=tk.WORD, bg=DEEP, fg=WHITE,
                                      insertbackground=CREAM, font=(FONT, 12),
                                      relief=tk.FLAT, padx=14, pady=12,
                                      highlightthickness=2, highlightbackground=AQUA,
                                      highlightcolor=CREAM)
        self.headline_input.pack(fill=tk.X, pady=(10, 12))

        actions = tk.Frame(wrap, bg=NAVY)
        actions.pack(fill=tk.X)

        self.classify_button = self._accent_button(actions, "Classify", self.classify_headline,
                                                   bg=CREAM, state=tk.DISABLED)
        self.classify_button.pack(side=tk.LEFT)
        tk.Label(actions, text="Try a sample:", bg=NAVY, fg=GREEN,
                 font=(FONT, 9)).pack(side=tk.LEFT, padx=(18, 8))

        for index, headline in enumerate(SAMPLE_HEADLINES):
            tk.Button(actions, text=CATEGORY_NAMES[index],
                      command=lambda h=headline: self._fill_sample(h),
                      bg=DEEP, fg=CREAM, activebackground=BLUE, activeforeground=WHITE,
                      font=(FONT, 9), relief=tk.FLAT, cursor="hand2",
                      padx=12, pady=5, borderwidth=0).pack(side=tk.LEFT, padx=3)

        panels = tk.Frame(wrap, bg=NAVY)
        panels.pack(fill=tk.BOTH, expand=True, pady=(22, 0))
        panels.columnconfigure(0, weight=1, uniform='pred')
        panels.columnconfigure(1, weight=1, uniform='pred')

        self.prediction_panels = {
            'lr': self._build_prediction_panel(panels, 0, "Logistic Regression", CYAN),
            'dt': self._build_prediction_panel(panels, 1, "Decision Tree", TEAL),
        }

    def _build_prediction_panel(self, parent, column, title, accent):
        panel = tk.Frame(parent, bg=DEEP, highlightbackground=accent, highlightthickness=2)
        panel.grid(row=0, column=column, sticky='nsew',
                   padx=(0, 10) if column == 0 else (10, 0))

        tk.Frame(panel, bg=accent, height=6).pack(fill=tk.X)
        tk.Label(panel, text=title, bg=DEEP, fg=GREEN,
                 font=(FONT, 10, 'bold')).pack(anchor='w', padx=20, pady=(16, 4))

        verdict = tk.Label(panel, text="-", bg=DEEP, fg=WHITE, font=(FONT, 24, 'bold'))
        verdict.pack(anchor='w', padx=20)

        confidence = tk.Label(panel, text="train the models first", bg=DEEP, fg=LIME, font=(FONT, 10))
        confidence.pack(anchor='w', padx=20, pady=(2, 14))

        bars = {}
        for index, name in enumerate(CATEGORY_NAMES):
            row = tk.Frame(panel, bg=DEEP)
            row.pack(fill=tk.X, padx=20, pady=3)

            tk.Label(row, text=name, bg=DEEP, fg=CREAM, font=(FONT, 9),
                     width=9, anchor='w').pack(side=tk.LEFT)
            percent = tk.Label(row, text="0%", bg=DEEP, fg=GREEN, font=(FONT, 9, 'bold'),
                               width=6, anchor='e')
            percent.pack(side=tk.RIGHT)
            meter = tk.Canvas(row, height=10, bg=NAVY, highlightthickness=0)
            meter.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))

            bars[index] = (meter, percent)

        tk.Frame(panel, bg=DEEP, height=16).pack()
        return {'verdict': verdict, 'confidence': confidence, 'bars': bars}

    def _fill_sample(self, headline):
        self.headline_input.delete('1.0', tk.END)
        self.headline_input.insert('1.0', headline)
        if self.vectorizer is not None:
            self.classify_headline()

    # ------------------------------------------------------------ reports tab

    def _build_reports_tab(self):
        wrap = tk.Frame(self.reports_tab, bg=NAVY)
        wrap.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        scrollbar = tk.Scrollbar(wrap, bg=DEEP, troughcolor=NAVY, borderwidth=0)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.reports_widget = tk.Text(wrap, wrap=tk.NONE, bg=DEEP, fg=CREAM,
                                      font=("Consolas", 10), relief=tk.FLAT,
                                      padx=18, pady=16, yscrollcommand=scrollbar.set)
        self.reports_widget.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.reports_widget.yview)

        self.reports_widget.insert(
            '1.0', "The per-class precision / recall / F1 tables appear here after training.")
        self.reports_widget.config(state=tk.DISABLED)

    # --------------------------------------------------------------- training

    def start_training(self):
        if not os.path.exists(DATASET_FILE_PATH):
            self._set_status("Dataset not found: " + DATASET_FILE_PATH, CREAM)
            return

        self.train_button.config(state=tk.DISABLED)
        self.classify_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.progress.start(12)
        self._set_status("Loading dataset and building TF-IDF features...", CREAM)

        threading.Thread(target=self._training_worker, daemon=True).start()

    def _training_worker(self):
        """Runs off the main thread; talks back through the queue only."""
        try:
            splits, vectorizer = load_and_preprocess_data(DATASET_FILE_PATH)
            self.message_queue.put(('status', "Training both models - this takes a minute..."))

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                lr_model, dt_model, lr_acc, dt_acc, lr_f1, dt_f1 = train_and_evaluate_models(*splits)

            self.message_queue.put(('done', {
                'vectorizer': vectorizer,
                'lr_model': lr_model,
                'dt_model': dt_model,
                'metrics': {'lr_acc': lr_acc, 'dt_acc': dt_acc, 'lr_f1': lr_f1, 'dt_f1': dt_f1},
                'reports': captured.getvalue(),
            }))
        except Exception as error:  # surface it in the UI instead of a silent thread death
            self.message_queue.put(('error', type(error).__name__ + ": " + str(error)))

    def _drain_queue(self):
        while True:
            try:
                kind, payload = self.message_queue.get_nowait()
            except queue.Empty:
                break

            if kind == 'status':
                self._set_status(payload, CREAM)
            elif kind == 'error':
                self.progress.stop()
                self.train_button.config(state=tk.NORMAL)
                self._set_status("Training failed - " + payload, CREAM)
            elif kind == 'done':
                self._on_training_finished(payload)

        self.root.after(100, self._drain_queue)

    def _on_training_finished(self, payload):
        self.progress.stop()
        self.vectorizer = payload['vectorizer']
        self.lr_model = payload['lr_model']
        self.dt_model = payload['dt_model']
        self.metrics = payload['metrics']
        self.reports_text = payload['reports']

        self._update_score_cards()
        self._update_reports()
        self._draw_comparison_chart()
        self._draw_tree_chart()

        self.train_button.config(state=tk.NORMAL, text="Re-train models")
        self.classify_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self._set_status("Training complete - open the Classify tab to try it.", LIME)

        for panel in self.prediction_panels.values():
            panel['confidence'].config(text="waiting for a headline")

    # ---------------------------------------------------------------- updates

    def _update_score_cards(self):
        m = self.metrics
        for key, acc, f1 in (('lr', m['lr_acc'], m['lr_f1']), ('dt', m['dt_acc'], m['dt_f1'])):
            card = self.score_cards[key]
            card['acc'].config(text="{:.1%}".format(acc))
            card['f1'].config(text="{:.1%}".format(f1))
            self._paint_meter(card['acc_meter'], acc, card['accent'])
            self._paint_meter(card['f1_meter'], f1, card['accent'])

        gap = abs(m['lr_acc'] - m['dt_acc']) * 100
        winner = "Logistic Regression" if m['lr_acc'] >= m['dt_acc'] else "Decision Tree"
        self.verdict_label.config(
            text="{} wins by {:.1f} accuracy points. TF-IDF produces thousands of sparse features, "
                 "which suits a linear model; a depth-limited tree can only test a handful of words "
                 "per prediction, so it under-fits.".format(winner, gap))

    def _update_reports(self):
        self.reports_widget.config(state=tk.NORMAL)
        self.reports_widget.delete('1.0', tk.END)
        self.reports_widget.insert('1.0', self.reports_text.strip() or "(no report captured)")
        self.reports_widget.config(state=tk.DISABLED)

    def _embed_figure(self, tab, figure):
        for child in tab.winfo_children():
            child.destroy()
        canvas = FigureCanvasTkAgg(figure, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

    def _draw_comparison_chart(self):
        m = self.metrics
        figure = Figure(figsize=(8, 4.6), facecolor=NAVY)
        axes = figure.add_subplot(111, facecolor=DEEP)

        positions = [0, 1]
        width = 0.34
        accuracy = [m['lr_acc'], m['dt_acc']]
        f1_scores = [m['lr_f1'], m['dt_f1']]

        acc_bars = axes.bar([p - width / 2 for p in positions], accuracy, width,
                            label='Accuracy', color=CREAM)
        f1_bars = axes.bar([p + width / 2 for p in positions], f1_scores, width,
                           label='Weighted F1', color=AQUA)

        axes.set_xticks(positions)
        axes.set_xticklabels(['Logistic Regression', 'Decision Tree'], color=CREAM, fontsize=11)
        axes.set_ylim(0, 1.12)
        axes.set_ylabel('Score', color=CREAM)
        axes.set_title('Model performance', color=CREAM, fontsize=13, pad=14)
        axes.tick_params(colors=GREEN)
        axes.grid(axis='y', color=BLUE, alpha=0.35, linewidth=0.7)
        axes.set_axisbelow(True)
        for spine in axes.spines.values():
            spine.set_color(BLUE)

        axes.legend(facecolor=DEEP, edgecolor=BLUE, labelcolor=CREAM)

        for bars in (acc_bars, f1_bars):
            for bar in bars:
                axes.annotate("{:.2f}".format(bar.get_height()),
                              xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                              xytext=(0, 4), textcoords='offset points',
                              ha='center', color=WHITE, fontweight='bold')

        figure.tight_layout()
        self._embed_figure(self.comparison_tab, figure)

    def _draw_tree_chart(self):
        for child in self.tree_tab.winfo_children():
            child.destroy()

        figure = Figure(figsize=(11, 5.2), facecolor=WHITE)
        # plot_tree measures text, so the figure needs a real renderer before we draw on it
        canvas = FigureCanvasTkAgg(figure, master=self.tree_tab)
        axes = figure.add_subplot(111)
        plot_tree(self.dt_model, max_depth=2, ax=axes,
                  feature_names=list(self.vectorizer.get_feature_names_out()),
                  class_names=CATEGORY_NAMES, filled=True, rounded=True, fontsize=7)
        axes.set_title("Decision tree - root and first two levels", fontsize=12)
        figure.tight_layout()
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

    # ------------------------------------------------------------- prediction

    def classify_headline(self):
        if self.vectorizer is None:
            return

        headline = self.headline_input.get('1.0', tk.END).strip()
        if not headline:
            self._set_status("Type a headline first.", CREAM)
            return

        # Same cleaning the training text went through
        features = self.vectorizer.transform([headline.lower()])

        for key, model in (('lr', self.lr_model), ('dt', self.dt_model)):
            probabilities = model.predict_proba(features)[0]
            predicted = int(probabilities.argmax())
            panel = self.prediction_panels[key]

            panel['verdict'].config(text=CATEGORY_NAMES[predicted], fg=CATEGORY_COLORS[predicted])
            panel['confidence'].config(text="{:.0%} confident".format(probabilities[predicted]))

            for index, (meter, percent) in panel['bars'].items():
                share = float(probabilities[index])
                percent.config(text="{:.0%}".format(share))
                self._paint_meter(meter, share,
                                  CATEGORY_COLORS[index] if index == predicted else BLUE)

        self._set_status("Classified.", LIME)

    # ------------------------------------------------------------------- misc

    def save_charts(self):
        self._set_status("Rendering the full-resolution PNGs...", CREAM)
        self.root.update_idletasks()
        m = self.metrics
        generate_performance_visuals(
            trained_decision_tree=self.dt_model,
            vectorizer_tool=self.vectorizer,
            logistic_accuracy=m['lr_acc'],
            decision_tree_accuracy=m['dt_acc'],
            logistic_f1_score=m['lr_f1'],
            decision_tree_f1_score=m['dt_f1'],
        )
        self._set_status("Saved decision_tree_graph.png and performance_comparison_chart.png.", LIME)

    def _set_status(self, message, colour=LIME):
        self.status_label.config(text=message, fg=colour)


if __name__ == '__main__':
    window = tk.Tk()
    NewsClassifierApp(window)
    window.mainloop()
