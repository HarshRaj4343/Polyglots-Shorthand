# `paper/` — LaTeX source of the technical report

`whitepaper.tex` is the typeset report covering v1.0 through v1.2: the feature
design, the hand-written gradients, the v1.1 attention branch, and the v1.2
teacher/student distillation, with results tables and error analysis.

It is **self-contained**. All figures are drawn inline with TikZ/PGF rather than
included as images, so there is no asset folder to keep in sync and the diagrams
restyle with the document.

Build with any modern TeX distribution (it uses `cleveref`, so run twice for
cross-references to resolve):

```sh
cd paper
pdflatex whitepaper.tex && pdflatex whitepaper.tex
```

Related, and intentionally separate:

- `../WHITEPAPER.md` — the editable Markdown draft of the same material.
- `../Q4_Solution.pdf` — the built eight-page PDF shipped with the submission.
- `../v12/SECTION12_DRAFT.md` — the v1.2 section drafted before it was folded in here.
