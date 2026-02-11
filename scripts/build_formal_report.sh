#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/docs/report"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found. Install TeX first:"
  echo "  sudo apt-get update"
  echo "  sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra texlive-pictures"
  exit 1
fi

if ! kpsewhich latex.ltx >/dev/null 2>&1; then
  echo "LaTeX core file latex.ltx not found."
  echo "On Arch/EndeavourOS install:"
  echo "  sudo pacman -S texlive-latex texlive-latexextra texlive-fontsrecommended"
  echo "On Ubuntu/Debian install:"
  echo "  sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended"
  exit 1
fi

cd "$REPORT_DIR"
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Generated: $REPORT_DIR/main.pdf"
