#!/usr/bin/env bash
# Build the anonymised code supplement for double-blind review: the committed tree at HEAD,
# without git metadata, with the author and the manuscript repository removed, and the package
# renamed to the anonymous name the manuscript uses (\pkgname in main.tex).
#
#     scripts/anonymous_supplement.sh [-n NAME] [-r RESULTS_DIR] [out.zip]
#
#   -n NAME          package name in the supplement (default: sparsete)
#   -r RESULTS_DIR   folder of result CSVs to ship as results/ (see results/README.md there)
#   out.zip          default: <NAME>-supplement.zip
#
# Citations of Hernandez & Samengo (2019) stay: the paper cites that work in the third person.
set -euo pipefail
name=sparsete
results=""
while getopts "n:r:" opt; do
    case $opt in
        n) name=$OPTARG ;;
        r) results=$(realpath "$OPTARG") ;;
        *) exit 2 ;;
    esac
done
shift $((OPTIND - 1))
out=$(realpath -m "${1:-$name-supplement.zip}")
root=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -C "$root" archive --format=tar --prefix="$name/" HEAD | tar -x -C "$tmp"
top="$tmp/$name"
rm -f "$top/scripts/anonymous_supplement.sh"
sed -i 's/^authors = .*/authors = [{ name = "Anonymous" }]/' "$top/pyproject.toml"
sed -i 's/^Copyright (c) \(.*\), .*/Copyright (c) \1, Anonymous Authors/' "$top/LICENSE"
sed -i 's| (damianghl/mitarget-paper)||' "$top/experiments/paper/README.md"
# rename the package: directory, imports, docs
mv "$top/src/mitarget" "$top/src/$name"
grep -rIl "mitarget" "$top" | xargs -r sed -i "s/mitarget/$name/g"
if [ -n "$results" ]; then
    cp -r "$results" "$top/results"
fi
if grep -rIn -i -E "damian|damianghl|ib\.edu|balseiro|mitarget|github\.com/" "$top"; then
    echo "identifying strings left in the supplement (above); fix before submitting" >&2
    exit 1
fi
rm -f "$out"
(cd "$tmp" && zip -qr "$out" "$name")
echo "wrote $out ($(du -h "$out" | cut -f1))"
