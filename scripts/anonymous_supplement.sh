#!/usr/bin/env bash
# Build the anonymised code supplement for double-blind review: the committed tree at HEAD,
# without git metadata, with the author and the manuscript repository removed.
#
#     scripts/anonymous_supplement.sh [out.zip]     (default: mitarget-supplement.zip)
#
# Citations of Hernandez & Samengo (2019) stay: the paper cites that work in the third person.
set -euo pipefail
out=$(realpath -m "${1:-mitarget-supplement.zip}")
root=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git -C "$root" archive --format=tar --prefix=mitarget/ HEAD | tar -x -C "$tmp"
rm -rf "$tmp/mitarget/scripts/anonymous_supplement.sh"
sed -i 's/^authors = .*/authors = [{ name = "Anonymous" }]/' "$tmp/mitarget/pyproject.toml"
sed -i 's| (damianghl/mitarget-paper)||' "$tmp/mitarget/experiments/paper/README.md"
if grep -rIn -i -E "damian|damianghl|ib\.edu|balseiro|github\.com/" "$tmp/mitarget"; then
    echo "identifying strings left in the supplement (above); fix before submitting" >&2
    exit 1
fi
rm -f "$out"
(cd "$tmp" && zip -qr "$out" mitarget)
echo "wrote $out"
