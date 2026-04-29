#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

configuration="${CONFIGURATION:-debug}"

swift build -c "$configuration"
bin_path="$(swift build -c "$configuration" --show-bin-path)"
app_path="$bin_path/Cantis.app"
resource_bundle="$bin_path/Cantis_Cantis.bundle"

rm -rf "$app_path"
mkdir -p "$app_path/Contents/MacOS" "$app_path/Contents/Resources"

cp "$bin_path/Cantis" "$app_path/Contents/MacOS/Cantis"
cp "Cantis/Info.plist" "$app_path/Contents/Info.plist"
perl -0pi -e 's/\$\(EXECUTABLE_NAME\)/Cantis/g' "$app_path/Contents/Info.plist"

cp "Cantis/Resources/AppIcon.icns" "$app_path/Contents/Resources/AppIcon.icns"

if [[ -d "$resource_bundle" ]]; then
    cp -R "$resource_bundle" "$app_path/Cantis_Cantis.bundle"
    cp -R "$resource_bundle" "$app_path/Contents/Resources/Cantis_Cantis.bundle"
else
    echo "Missing resource bundle: $resource_bundle" >&2
    exit 1
fi

plutil -lint "$app_path/Contents/Info.plist" >/dev/null

echo "$app_path"
