"""
Fix Julia Manifest.toml by adding stub entries for weakdeps that are
referenced but missing. Julia 1.11 validates that all weakdep UUIDs
have a corresponding entry in the manifest.
"""
import re
import sys
import os


def fix_manifest(manifest_path):
    content = open(manifest_path).read()

    # Match indented weakdeps blocks (4-space indented lines)
    weakdep_refs = re.findall(
        r'\[deps\.\w+\.weakdeps\]\n((?:[ \t]+\w+ = "[a-f0-9\-]+"\n)+)',
        content
    )

    stubs = {}
    for block in weakdep_refs:
        for m in re.finditer(r'(\w+) = "([a-f0-9\-]+)"', block):
            name, uuid = m.group(1), m.group(2)
            if f'[[deps.{name}]]' not in content:
                stubs[name] = uuid

    if not stubs:
        print("Manifest OK — no missing weakdep entries.")
        return

    print(f"Adding {len(stubs)} missing weakdep stub(s): {', '.join(sorted(stubs))}")

    # Insert stubs before the first [[deps.X]] whose name sorts after each stub
    lines = content.split('\n')
    for name, uuid in sorted(stubs.items()):
        stub = f'[[deps.{name}]]\nuuid = "{uuid}"'
        inserted = False
        for i, line in enumerate(lines):
            m = re.match(r'^\[\[deps\.(\w+)\]\]$', line)
            if m and m.group(1) > name:
                lines.insert(i, stub + '\n')
                inserted = True
                break
        if not inserted:
            lines.append(stub)

    open(manifest_path, 'w').write('\n'.join(lines))
    print("Manifest fixed.")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        julia_env = os.environ.get('PYTHON_JULIAPKG_PROJECT', '')
        path = os.path.join(julia_env, 'Manifest.toml')

    if not os.path.exists(path):
        print(f"No manifest found at {path}, skipping.")
        sys.exit(0)

    fix_manifest(path)
