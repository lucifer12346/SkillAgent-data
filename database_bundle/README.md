# Complete SQLite Database Bundle

The complete portable data bundle is stored with Git LFS as numbered parts in
`database_bundle/`.

## Reassemble

Linux/macOS:

```bash
cat database_bundle/SkillAgent_data_with_db_20260804_max.zip.part_* \
  > SkillAgent_data_with_db_20260804_max.zip
sha256sum SkillAgent_data_with_db_20260804_max.zip
```

Expected SHA256:

```text
a9af9e554505ad67d5e6bf2712f31b94525cafa2bdd1015f620ecad7fcac8881
```

Windows PowerShell:

```powershell
$parts = Get-ChildItem database_bundle\*.part_* | Sort-Object Name
$output = [IO.File]::Create('SkillAgent_data_with_db_20260804_max.zip')
try {
  foreach ($part in $parts) {
    $input = [IO.File]::OpenRead($part.FullName)
    try { $input.CopyTo($output) } finally { $input.Dispose() }
  }
} finally { $output.Dispose() }
Get-FileHash SkillAgent_data_with_db_20260804_max.zip -Algorithm SHA256
```

## Contents

- Filtered prepared Train, Validation, BIRD, EHRSQL and Spider2.0 JSON
- 303 unique SQLite database files
- Spider2.0 evaluation metadata and gold execution results
- `DB_MANIFEST.json`
- Path relocation utility and usage documentation

Uncompressed SQLite size: `35,792,005,120` bytes.

Compressed ZIP size: `9,937,483,908` bytes.

Git LFS is required when cloning this repository:

```bash
git lfs install
git clone git@github.com:lucifer12346/SkillAgent-data.git
```
