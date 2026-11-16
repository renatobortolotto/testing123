from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import kraken

package_dir = Path(kraken.__file__).resolve().parent
output_path = Path("/tmp/kraken_codigo.zip")

with ZipFile(output_path, "w", ZIP_DEFLATED) as archive:
    for file_path in package_dir.rglob("*"):
        if (
            file_path.is_file()
            and file_path.suffix in {".py", ".md", ".rst"}
            and "__pycache__" not in file_path.parts
        ):
            archive.write(
                file_path,
                arcname=file_path.relative_to(package_dir.parent),
            )

print(f"Arquivo gerado: {output_path}")