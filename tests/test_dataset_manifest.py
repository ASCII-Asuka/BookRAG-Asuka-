import hashlib
import json
import tempfile
import unittest
from pathlib import Path


def write_fixture(root: Path):
    rows = [
        {
            "doc_uuid": "q-1",
            "question_id": "q-1",
            "hotpotqa_question_id": "q-1",
            "question": "First?",
            "answer": "first",
        },
        {
            "doc_uuid": "q-2",
            "question_id": "q-2",
            "hotpotqa_question_id": "q-2",
            "question": "Second?",
            "answer": "second",
        },
    ]
    dataset_path = root / "dataset.json"
    dataset_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest_path = root / "dataset.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "question_count": 2,
                "selected_question_ids": ["q-1", "q-2"],
                "unified_sha256": hashlib.sha256(
                    dataset_path.read_bytes()
                ).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    config_path = root / "dataset.yaml"
    config_path.write_text(
        f'dataset_path: "{dataset_path.as_posix()}"\n'
        f'working_dir: "{(root / "work").as_posix()}"\n'
        "dataset_name: hotpotqa\n"
        f'manifest_path: "{manifest_path.as_posix()}"\n',
        encoding="utf-8",
    )
    return rows, dataset_path, manifest_path, config_path


class DatasetManifestTests(unittest.TestCase):
    def test_dataset_config_uses_qasper_question_ids_for_manifest(self):
        from Core.configs.dataset_config import load_dataset_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, dataset_path, manifest_path, config_path = write_fixture(root)
            rows = json.loads(dataset_path.read_text(encoding="utf-8"))
            for index, row in enumerate(rows, 1):
                row.pop("hotpotqa_question_id", None)
                row.pop("question_id", None)
                row["doc_uuid"] = "shared-paper"
                row["qasper_question_id"] = f"qasper-{index}"
            dataset_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["selected_question_ids"] = ["qasper-1", "qasper-2"]
            manifest["unified_sha256"] = hashlib.sha256(
                dataset_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            cfg = load_dataset_config(str(config_path))

        self.assertEqual(cfg.dataset_name, "hotpotqa")

    def test_dataset_config_validates_ordered_ids_and_sha(self):
        from Core.configs.dataset_config import load_dataset_config

        with tempfile.TemporaryDirectory() as tmp:
            _, _, manifest_path, config_path = write_fixture(Path(tmp))

            cfg = load_dataset_config(str(config_path))

        self.assertEqual(
            Path(cfg.manifest_path).resolve(),
            manifest_path.resolve(),
        )

    def test_dataset_config_rejects_tampered_dataset(self):
        from Core.configs.dataset_config import load_dataset_config

        with tempfile.TemporaryDirectory() as tmp:
            _, dataset_path, _, config_path = write_fixture(Path(tmp))
            dataset_path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dataset SHA256 mismatch"):
                load_dataset_config(str(config_path))

    def test_dataset_config_rejects_reordered_question_ids(self):
        from Core.configs.dataset_config import load_dataset_config

        with tempfile.TemporaryDirectory() as tmp:
            rows, dataset_path, manifest_path, config_path = write_fixture(
                Path(tmp)
            )
            dataset_path.write_text(
                json.dumps(list(reversed(rows)), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["unified_sha256"] = hashlib.sha256(
                dataset_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "selected question ID order mismatch",
            ):
                load_dataset_config(str(config_path))


if __name__ == "__main__":
    unittest.main()
