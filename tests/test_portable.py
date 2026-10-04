import csv
import json
from pathlib import Path
import tempfile
import unittest

from portable import check_data, relocated_path, repair_artifact_paths


class PortableTests(unittest.TestCase):
    def test_rebase_windows_mac_and_relative_paths_without_touching_metrics(self):
        with tempfile.TemporaryDirectory(prefix='relocated project ') as directory:
            root = Path(directory).resolve()
            image = root / 'dataset' / 'aptos2019' / 'train_images' / 'example.png'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'image')
            logs = root / 'artifacts' / 'logs'
            logs.mkdir(parents=True)
            model = root / 'artifacts' / 'checkpoints' / 'model.pt'
            model.parent.mkdir()
            model.write_bytes(b'unchanged model')
            record = logs / 'latest.json'
            record.write_text(json.dumps({'checkpoint_path': r'C:\old project\artifacts\checkpoints\model.pt', 'accuracy': .82}))
            manifest = logs / 'manifest.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['image_path', 'class_id'])
                writer.writerow(['/Users/teammate/project/dataset/aptos2019/train_images/example.png', 2])
            self.assertEqual(repair_artifact_paths(root), 2)
            self.assertEqual(repair_artifact_paths(root), 0)
            self.assertEqual(json.loads(record.read_text())['checkpoint_path'], str(model))
            self.assertEqual(json.loads(record.read_text())['accuracy'], .82)
            with manifest.open(newline='') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['image_path'], str(image))
            self.assertEqual(row['class_id'], '2')
            self.assertEqual(model.read_bytes(), b'unchanged model')
            self.assertEqual(relocated_path('dataset/aptos2019/train_images/example.png', root), str(image))
            self.assertEqual(relocated_path('/outside/file.png', root), '/outside/file.png')

    def test_data_check_detects_missing_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data = root/'dataset'/'aptos2019'
            data.mkdir(parents=True)
            for filename, folder in [('train_1.csv','train_images'), ('valid.csv','val_images'), ('test.csv','test_images')]:
                (data/filename).write_text('id_code,diagnosis\nexample,0\n')
                (data/folder).mkdir()
                (data/folder/'example.png').write_bytes(b'image')
            self.assertEqual(check_data(root), (3, []))
            (data/'test_images'/'example.png').unlink()
            count, missing = check_data(root)
            self.assertEqual(count, 3)
            self.assertEqual(len(missing), 1)


if __name__ == '__main__':
    unittest.main()
