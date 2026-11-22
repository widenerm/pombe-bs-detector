import unittest
import numpy as np
from validation.scar_batch import associate, classify, metrics


class ValidationTests(unittest.TestCase):
    def test_portable_cache_preserves_model_settings_and_precision(self):
        from validation.scar_batch import segmentation_signature
        original = dict(cellpose='4.0.9', torch='2.10.0', model='/mac/cpsam',
                        model_sha256='verified', device='mps', diameter=None,
                        channels=[0, 0], batch_size=1)
        colab = dict(original, model='/root/cpsam', device='cuda:0',
                     torch='2.10.0+cu128', dtype='bfloat16')
        self.assertNotEqual(segmentation_signature(original), segmentation_signature(colab))
        self.assertEqual(segmentation_signature(original, True), segmentation_signature(colab, True))
        for field, value in [('model_sha256', 'different'), ('dtype', 'float32'),
                             ('batch_size', 8), ('diameter', 30),
                             ('cellpose', '4.1.0'), ('torch', '2.11.0')]:
            with self.subTest(field=field):
                self.assertNotEqual(segmentation_signature(original, True),
                                    segmentation_signature(dict(colab, **{field: value}), True))

    def test_missing_predictions_count_against_target(self):
        rows=[dict(gt_positive=True, detected=True,error_px=3.,oracle_error_px=2.,failure='success'),
              dict(gt_positive=True,detected=False,error_px=None,oracle_error_px=None,failure='no-candidate failure'),
              dict(gt_positive=True,detected=True,error_px=20.,oracle_error_px=2.,failure='wrong-candidate selection')]
        m=metrics(rows)
        self.assertEqual(m['recall_5px'],1/3)
        self.assertEqual(m['precision_5px_annotated'],.5)
        self.assertEqual(m['detection_recall'],2/3)
        self.assertIsNone(m['population_detection_precision'])
        self.assertEqual(m['oracle_within_5'],2)

    def test_failure_taxonomy(self):
        self.assertEqual(classify(None,None,'segmentation failure',False),'segmentation failure')
        self.assertEqual(classify(None,None,has_candidate=False),'no-candidate failure')
        self.assertEqual(classify(5,5),'success')
        self.assertEqual(classify(6,5),'wrong-candidate selection')
        self.assertEqual(classify(6,6),'localization error')

    def test_containment_does_not_choose_best_prediction(self):
        labels=np.zeros((20,20),int); labels[3:15,4:12]=7
        gt={'gt_scar_mid_y_px':'8','gt_scar_mid_x_px':'9'}
        self.assertEqual(associate(labels,{7:object()},gt),(7,'ok'))
        self.assertEqual(associate(labels,{},gt),(7,'segmentation failure'))
        gt['gt_scar_mid_x_px']='19'
        self.assertEqual(associate(labels,{7:object()},gt),(0,'segmentation failure'))


class BatchIntegrationTests(unittest.TestCase):
    def test_four_video_cache_and_original_frame_numbers(self):
        import csv
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        import h5py
        from validation.scar_batch import run_batch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'h5';data.mkdir();cache=root/'cache'
            yy,xx=np.mgrid[:100,:180]
            labels=(((yy-50)/15)**2+((xx-90)/65)**2<1).astype(np.int32)
            rows=[]
            provenance=json.dumps(dict(diameter=None,model_sha256='synthetic-mask-only'))
            for i in range(4):
                video=f'video{i}';image=(labels*100+i).astype(np.uint16)
                with h5py.File(data/f'{video}.h5','w') as f:
                    f['frames']=image[None];f['frame_numbers']=[77]
                folder=cache/video;folder.mkdir(parents=True)
                np.savez_compressed(folder/'0077.npz',labels=labels,
                                    image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                    provenance=provenance)
                rows.append(dict(video_id=video,frame=77,cell_name='A',scorer_initials='test',
                                 gt_scar_mid_x_px=65,gt_scar_mid_y_px=50,
                                 gt_area_px2=int(labels.sum()),notes='synthetic positive'))
            gt=root/'gt.csv'
            with gt.open('w',newline='') as f:
                w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
            summary=run_batch(data,gt,cache,root/'out')
            self.assertEqual(summary['combined']['n'],4)
            self.assertEqual(len(summary['per_video']),4)
            with (root/'out/per_cell.csv').open() as f:predictions=list(csv.DictReader(f))
            self.assertEqual(len(predictions),4)
            self.assertEqual({int(r['frame']) for r in predictions},{77})
            # Mutating image content must invalidate the cache, even when
            # shape, video name and original frame number are unchanged.
            with h5py.File(data/'video0.h5','r+') as f:f['frames'][0,0,0]=999
            with self.assertRaisesRegex(ValueError,'Stale image cache'):
                run_batch(data,gt,cache,root/'stale')


if __name__=='__main__': unittest.main()
