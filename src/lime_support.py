"""LIME explanations in the same preprocessed image space as Grad-CAM/SHAP."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def explain_lime_image(model, image, transform, device, class_id, *, seed=1988,
                       num_samples=256, batch_size=4, n_segments=50):
    """Return signed superpixel weights and the local surrogate's weighted R².

    `image` is already fundus-preprocessed. Perturbations are only resized and
    normalized by `transform`; applying fundus preprocessing again would change
    the function being explained. Probabilities are uncalibrated, matching the
    existing attribution/faithfulness functions.
    """
    from lime.lime_image import LimeImageExplainer
    from skimage.segmentation import slic

    if num_samples < 2 or batch_size < 1 or n_segments < 2:
        raise ValueError('LIME needs num_samples>=2, batch_size>=1, n_segments>=2')
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)

    def predict(images):
        chunks = []
        for offset in range(0, len(images), batch_size):
            batch = torch.stack([
                transform(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
                for arr in images[offset:offset + batch_size]
            ]).to(device)
            with torch.inference_mode():
                chunks.append(torch.softmax(model(batch), dim=1).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    explanation = LimeImageExplainer(random_state=seed).explain_instance(
        rgb, predict, labels=(int(class_id),), top_labels=None,
        hide_color=None, num_samples=int(num_samples), batch_size=int(batch_size),
        segmentation_fn=lambda arr: slic(arr, n_segments=int(n_segments),
                                         compactness=10, start_label=0, channel_axis=-1),
        random_seed=seed,
    )
    signed = np.zeros(explanation.segments.shape, dtype=np.float32)
    for segment, weight in explanation.local_exp[int(class_id)]:
        signed[explanation.segments == segment] = weight
    return signed, float(explanation.score), explanation.segments


def run_lime_targets(api, conf, targets, seed, split, checkpoint, run_id):
    """Produce per-target panels, numerical audits and explicit failure status."""
    import pandas as pd
    import matplotlib.pyplot as plt

    settings = conf.get('xai', {})
    columns = ['sample_id', 'target_class', 'pred_class', 'confidence',
               'artifact_path', 'border_ratio', 'retina_ratio', 'faithfulness_delta',
               'lime_pass', 'faith_delta_k10', 'faith_delta_k20', 'faith_delta_k30',
               'aopc_delta', 'aopc_pass', 'local_r2', 'num_samples', 'n_segments']
    rows, errors = [], {}
    enabled = bool(settings.get('lime_enabled', True))
    status = 'disabled' if not enabled else 'no_targets'
    device = api._resolve_xai_device(conf)
    count = int(settings.get('lime_num_samples', 256))
    batch = int(settings.get('lime_batch_size', 4))
    segments = int(settings.get('lime_n_segments', 50))
    directory = Path(conf['paths']['figures_dir']) / 'lime'
    directory.mkdir(parents=True, exist_ok=True)
    if enabled and len(targets):
        try:
            # Import before loading the model so missing packages fail cheaply.
            from lime.lime_image import LimeImageExplainer  # noqa: F401
            model = api._load_model(conf, seed, device, checkpoint=checkpoint)
            size = api._model_image_size(conf)
            prep = conf.get('preprocessing', {})
            transform = api._build_transform(size, False, {}, prep,
                                             skip_resize=bool(prep.get('enabled', True)))
            border_max, faith_min = api._xai_pass_rule_thresholds(conf)
            ks = api._parse_faithfulness_k_list(settings.get('faithfulness_k_list', [.1, .2, .3]))
            for index, row in targets.reset_index(drop=True).iterrows():
                sid = str(row['sample_id'])
                figure = None
                try:
                    print(f'[XAI][LIME] {index + 1}/{len(targets)}: {sid}', flush=True)
                    image, tensor = api._load_image_for_inference(row['image_path'], size, prep)
                    cls = int(row['pred_class'])
                    signed, r2, segmentation = explain_lime_image(
                        model, image, transform, device, cls, seed=seed + index,
                        num_samples=count, batch_size=batch, n_segments=segments)
                    # Positive evidence is used for the same audit as SHAP positive mode.
                    positive = np.maximum(signed, 0)
                    if not np.any(positive > 0):
                        raise ValueError('No positive LIME evidence; attribution audit undefined')
                    positive /= positive.max()
                    tensor = tensor.to(device)
                    with torch.no_grad():
                        deltas, aopc = api._faithfulness_multi_k(model, tensor, positive, cls, ks, seed)
                        faith = deltas.get('faith_delta_k20')
                        if faith is None:
                            faith = api._faithfulness_delta(model, tensor, positive, cls, .2, seed)
                    ratios = api._attribution_mass_ratios(positive)
                    path = directory / f'{sid}_cls{cls}.png'
                    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
                    axes[0].imshow(image)
                    axes[0].set_title('Preprocessed image')
                    axes[1].imshow(image)
                    bound = max(float(np.abs(signed).max()), 1e-8)
                    overlay = axes[1].imshow(signed, cmap='RdBu_r', vmin=-bound,
                                            vmax=bound, alpha=.65)
                    label = conf['data']['label_order'][cls]
                    axes[1].set_title(f'LIME: {label}\nLocal surrogate R²={r2:.3f}')
                    for ax in axes:
                        ax.axis('off')
                    figure.colorbar(overlay, ax=axes.tolist(), fraction=.03,
                                    label='Superpixel weight (red supports, blue opposes)')
                    figure.savefig(path, dpi=min(int(settings.get('figure_dpi', 180)), 240),
                                   bbox_inches='tight')
                    np.savez_compressed(path.with_suffix('.npz'), signed_weights=signed,
                                        segments=segmentation)
                    rows.append({
                        'sample_id': sid, 'target_class': int(row['true_class']),
                        'pred_class': cls, 'confidence': float(row['confidence']),
                        'artifact_path': str(path), **ratios, **deltas,
                        'faith_delta_k20': faith, 'faithfulness_delta': faith,
                        'lime_pass': api._xai_pass_flag(ratios['border_ratio'], faith, border_max, faith_min),
                        'aopc_delta': aopc, 'aopc_pass': int(aopc > 0.0),
                        'local_r2': r2, 'num_samples': count,
                        'n_segments': len(np.unique(segmentation)),
                    })
                except Exception as exc:
                    errors[sid] = str(exc)
                finally:
                    if figure is not None:
                        plt.close(figure)
            status = 'partial' if errors and rows else 'failed' if errors else 'success'
        except Exception as exc:
            status = 'failed'
            errors['setup'] = str(exc)
    frame = pd.DataFrame(rows, columns=list(dict.fromkeys(columns + [k for r in rows for k in r])))
    table = Path(conf['paths']['tables_dir']) / f'rq3_lime_seed{seed}_{split}.csv'
    frame.to_csv(table, index=False)
    status_path = Path(conf['paths']['logs_dir']) / f'{run_id}_{split}_lime_status.json'
    api._save_json(status_path, dict(status=status, errors=errors, rows=len(frame),
                                   target_rows=len(targets), device=str(device),
                                   num_samples=count, batch_size=batch, seed=seed,
                                   score_mode='positive', run_id=run_id))
    if errors:
        print(f'[XAI][LIME] {status}: {errors}')
    return frame, str(table), str(status_path), status
