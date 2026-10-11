"""Demonstrate dimensions on generated signals, with no EDF or classifier."""
import json
from pathlib import Path
import numpy as np
from eegref.features import connectivity, epochs, graph_features, spectral


def main():
    config = json.loads((Path(__file__).resolve().parents[1]/'config/study.json').read_text())
    rng = np.random.default_rng(20260930)
    time = np.arange(256*300)/256
    data = (rng.normal(size=(18, len(time))) + .2*np.sin(2*np.pi*10*time))*1e-6
    segmented = epochs(data, config)
    # Limit the connectivity example to one epoch to keep this demo lightweight.
    matrices = connectivity(segmented[:1], config)
    summary = dict(source='synthetic signals only', full_window_epochs=len(segmented),
                   spectral_features=len(spectral(segmented, 256)),
                   graph_features_per_method={method: graph_features(matrix).shape[-1]
                                              for method, matrix in matrices.items()},
                   connectivity_epochs_demonstrated=1, classifier_fitted=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
