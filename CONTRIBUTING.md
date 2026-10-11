# Contributing

Scientific changes must identify the affected manuscript section and explain their numerical effect. Keep fixed protocol settings in `config/study.json`. Do not modify patient selection using connectivity values or prediction scores. Changes to estimator definitions require independent numerical checks.

Before proposing a change:

```bash
python -m pytest -q
python -m eegref --help
```

Never add EDF recordings, private credentials, absolute machine paths, generated study artifacts, or model caches to a source commit. Use a new output directory after changing configuration, dependencies, or source code. Reviewers may report issues without running the full study. Coordinate redistribution or derivative work with the rights holders; see LICENSE.
