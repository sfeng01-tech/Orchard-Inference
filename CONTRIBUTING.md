# Contributing

Thanks for taking a look at Orchard. This project is intentionally small and
systems-focused, so changes should preserve clear behavior and reproducible
measurements.

## Development setup

```shell
uv sync --all-groups
make check
```

Optional PyTorch MPS comparison support:

```shell
uv sync --all-groups --extra mps
```

## Expectations

- Keep backend selection explicit; do not add silent CPU fallback.
- Add tests for scheduler, lifecycle, cache, API, or backend behavior when a
  change touches those paths.
- Do not publish benchmark claims without raw artifacts and environment details.
- Keep metrics labels bounded and avoid request IDs or prompt text in metrics.
- Prefer focused patches over broad rewrites.

## Pull request checklist

- Formatting, linting, type checking, and tests pass.
- Docs are updated for user-visible behavior.
- Failure modes are documented for reliability-sensitive changes.
- Benchmark changes include a real artifact path or clearly state that numbers
  are not included.

