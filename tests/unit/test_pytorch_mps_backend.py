import pytest

from orchard_inference.backends.pytorch_mps import PyTorchMPSBackend
from orchard_inference.models import HealthStatus


class _MPSCapability:
    def __init__(self, built: bool, available: bool) -> None:
        self._built = built
        self._available = available

    def is_built(self) -> bool:
        return self._built

    def is_available(self) -> bool:
        return self._available


class _Backends:
    def __init__(self, mps: _MPSCapability) -> None:
        self.mps = mps


class _Torch:
    def __init__(self, built: bool, available: bool) -> None:
        self.backends = _Backends(_MPSCapability(built, available))


def test_capability_check_rejects_non_mps_build() -> None:
    with pytest.raises(RuntimeError, match="not built"):
        PyTorchMPSBackend._require_mps(_Torch(False, False))


def test_capability_check_rejects_unavailable_device() -> None:
    with pytest.raises(RuntimeError, match="unavailable"):
        PyTorchMPSBackend._require_mps(_Torch(True, False))


def test_unloaded_backend_metadata_is_explicit() -> None:
    backend = PyTorchMPSBackend("org/model", "architecture", "fp16", "float16")
    assert backend.health().status is HealthStatus.UNLOADED
    info = backend.model_info()
    assert info.backend == "pytorch_mps"
    assert info.architecture == "architecture"
    assert info.quantization == "fp16"
