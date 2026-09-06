import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--real-weights", action="store", default=None,
        help="directory containing the 256, 512 and secondary checkpoints",
    )


@pytest.fixture
def real_weights(request):
    path = request.config.getoption("--real-weights")
    if not path:
        pytest.skip("real model weights were not requested; pass --real-weights")
    return path
