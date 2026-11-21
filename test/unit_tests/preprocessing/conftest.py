import pytest
from joblib.externals.loky import get_reusable_executor


@pytest.fixture(scope="module", autouse=True)
def _stop_joblib_workers():
    """Stop the loky workers that this module's ``n_jobs != 1`` tests started.

    Each worker imports torch and braindecode to unpickle the datasets (about
    0.5 GB), and joblib keeps idle workers for 300 s, while the xdist worker
    goes on with other modules.
    """
    yield
    get_reusable_executor().shutdown(wait=True)
