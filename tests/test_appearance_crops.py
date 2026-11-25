"""appearance: body crops are saved only where the product says (`crops_dir`), independent of customer faces."""
import pytest

from perception.engine.modules import ModuleConfigError
from perception.modules.appearance import Settings


def test_no_crops_dir_means_no_body_crops():
    assert Settings().crops_dir == ""


def test_a_product_names_the_folder_body_crops_go_to():
    assert Settings(crops_dir="/data/crops").crops_dir == "/data/crops"


def test_settings_refuse_an_unknown_staff_model_or_reid_spec():
    for bad in ({"staff_model": "dinov2"}, {"reid_spec": "nope"}, {"interval_s": 0}, {"body_interval_s": -1}):
        with pytest.raises(ModuleConfigError):
            Settings(**bad)
