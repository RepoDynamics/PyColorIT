import json as _json
import pkgdata as _pkgdata


_data_dir_path = _pkgdata.get_package_path_from_caller(top_level=True) / "_data"


def css_named_colors() -> dict:
    """
    Get a dictionary of named CSS colors.

    Returns
    -------
    dict
        A dictionary of named CSS colors.

    References
    ----------
    - https://github.com/bahamas10/css-color-names/blob/master/css-color-names.json
    """
    filepath = _data_dir_path / "css-named-colors.json"
    return _json.loads(filepath.read_bytes())
