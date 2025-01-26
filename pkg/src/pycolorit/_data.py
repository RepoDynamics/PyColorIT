from pycolorit import data as _data


_css_named_colors = None


def css_named_colors():
    global _css_named_colors
    if _css_named_colors is None:
        _css_named_colors = _data.css_named_colors()
    return _css_named_colors
