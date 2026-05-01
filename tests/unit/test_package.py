def test_atha_version():
    import atha
    assert atha.__version__ == "0.1.0"

def test_atha_subpackages_importable():
    import atha.core
    import atha.solver
    import atha.thermo
    import atha.components
    import atha.jannaf
