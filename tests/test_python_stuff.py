from swiss_cheese import MNARrs, MAR


def test_name_mnar():
    m = MNARrs(0.5)
    assert str(m) == "MNAR[0.5]"


def test_name_mar():
    m = MAR()
    assert str(m) == "MAR"
