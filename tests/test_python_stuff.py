from swiss_cheese import MNARrs

def test_name():
  m = MNARrs(0.5)
  assert str(m) == "MNAR[0.5]"
