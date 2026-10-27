import numpy as np

from resonance.sam import member_from_box, parse_line, parse_output

# Records in the documented wire grammar (payloads are dummy base85 text that may contain ~ ! | >)
LINE0 = ("<0f>0<|box;x1=118;y1=110;x2=124;y2=1107;w=720;h=1280;c=0.94|>"
         "<|mask;x=0;y=0;c=0.94;data=160,90,~abc|>,3<|box;x1=238;y1=110;x2=244;y2=1084;w=720;h=1280;c=0.91|>"
         "<|mask;x=0;y=0;c=0.91;data=160,90,!z~!|x>|>")
LINE5 = "<5f>3<|box;x1=239;y1=111;x2=245;y2=1085;w=720;h=1280|><|mask;x=0;y=0;data=160,90,~qq|>"


def test_parse_two_records_with_odd_payloads():
    d = parse_line(LINE0)
    assert [x.obj for x in d] == [0, 3]
    assert d[0].box == (118, 110, 124, 1107) and d[0].size == (720, 1280)
    assert d[0].conf == 0.94 and d[1].mask_enc == "!" and d[1].mask_hw == (160, 90)


def test_parse_output_skips_frames_and_keeps_ids():
    d = parse_output(LINE0 + "\n" + LINE5 + "\n")
    assert [(x.frame, x.obj) for x in d] == [(0, 0), (0, 3), (5, 3)]
    assert d[2].conf is None


def test_member_from_thin_box_is_vertical():
    det = parse_line(LINE0)[0]
    m = member_from_box("rope1", det, np.zeros((1280, 720), np.float32))
    assert m.p0[0] == m.p1[0] == 121.0 and m.p0[1] == 110 and m.p1[1] == 1107


def test_member_from_wide_box_picks_the_visible_diagonal():
    g = np.full((400, 400), 200.0, np.float32)
    for i in range(300):  # dark line from (50, 350) up to (350, 50)
        g[350 - i, 50 + i] = 30.0
    det = parse_line("<0f>1<|box;x1=50;y1=50;x2=350;y2=350;w=400;h=400|><|mask;x=0;y=0;data=10,10,~a|>")[0]
    m = member_from_box("cable", det, g)
    assert {tuple(map(int, m.p0)), tuple(map(int, m.p1))} == {(50, 350), (350, 50)}
