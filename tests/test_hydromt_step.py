from hydromt import hydromt_step


def test_hydromt_step_adds_ishydromtstep_attribute():
    @hydromt_step
    def foo():
        pass

    assert foo.__ishydromtstep__
