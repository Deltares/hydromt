from pathlib import Path

from hydromt.typing.deferred_file_close import _MAX_CLOSE_ATTEMPTS, DeferredFileClose


def test_close_successful_move(monkeypatch):
    original = Path("original.txt")
    temp = Path("temp.txt")
    move_called = []

    def fake_move(src, dst):
        move_called.append((src, dst))

    monkeypatch.setattr("shutil.move", fake_move)
    dfc = DeferredFileClose(original_path=original, temp_path=temp)
    dfc.close()
    assert move_called == [(temp, original)]


def test_close_permission_error_then_success(monkeypatch, caplog):
    original = Path("original.txt")
    temp = Path("temp.txt")
    call_count = 0

    def fake_move(_src, _dst):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            raise PermissionError("Denied")

    monkeypatch.setattr("shutil.move", fake_move)
    dfc = DeferredFileClose(original_path=original, temp_path=temp)

    dfc.close()

    assert (
        f"Could not write to destination file {original} because the following error was raised: Denied"
        in caplog.text
    )


def test_close_file_not_found(monkeypatch, caplog):
    original = Path("original.txt")
    temp = Path("temp.txt")

    def fake_move(_src, _dst):
        raise FileNotFoundError

    monkeypatch.setattr("shutil.move", fake_move)
    dfc = DeferredFileClose(original_path=original, temp_path=temp)

    dfc.close()

    assert (
        f"Could not find temporary file {temp}. It was likely already deleted by another component that updates the same dataset."
        in caplog.text
    )


def test_close_max_attempts_exceeded(monkeypatch, caplog):
    original = Path("original.txt")
    temp = Path("temp.txt")

    def fake_move(_src, _dst):
        raise PermissionError("Denied")

    monkeypatch.setattr("shutil.move", fake_move)
    dfc = DeferredFileClose(original_path=original, temp_path=temp)

    dfc.close()

    assert (
        f"Max write attempts to file {original} exceeded. Skipping... "
        f"Instead, data was written to a temporary file: {temp}." in caplog.text
    )
    assert dfc._close_attempts == _MAX_CLOSE_ATTEMPTS
