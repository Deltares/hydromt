from hydromt._utils.capture_glob import capture_glob


def test_capture_glob():
    pat = "here-is-some-more-leading-{time}-text-for-{name}-you-{id}.pq"
    example = "here-is-some-more-leading-2024-04-02-text-for-era5-you-0001.pq"

    glob, regex = capture_glob(pat)
    assert glob == "here-is-some-more-leading-*-text-for-*-you-*.pq"
    assert regex.match(example).groups() == ("2024-04-02", "era5", "0001")
