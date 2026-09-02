from io import StringIO

import pandas as pd
import pyais as ais
import pytest

import pytsa
from pytsa.decoder.ais_decoder import MSG12318SLOTS, MSG5SLOTS, StructuralError, _get_decoder


def test_raw_dynamic_message_decodes_as_type_one(dynamic_message: str) -> None:
    assert ais.decode(dynamic_message).msg_type == 1


def test_raw_static_message_decodes_as_type_five(static_messages: tuple[str, str]) -> None:
    assert ais.decode(*static_messages).msg_type == 5


def test_decode_dynamic_buffer_preserves_source_columns_and_adds_message_fields(
    dynamic_csv: StringIO,
) -> None:
    decoded = pytsa.decode_from_file(dynamic_csv, None, save_to_file=False)

    assert len(decoded) == 1
    assert set(decoded.columns) == {
        "timestamp",
        "message_id",
        "raw_message",
        "DECODE_START",
        *MSG12318SLOTS,
    }
    assert decoded.loc[0, "msg_type"] == 1


def test_decode_static_buffer_preserves_source_columns_and_adds_message_fields(
    static_csv: StringIO,
) -> None:
    decoded = pytsa.decode_from_file(static_csv, None, save_to_file=False)

    assert len(decoded) == 1
    assert set(decoded.columns) == {
        "timestamp",
        "message_id",
        "raw_message1",
        "raw_message2",
        "DECODE_START",
        *MSG5SLOTS,
    }
    assert decoded.loc[0, "msg_type"] == 5


def test_decode_drops_malformed_messages_but_retains_valid_rows(dynamic_message: str) -> None:
    source = StringIO(
        "timestamp,message_id,raw_message\n"
        f'2021-08-02T00:00:00.000Z,1,"{dynamic_message}"\n'
        '2021-08-02T00:01:00.000Z,1,"not-an-ais-message"\n'
    )

    decoded = pytsa.decode_from_file(source, None, save_to_file=False)

    assert len(decoded) == 1
    assert decoded.loc[0, "raw_message"] == dynamic_message


@pytest.mark.parametrize(
    ("dataframe", "match"),
    [
        (
            pd.DataFrame({"message_id": [1, 5], "raw_message": ["a", "b"]}),
            "Found not processable combination",
        ),
        (
            pd.DataFrame(
                {
                    "message_id": [1],
                    "raw_message1": ["a"],
                    "raw_message2": ["b"],
                }
            ),
            "Assumed type-5-only dataframe",
        ),
    ],
)
def test_decoder_selection_rejects_invalid_message_groups(
    dataframe: pd.DataFrame, match: str
) -> None:
    with pytest.raises(StructuralError, match=match):
        _get_decoder(dataframe)


def test_decode_from_file_writes_csv(tmp_path, dynamic_message: str) -> None:
    source = tmp_path / "raw.csv"
    destination = tmp_path / "decoded.csv"
    source.write_text(
        "timestamp,message_id,raw_message\n"
        f'2021-08-02T00:00:00.000Z,1,"{dynamic_message}"\n', encoding="utf-8"
    )

    result = pytsa.decode_from_file(source, destination)

    assert result is None
    decoded = pd.read_csv(destination)
    assert decoded.loc[0, "msg_type"] == 1


def test_decode_directory_skips_existing_files_unless_overwrite_requested(
    tmp_path, dynamic_message: str
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    raw_file = source / "2021_08_02.csv"
    raw_file.write_text(
        "timestamp,message_id,raw_message\n"
        f'2021-08-02T00:00:00.000Z,1,"{dynamic_message}"\n', encoding="utf-8"
    )
    output_file = destination / raw_file.name
    output_file.write_text("existing output\n", encoding="utf-8")

    pytsa.decode(source, destination, njobs=1)
    assert output_file.read_text(encoding="utf-8") == "existing output\n"

    pytsa.decode(source, destination, njobs=1, overwrite=True)
    assert pd.read_csv(output_file).loc[0, "msg_type"] == 1


@pytest.mark.parametrize(
    ("source_is_directory", "destination_is_directory", "message"),
    [
        (False, True, "Source must be a directory"),
        (True, False, "Destination must be a directory"),
    ],
)
def test_decode_requires_directory_paths(
    tmp_path,
    source_is_directory: bool,
    destination_is_directory: bool,
    message: str,
) -> None:
    source_path = tmp_path / "source"
    destination_path = tmp_path / "destination"
    for path, is_directory in (
        (source_path, source_is_directory),
        (destination_path, destination_is_directory),
    ):
        if is_directory:
            path.mkdir()
        else:
            path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(AssertionError, match=message):
        pytsa.decode(source_path, destination_path)