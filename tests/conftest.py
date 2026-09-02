from io import StringIO

import pytest


DYNAMIC_MESSAGE = "!ABVDM,1,1,,B,177PhT001iPWhwJPsK=9DoQH0<>i,0*7C"
STATIC_MESSAGES = (
    "!ABVDM,2,1,5,A,53aQ5aD2;PAQ0@8l000lE9LD8u8L00000000001??H<886?80@@C1F0CQ4R@,0*35",
    "!ABVDM,2,2,5,A,@0000000000,2*5A",
)


@pytest.fixture
def dynamic_message() -> str:
    return DYNAMIC_MESSAGE


@pytest.fixture
def static_messages() -> tuple[str, str]:
    return STATIC_MESSAGES


@pytest.fixture
def dynamic_csv(dynamic_message: str) -> StringIO:
    return StringIO(
        "timestamp,message_id,raw_message\n"
        f'2021-08-02T00:00:00.000Z,1,"{dynamic_message}"\n'
    )


@pytest.fixture
def static_csv(static_messages: tuple[str, str]) -> StringIO:
    first, second = static_messages
    return StringIO(
        "timestamp,message_id,raw_message1,raw_message2\n"
        f'2021-08-02T00:00:00.000Z,5,"{first}","{second}"\n'
    )