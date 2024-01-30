import pyais as ais
import pandas as pd
import unittest
import pytsa
from io import StringIO
from pytsa.decode.ais_decoder import (
    MSG12318SLOTS, MSG5SLOTS
)


class TestAISDecode(unittest.TestCase):

    # Test messages
    msg = r"!ABVDM,1,1,,B,177PhT001iPWhwJPsK=9DoQH0<>i,0*7C"
    msg5 = [
        r"!ABVDM,2,1,5,A,53aQ5aD2;PAQ0@8l000lE9LD8u8L00000000001??H<886?80@@C1F0CQ4R@,0*35",
        r"!ABVDM,2,2,5,A,@0000000000,2*5A" 
    ]

    # Dummy data set
    dyndata = StringIO(
    f"""timestamp,message_id,raw_message
    2021-08-02T00:00:00.000Z,1,"{msg}"
    """)
    
    statdata = StringIO(
    f"""timestamp,message_id,raw_message1,raw_message2
    2021-08-02T00:00:00.000Z,5,"{msg5[0]}","{msg5[1]}"
    """)

    def test_raw_msg1_decode(self):
        """
        Test decoding of message type 1
        """
        self.assertEqual(
            ais.decode(self.msg).msg_type, 
            1, 
            "Message type should be 1"
        )
        
    def test_raw_msg5_decode(self):
        """
        Test decoding of message type 5
        """
        self.assertEqual(
            ais.decode(*self.msg5).msg_type, 
            5, 
            "Message type should be 5"
        )
        
    def test_decode_dynamic_from_buffer(self):
        """
        Test if decoding from buffer works and 
        extracts all message fields/slots
        """
        df: pd.DataFrame = pytsa.decode_from_file(
            source=self.dyndata,
            dest=None,
            save_to_file=False
        )
        # Raw message columns that are not decoded
        # but are prepended to the decoded columns
        # in the DataFrame for completeness
        _sticky = {
            "timestamp",
            "message_id",
            "raw_message",
            "DECODE_START",
        }

        # Check if decoded columns
        # match the expected columns
        self.assertEqual(
            set(df.columns).difference(_sticky), 
            set(MSG12318SLOTS),
            "Decoded columns do not match expected columns"
        )

    def test_decode_static_from_buffer(self):
        """
        Test if decoding from buffer works and 
        extracts all message fields/slots
        """
        df: pd.DataFrame = pytsa.decode_from_file(
            source=self.statdata,
            dest=None,
            save_to_file=False
        )
        # Raw message columns that are not decoded
        # but are prepended to the decoded columns
        # in the DataFrame for completeness
        _sticky = {
            "timestamp",
            "message_id",
            "raw_message1",
            "raw_message2",
            "DECODE_START",
        }

        # Check if decoded columns
        # match the expected columns
        self.assertEqual(
            set(df.columns).difference(_sticky), 
            set(MSG5SLOTS),
            "Decoded columns do not match expected columns"
        )
    
if __name__ == "__main__":
    unittest.main()