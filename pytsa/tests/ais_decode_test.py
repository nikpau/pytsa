import pyais as ais

# Type 
msg = r"!AIVDM,1,1,,B,10?Ej=P00tPwi3RNn`supK602@0K,0*70"
msg5 = [
    r"!ABVDM,2,1,5,A,53aQ5aD2;PAQ0@8l000lE9LD8u8L00000000001??H<886?80@@C1F0CQ4R@,0*35",
    r"!ABVDM,2,2,5,A,@0000000000,2*5A"
    
]

def test_decode():
    assert ais.decode(msg).msg_type == 1
    assert ais.decode(*msg5).msg_type == 5
    
if __name__ == "__main__":
    test_decode()
    print("All tests passed!")