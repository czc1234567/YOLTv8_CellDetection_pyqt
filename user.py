from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import json
import base64

import time
class My_AES_CBC():
    def __init__(self, key, iv):
        # key 和 iv 必须为 16 位
        self.key = key
        self.mode = AES.MODE_CBC
        self.cryptor = AES.new(self.key, self.mode, iv)

    def encrypt(self, plain_text):
        encode_text = plain_text.encode('utf-8')
        pad_text = pad(encode_text, AES.block_size)
        encrypted_text = self.cryptor.encrypt(pad_text)
        return encrypted_text

    def decrypt(self, encrypted_text):
        plain_text = self.cryptor.decrypt(encrypted_text)
        plain_text = unpad(plain_text, AES.block_size).decode()
        return plain_text


Aes_key = '9B8FD68A366F4D03'.encode()
Aes_IV = '305FB72D83134CA0'.encode('utf-8')

def getActiveCode(machine_code):
    """
    用于通过机器码，生成激活码
    machine_code: 机器码
    """
    encrypt_code = My_AES_CBC(Aes_key, Aes_IV).encrypt(machine_code)
    active_code = hashlib.md5(encrypt_code).hexdigest().upper()
    return active_code

def getTimeLimitedCode(machine_code, timestamp):
    """
    用于通过机器码和有效期时间戳，生成限时授权码
    machine_code: 机器码
    timestamp: 有效期的时间戳
    """
    # 生成激活码
    active_code = getActiveCode(machine_code)
    # 组合成 json 格式，并转换成字符串
    data = {
        "code": active_code,
        "endTs": timestamp,
    }
    text = json.dumps(data)
    # AES 加密
    encrypt_code = My_AES_CBC(Aes_key, Aes_IV).encrypt(text)
    # base64 加密，生成授权码
    active_code = base64.b32encode(encrypt_code)
    return active_code.decode()


# AES_CBC 加密
def Encrypt(plain_text):
    e = My_AES_CBC(Aes_key, Aes_IV).encrypt(plain_text)
    return e


# AES_CBC 解密
def Decrypt(encrypted_text):
    d = My_AES_CBC(Aes_key, Aes_IV).decrypt(encrypted_text)
    return d


def checkKeyCode(machine_code, key_code):
    # 授权码解密，提取出激活码和有效期
    register_str = base64.b32decode(key_code)
    decode_key_data = json.loads(Decrypt(register_str))
    active_code = decode_key_data["code"].upper()  # 激活码
    end_timestamp = decode_key_data["endTs"]  # 有效期

    # 加密机器码，用于跟激活码对比
    encrypt_code = Encrypt(machine_code)
    md5_code = hashlib.md5(encrypt_code).hexdigest().upper()

    # 获取本地时间，用于跟有效期对比
    curTs = int(time.time())

    if md5_code != active_code:
        print("激活码错误，请重新输入！")
    elif curTs >= end_timestamp:
        print("激活码已过期，请重新输入！")
    else:
        time_local = time.localtime(end_timestamp)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
        print("激活成功！有效期至 %s" % dt)


if __name__ == '__main__':
    # 机器码
    machine_code = "7BA4E7D187C65A2EBE993C8257743487"
    # 授权码
    key_code = "MCDJE3GV5C23LWNNO7WQYEOZ5SW4CKSG5M6JOJVXJNL5CRBNEWMOQKR3FIFVGTQVIDB7WWHLK47DBGV5FPMKKDNA2GHD6HETRTBFF6W23SHRQY4O26LLV7I7BCTH5PAR"
    # 授权校验
    checkKeyCode(machine_code, key_code)