# code này dùng để biến file ONNX thành dạng “bí mật” để bảo vệ (encrypt: hoạt động khóa hộp)
# khi muốn sử dụng thì khôi phục lại file gốc để sử dụng (decrypt: hoạt động mở hộp)
# mục đích: tránh bị người ngoài lấy cắp mô hình khi chia sẻ hoặc deploy 
# Thuật toán mã hóa được sử dụng là AES (Advanced Encryption Standard) - có thể hiểu là cái ổ khóa
# AES cần một KEY_HEX có độ dài chuẩn (ví dụ AES-256 cần 32 bytes)
# KEY là một chuỗi biểu diễn dưới dạng hexa, quyết định việc mã hóa/giải mã, nếu không có đúng key thì không thể mở được dữ liệu.
# IV_HEX (Initialization Vector): giống như số ngẫu nhiên để đảm bảo mỗi lần khóa thì kết quả khác nhau, tránh bị đoán.
# unhexlify(KEY_HEX) và unhexlify(IV_HEX): Vì AES cần dữ liệu dạng bytes, nên ta phải chuyển từ chuỗi hex sang bytes.
# pad/unpad: vì dữ liệu không phải lúc nào cũng vừa khít với ổ khóa, nên phải “đệm thêm” cho đủ chỗ. Khi mở khóa thì bỏ phần đệm đi.

"""Historical, incomplete encryption example; not used by the classifier.

The empty key/IV must be configured before use. Do not treat the static-IV CBC
example as a deployment security implementation. External ONNX weight files
also need handling before encrypted deployment can work.
"""

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from binascii import unhexlify

class ONNXEncryptor:
    # Static HEX key and IV (AES-256)
    KEY_HEX = ""   # mấy cái này tạo sau, đây là code mẫu nên không có key nào cả
    IV_HEX = ""

    # Byte values
    KEY = unhexlify(KEY_HEX)
    IV = unhexlify(IV_HEX)

    @staticmethod
    def encrypt(input_path="model.onnx", output_path="model.enc") -> bool:
        try:
            with open(input_path, "rb") as f:
                data = f.read()     # mở file onnx và đọc dữ liệu gốc, Tạo đối tượng AES với key và IV đã định nghĩa, ở chế độ CBC.
            cipher = AES.new(ONNXEncryptor.KEY, AES.MODE_CBC, ONNXEncryptor.IV)     
            padded_data = pad(data, AES.block_size)     # Pad dữ liệu để khớp với block size (AES block size = 16 bytes).
            encrypted = cipher.encrypt(padded_data)     # Mã hóa dữ liệu padded.
            with open(output_path, "wb") as f:
                f.write(encrypted)      # Ghi dữ liệu mã hóa ra file model.enc
                
            return True
        except Exception as e:
            return False

    @staticmethod
    def decrypt(input_path="model.enc", output_path="model_decrypted.onnx") -> bool:
        try:
            with open(input_path, "rb") as f:
                encrypted = f.read()        # Mở file mã hóa (model.enc) và đọc dữ liệu. Tạo đối tượng AES ...
            cipher = AES.new(ONNXEncryptor.KEY, AES.MODE_CBC, ONNXEncryptor.IV)
            padded_raw_data = cipher.decrypt(encrypted)         # Giải mã dữ liệu
            decrypted = unpad(padded_raw_data, AES.block_size)    # unpad để loại bỏ padding đã thêm lúc mã hóa.
            with open(output_path, "wb") as f:
                f.write(decrypted)      # Ghi dữ liệu đã giải mã ra file model_decrypted.onnx

            return True
        except Exception as e:
            return False
