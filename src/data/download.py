from pathlib import Path
import zipfile

import requests
from ..config import cfg

def download_and_extract(url, extract_to):
    """Tải và giải nén file từ URL vào thư mục chỉ định."""
    file_name = url.split('/')[-1]
    save_path = Path(extract_to)
    save_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = save_path / file_name

    # 1. Tải file
    if not zip_path.exists():
        print(f"Downloading {file_name}...")
        partial_path = zip_path.with_suffix(zip_path.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                with partial_path.open('wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            partial_path.replace(zip_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise
        print("Download complete.")
    else:
        print(f"{file_name} đã tồn tại, bỏ qua bước tải.")

    # 2. Giải nén
    print(f"Extracting {file_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        root = save_path.resolve()
        for member in zip_ref.infolist():
            destination = (root / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        zip_ref.extractall(save_path)
    
    # 3. Xóa file zip sau khi giải nén để tiết kiệm dung lượng
    zip_path.unlink()
    print(f"Done! Files extracted to {extract_to}")

def prepare_data():
    """Hàm chính để chuẩn bị dữ liệu cho toàn dự án."""
    base_data_path = cfg.DATA_DIR
    
    # Link dữ liệu
    input_url = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip"
    truth_url = "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip"
    
    # Tải và giải nén
    download_and_extract(input_url, base_data_path)
    download_and_extract(truth_url, base_data_path)
    
    # Lưu ý: File CSV sau khi giải nén từ ISIC2018_Task3_Training_GroundTruth.zip 
    # thường nằm trong một thư mục con, bạn có thể cần dùng os.rename để đưa nó ra đúng vị trí mong muốn.
    print("Dữ liệu đã sẵn sàng!")

def main():
    print("Bắt đầu chuẩn bị dữ liệu...")
    prepare_data()
    print("Hoàn tất!")

if __name__ == "__main__":
    main()
