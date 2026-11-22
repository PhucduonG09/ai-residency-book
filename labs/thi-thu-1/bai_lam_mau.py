"""Khung bài làm Phần C, Đề thi thử 1.

Chép file này thành `bai_lam.py` ở thư mục gốc của sách rồi điền hai hàm trong giờ làm Phần C.
Chỉ dùng NumPy, không vòng lặp Python theo mẫu hay theo lớp. Được chạy thử với dữ liệu tự nghĩ ra
(viết ở khối `if __name__ == "__main__":` bên dưới).

Sau khi hết giờ mới chấm:
    python labs/thi-thu-1/cham_phan_c.py --student bai_lam.py
"""
import numpy as np


def softmax_cross_entropy(Z, y):
    """Z: mảng (N, K) logit, float64. y: mảng (N,) nhãn nguyên trong [0, K).
    Trả về (loss, dZ):
      loss: số vô hướng, cross-entropy TRUNG BÌNH trên N mẫu;
      dZ:   mảng (N, K) = ∂loss/∂Z.
    Giữ float64. Không được sửa Z tại chỗ. Phải đúng cả khi logit cỡ ±1000."""
    # TODO
    raise NotImplementedError


def logreg_step(w, b, X, y, lr, lam):
    """w: (d,), b: float, X: (N, d), y: (N,) nhãn 0/1, lr: learning rate, lam: λ.
    Hàm mục tiêu J(w, b) = mean(ln(1 + e^z) − y·z) + (λ/2)‖w‖², z = Xw + b, không phạt b.
    Trả về (w_moi, b_moi, J) với J tính tại (w, b) CŨ, sau đúng một bước gradient descent.
    Giữ float64. Phải đúng cả khi |z| cỡ 800 (không inf, không nan)."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # Chỗ để bạn tự thử với dữ liệu nhỏ, ví dụ:
    # print(softmax_cross_entropy(np.array([[0.0, np.log(3.0)]]), np.array([1])))
    pass
