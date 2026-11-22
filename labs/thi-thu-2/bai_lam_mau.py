"""Khung bài làm Phần C, Đề thi thử 2.

Chép file này thành `bai_lam.py` ở thư mục gốc của sách rồi điền hai hàm trong giờ làm Phần C.
Chỉ dùng NumPy. C1: không vòng lặp Python theo mẫu trong batch hay theo vị trí (token).
C2: không vòng lặp Python theo câu test (vòng lặp theo lần bootstrap thì được phép).
Được chạy thử với dữ liệu tự nghĩ ra (viết ở khối `if __name__ == "__main__":` bên dưới).

Sau khi hết giờ mới chấm:
    python labs/thi-thu-2/cham_phan_c.py --student bai_lam.py
"""
import numpy as np


def causal_attention(X, Wq, Wk, Wv):
    """Self-attention một head có mặt nạ nhân quả.
    X: mảng (B, T, d) float64. Wq, Wk: (d, dk). Wv: (d, dv).
    Q = X Wq, K = X Wk, V = X Wv (nhân theo từng mẫu trong batch).
    Trả về (O, A):
      A: mảng (B, T, T); A[b, i, j] là trọng số query i dành cho key j, bằng 0 khi j > i,
         mỗi hàng cộng bằng 1 (softmax của q_i·k_j/√dk trên các j ≤ i);
      O: mảng (B, T, dv) = A V.
    dk là số cột của Wq. Giữ float64. Phải đúng cả khi q·k/√dk cỡ ±10⁴ (không inf, không nan)."""
    # TODO
    raise NotImplementedError


def paired_bootstrap_ci(correct_a, correct_b, n_boot=10000, seed=0):
    """correct_a, correct_b: mảng (n,) kiểu int gồm 0 và 1; phần tử i cho biết mô hình A (hoặc B)
    trả lời đúng câu test i hay không. Hai mô hình được chấm trên CÙNG n câu.
    Trả về (diff, lo, hi), ba số thực:
      diff = accuracy(A) − accuracy(B), dạng tỉ lệ (0.012, không phải 1.2);
      [lo, hi] = khoảng tin cậy 95% của diff bằng paired bootstrap phân vị: n_boot lần, mỗi lần
                 lấy n chỉ số câu CÓ HOÀN LẠI và dùng cùng các chỉ số đó cho A và B.
    Dùng rng = np.random.default_rng(seed) cho mọi phép ngẫu nhiên: cùng seed phải cho cùng kết quả."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # Chỗ để bạn tự thử với dữ liệu nhỏ, ví dụ:
    # X = np.zeros((1, 3, 2)); print(causal_attention(X, np.eye(2), np.eye(2), np.eye(2))[1])
    # print(paired_bootstrap_ci(np.array([1, 1, 0, 1]), np.array([1, 0, 0, 1]), 1000, 0))
    pass
